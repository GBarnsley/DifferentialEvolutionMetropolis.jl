# Tests for the population/memory covariance metric (Stage 2 of ext/AdvancedHMCExt.jl):
# `population_metric()` estimates the HMC mass matrix from the sampler's own population
# instead of AdvancedHMC's windowed adaptor. `CorrelatedGaussianModel`, `hmc_adaptive_state`,
# and `backwards_compat_rng` come from test_hmc.jl / runtests.jl.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag, Diagonal

@testset "HMC population/memory metric (AdvancedHMCExt Stage 2)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, raw_model))

    pop_update(; kwargs...) = DifferentialEvolutionMetropolis.setup_hmc_update(
        NUTS(0.8); n_dims = length(μ),
        metric_strategy = population_metric(; kwargs...)
    )

    @testset "constructor validates its arguments" begin
        @test_throws ErrorException population_metric(; shrinkage = -0.1)
        @test_throws ErrorException population_metric(; shrinkage = 1.5)
        @test_throws ErrorException population_metric(; every = 0)
        # A valid call returns a strategy that `setup_hmc_update` accepts without error.
        strat = population_metric(; shrinkage = 0.1, every = 5)
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8); n_dims = 1, metric_strategy = strat
        ).metric_strategy === strat
    end

    @testset "recovers a unimodal correlated Gaussian (memory on)" begin
        scheme = setup_sampler_scheme(pop_update())
        out = sample(
            backwards_compat_rng(42), model, scheme, 800;
            n_chains = 6, num_warmup = 300, memory = true, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.15)
        @test isapprox(cov(flat), Σ; atol = 0.25)
    end

    @testset "the frozen diagonal metric tracks the marginal variances" begin
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(pop_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 300, memory = true, silent = true
        )
        for _ in 1:300
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 300)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DiagEuclideanMetric
        # M⁻¹ = diag(Σ): the population covariance's diagonal should approach the marginals,
        # in both value and ordering (the middle coordinate is the most variable).
        @test isapprox(a.metric.M⁻¹, diag(Σ); atol = 0.3)
        @test argmax(a.metric.M⁻¹) == argmax(diag(Σ))
        @test 0 < a.integrator.ϵ < 10
    end

    @testset "memory-off fallback runs without crashing" begin
        scheme = setup_sampler_scheme(pop_update(; shrinkage = 0.3))
        out = sample(
            backwards_compat_rng(7), model, scheme, 600;
            n_chains = 8, num_warmup = 300, memory = false, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        @test all(isfinite, flat)
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.3)
    end

    @testset "accepts both diagonal and dense metrics, rejects others, at setup time" begin
        # Validation fires when the update is constructed, before any sampling. Both Euclidean
        # metrics the strategy can populate are accepted; an unsupported metric is rejected.
        d = length(μ)
        integrator = Leapfrog(0.1)
        κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
        mksampler(metric) = HMCSampler(
            κ, metric, StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        )

        diag_accepted = DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DiagEuclideanMetric(d)); metric_strategy = population_metric()
        )
        @test diag_accepted.metric isa DiagEuclideanMetric

        dense_accepted = DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DenseEuclideanMetric(d)); metric_strategy = population_metric()
        )
        @test dense_accepted.metric isa DenseEuclideanMetric

        # A `UnitEuclideanMetric` has no estimable mass matrix, so it must be rejected.
        @test_throws ErrorException DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(UnitEuclideanMetric(d)); metric_strategy = population_metric()
        )
    end

    @testset "a dense metric recovers the full population covariance (off-diagonals)" begin
        # The payoff of a dense population metric: M⁻¹ approaches the *full* covariance Σ, so the
        # off-diagonal correlations (which a diagonal metric discards) are recovered in sign and
        # magnitude.
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(
                NUTS(0.8; metric = :dense); n_dims = length(μ), metric_strategy = population_metric()
            )
        )
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 300, memory = true, silent = true
        )
        for _ in 1:300
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 300)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DenseEuclideanMetric
        @test isapprox(a.metric.M⁻¹, Σ; atol = 0.3)
        # The strongest correlation (coords 1,2: +0.6) must be recovered with the right sign.
        @test a.metric.M⁻¹[1, 2] > 0.25
        @test 0 < a.integrator.ϵ < 10
    end

    @testset "a dense population metric recovers the target and is thread-safe" begin
        # End-to-end with a dense metric, serial vs parallel: the private-scratch copies must make
        # the threaded run agree with the same-seed serial run (a `_temp` race would perturb one).
        dense_update() = DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8; metric = :dense); n_dims = length(μ), metric_strategy = population_metric()
        )
        runit(parallel) = reshape(
            sample(
                backwards_compat_rng(13), model, setup_sampler_scheme(dense_update()), 800;
                n_chains = 6, num_warmup = 300, parallel = parallel, memory = true,
                progress = false, chain_type = DifferentialEvolutionOutput
            ).samples, :, length(μ)
        )
        serial = runit(false)
        threaded = runit(true)
        @test all(isfinite, threaded)
        @test isapprox(vec(sum(serial; dims = 1) ./ size(serial, 1)), μ; atol = 0.2)
        @test isapprox(cov(serial), Σ; atol = 0.3)
        @test threaded == serial
    end

    @testset "the recompute schedule (`every`) installs the metric and recovers the target" begin
        scheme = setup_sampler_scheme(pop_update(; every = 10))
        out = sample(
            backwards_compat_rng(11), model, scheme, 800;
            n_chains = 6, num_warmup = 400, memory = true, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.2)
    end

    @testset "the metric is installed even when `every` exceeds the warmup budget" begin
        # `every` far larger than num_warmup: the first-step compute must still install the
        # population covariance, rather than silently leaving the placeholder identity metric.
        rng = backwards_compat_rng(5)
        scheme = setup_sampler_scheme(pop_update(; every = 10_000))
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = true, silent = true
        )
        for _ in 1:50
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DiagEuclideanMetric
        # The placeholder metric is all-ones; a real covariance estimate has moved off it.
        @test a.metric.M⁻¹ != ones(length(μ))
        @test isapprox(a.metric.M⁻¹, diag(Σ); atol = 0.5)
    end

    @testset "ill-conditioned target stays finite (shrinkage conditions the metric)" begin
        # One nearly-degenerate direction: variance 1e-3 against unit scales.
        Σ_ill = Diagonal([1.0, 1.0e-3, 1.0])
        ill_model = AbstractMCMC.LogDensityModel(
            ADgradient(:ForwardDiff, CorrelatedGaussianModel(MvNormal(zeros(3), Matrix(Σ_ill))))
        )
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(
                NUTS(0.8); n_dims = 3, metric_strategy = population_metric(; shrinkage = 0.5)
            )
        )
        out = sample(
            backwards_compat_rng(3), ill_model, scheme, 600;
            n_chains = 8, num_warmup = 300, memory = true, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, 3)
        @test all(isfinite, flat)
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, zeros(3); atol = 0.3)
    end

    @testset "threaded trajectory loop matches the serial result statistically" begin
        # The diagonal metric is read-only across threads, so the parallel run must agree with a
        # same-seed serial run (a race would perturb one of them). The dense variant — which does
        # need private scratch — gets its own equivalence check above.
        runit(parallel) = reshape(
            sample(
                backwards_compat_rng(13), model, setup_sampler_scheme(pop_update()), 800;
                n_chains = 6, num_warmup = 300, parallel = parallel, memory = true,
                progress = false, chain_type = DifferentialEvolutionOutput
            ).samples, :, length(μ)
        )
        serial = runit(false)
        threaded = runit(true)
        @test all(isfinite, threaded)
        serial_mean = vec(sum(serial; dims = 1) ./ size(serial, 1))
        threaded_mean = vec(sum(threaded; dims = 1) ./ size(threaded, 1))
        @test isapprox(serial_mean, μ; atol = 0.15)
        @test isapprox(threaded_mean, serial_mean; atol = 0.1)
    end
end
