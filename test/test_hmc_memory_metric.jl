# Tests for the memory covariance metric (Stage 2 of ext/AdvancedHMCExt.jl):
# `memory_metric()` estimates the HMC mass matrix from the sampler's memory archive instead of
# AdvancedHMC's windowed adaptor. It requires memory and is rejected under parallel tempering
# (annealing is fine). `CorrelatedGaussianModel`, `hmc_adaptive_state`, and `backwards_compat_rng`
# come from test_hmc.jl / runtests.jl.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag, Diagonal

@testset "HMC memory metric (AdvancedHMCExt Stage 2)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, raw_model))

    mem_update(; kwargs...) = DifferentialEvolutionMetropolis.setup_hmc_update(
        NUTS(0.8); n_dims = length(μ),
        metric_strategy = memory_metric(; kwargs...)
    )

    @testset "constructor validates its arguments" begin
        @test_throws ErrorException memory_metric(; shrinkage = -0.1)
        @test_throws ErrorException memory_metric(; shrinkage = 1.5)
        @test_throws ErrorException memory_metric(; every = 0)
        # A valid call returns a strategy that `setup_hmc_update` accepts without error.
        strat = memory_metric(; shrinkage = 0.1, every = 5)
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8); n_dims = 1, metric_strategy = strat
        ).metric_strategy === strat
    end

    @testset "recovers a unimodal correlated Gaussian (memory on)" begin
        scheme = setup_sampler_scheme(mem_update())
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
        scheme = setup_sampler_scheme(mem_update())
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

    @testset "the metric keeps tracking the archive during sampling" begin
        # The metric is NOT frozen at the warmup→sampling boundary: it keeps recomputing from the
        # growing memory archive during sampling, on the same `every` cadence.
        rng = backwards_compat_rng(77)
        scheme = setup_sampler_scheme(mem_update(; every = 1))
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = true, silent = true
        )
        for _ in 1:50
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        end
        a = hmc_adaptive_state(state)
        steps_after_warmup = a.metric_steps
        metric_after_warmup = copy(a.metric.M⁻¹)

        for _ in 1:60
            _, state = AbstractMCMC.step(rng, model, scheme, state)
        end
        a = hmc_adaptive_state(state)
        # The cadence counter advanced through the sampling steps (a frozen metric would not move)
        @test a.metric_steps == steps_after_warmup + 60
        # and the live metric was recomputed, still tracking the marginal variances.
        @test a.metric.M⁻¹ != metric_after_warmup
        @test isapprox(a.metric.M⁻¹, diag(Σ); atol = 0.3)
    end

    @testset "running memoryless is rejected on the first warmup step" begin
        # memory_metric reads the memory archive, so a memoryless scheme has nothing to estimate
        # from. The check fires once, on the first warmup step, after a clean init.
        rng = backwards_compat_rng(7)
        scheme = setup_sampler_scheme(mem_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = false, silent = true
        )
        @test_throws ErrorException AbstractMCMC.step_warmup(
            rng, model, scheme, state; num_warmup = 50
        )
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
            mksampler(DiagEuclideanMetric(d)); metric_strategy = memory_metric()
        )
        @test diag_accepted.metric isa DiagEuclideanMetric

        dense_accepted = DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DenseEuclideanMetric(d)); metric_strategy = memory_metric()
        )
        @test dense_accepted.metric isa DenseEuclideanMetric

        # A `UnitEuclideanMetric` has no estimable mass matrix, so it must be rejected.
        @test_throws ErrorException DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(UnitEuclideanMetric(d)); metric_strategy = memory_metric()
        )
    end

    @testset "a dense metric recovers the full population covariance (off-diagonals)" begin
        # The payoff of a dense population metric: M⁻¹ approaches the *full* covariance Σ, so the
        # off-diagonal correlations (which a diagonal metric discards) are recovered in sign and
        # magnitude.
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(
                NUTS(0.8; metric = :dense); n_dims = length(μ), metric_strategy = memory_metric()
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
            NUTS(0.8; metric = :dense); n_dims = length(μ), metric_strategy = memory_metric()
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
        scheme = setup_sampler_scheme(mem_update(; every = 10))
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
        # `every` far larger than num_warmup: the metric is computed exactly once, on the first
        # step, and never again. We seed the population from the target so that one-shot estimate
        # is informative — the test then checks the metric is installed (off the placeholder
        # identity) AND recovers the marginals, rather than silently keeping the identity.
        rng = backwards_compat_rng(5)
        init = [rand(Xoshiro(100 + i), MvNormal(μ, Σ)) for i in 1:24]   # 8 chains + 16 memory, from target
        scheme = setup_sampler_scheme(mem_update(; every = 10_000))
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = true,
            initial_position = init, silent = true
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
                NUTS(0.8); n_dims = 3, metric_strategy = memory_metric(; shrinkage = 0.5)
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

    @testset "static parallel tempering is rejected on the first warmup step" begin
        # Under parallel tempering the memory archive interleaves hot-chain positions, so the
        # covariance would mix temperatures. memory_metric refuses on the first warmup step.
        rng = backwards_compat_rng(123)
        ncold, nhot = 6, 4
        scheme = setup_sampler_scheme(mem_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = ncold, n_hot_chains = nhot,
            num_warmup = 50, memory = true, silent = true,
            initial_position = [randn(rng, length(μ)) for _ in 1:(ncold + nhot)]
        )
        @test state.temperature_ladder isa
            DifferentialEvolutionMetropolis.DifferentialEvolutionStaticTemperatureLadder
        @test_throws ErrorException AbstractMCMC.step_warmup(
            rng, model, scheme, state; num_warmup = 50
        )
    end

    @testset "annealing alongside persistent hot chains is rejected" begin
        # An annealing ladder can still parallel-temper: chains hot at the end stay out of
        # `cold_chains`, so the rejection keys off the cold/total count, not the ladder type.
        rng = backwards_compat_rng(124)
        ncold, nhot = 6, 4
        scheme = setup_sampler_scheme(mem_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = ncold, n_hot_chains = nhot, annealing_steps = 5,
            num_warmup = 50, memory = true, silent = true,
            initial_position = [randn(rng, length(μ)) for _ in 1:(ncold + nhot)]
        )
        @test state.temperature_ladder isa
            DifferentialEvolutionMetropolis.DifferentialEvolutionAnnealingTemperatureLadder
        @test length(state.temperature_ladder.cold_chains) < length(state.x)
        @test_throws ErrorException AbstractMCMC.step_warmup(
            rng, model, scheme, state; num_warmup = 50
        )
    end

    @testset "pure annealing (every chain cools) is accepted" begin
        rng = backwards_compat_rng(125)
        init = [rand(Xoshiro(200 + i), MvNormal(μ, Σ)) for i in 1:24]
        scheme = setup_sampler_scheme(mem_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, annealing_steps = 5,
            num_warmup = 50, memory = true, silent = true, initial_position = init
        )
        @test state.temperature_ladder isa
            DifferentialEvolutionMetropolis.DifferentialEvolutionAnnealingTemperatureLadder
        @test length(state.temperature_ladder.cold_chains) == length(state.x)
        for _ in 1:30
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DiagEuclideanMetric
        @test all(isfinite, a.metric.M⁻¹)
        @test a.metric.M⁻¹ != ones(length(μ))
    end

    @testset "parallel tempering: HMC advances only the cold chains (stock metric)" begin
        # The HMC kernel samples the untempered target, so it must not move the hot chains (they
        # mix through the tempered DE updates instead). memory_metric is rejected under tempering,
        # so this run_trajectories! regression uses the default stock-adaptor metric, which does
        # support tempering. We check the hot chains are bit-for-bit unchanged and the cold move.
        rng = backwards_compat_rng(123)
        ncold, nhot = 6, 4
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ))
        )
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = ncold, n_hot_chains = nhot,
            num_warmup = 50, memory = false, silent = true,
            initial_position = [randn(rng, length(μ)) for _ in 1:(ncold + nhot)]
        )
        cold = state.temperature_ladder.cold_chains
        hot = setdiff(1:length(state.x), cold)
        @test length(cold) == ncold && length(hot) == nhot
        hot_before = [copy(state.x[i]) for i in hot]
        cold_before = [copy(state.x[i]) for i in cold]
        for _ in 1:20
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        end
        # Hot chains untouched by HMC; cold chains have moved.
        @test all(state.x[i] == hot_before[k] for (k, i) in enumerate(hot))
        @test any(state.x[i] != cold_before[k] for (k, i) in enumerate(cold))
    end

    @testset "threaded trajectory loop matches the serial result statistically" begin
        # The diagonal metric is read-only across threads, so the parallel run must agree with a
        # same-seed serial run (a race would perturb one of them). The dense variant — which does
        # need private scratch — gets its own equivalence check above.
        runit(parallel) = reshape(
            sample(
                backwards_compat_rng(13), model, setup_sampler_scheme(mem_update()), 800;
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
