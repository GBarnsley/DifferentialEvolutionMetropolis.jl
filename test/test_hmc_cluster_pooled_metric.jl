# Tests for the within-cluster-pooled covariance metric (Stage 3 of ext/AdvancedHMCExt.jl):
# `cluster_pooled_metric()` clusters the memory archive, centres each point on its cluster
# mean, and pools the centred deviations into one metric — stripping the between-mode
# inflation that defeats `memory_metric()` on separated modes. It gates on a cheap
# multimodality test so it degrades to `memory_metric()` when the target is unimodal, and it
# carries the same memory/no-tempering requirements. `CorrelatedGaussianModel`,
# `hmc_adaptive_state`, and `backwards_compat_rng` come from test_hmc.jl / runtests.jl.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag, Diagonal

# Internal helpers (gate, clustering, estimators) live in the package extension.
const HMCExt = Base.get_extension(DifferentialEvolutionMetropolis, :AdvancedHMCExt)

@testset "HMC cluster-pooled metric (AdvancedHMCExt Stage 3)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, raw_model))

    cluster_update(; kwargs...) = DifferentialEvolutionMetropolis.setup_hmc_update(
        NUTS(0.8); n_dims = length(μ),
        metric_strategy = cluster_pooled_metric(; kwargs...)
    )

    @testset "constructor validates its arguments" begin
        @test_throws ErrorException cluster_pooled_metric(; shrinkage = -0.1)
        @test_throws ErrorException cluster_pooled_metric(; shrinkage = 1.5)
        @test_throws ErrorException cluster_pooled_metric(; every = 0)
        @test_throws ErrorException cluster_pooled_metric(; kmax = 0)
        strat = cluster_pooled_metric(; shrinkage = 0.1, every = 5, kmax = 4)
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8); n_dims = 1, metric_strategy = strat
        ).metric_strategy === strat
    end

    @testset "rejects a unit metric at setup, accepts diagonal and dense" begin
        d = length(μ)
        integrator = Leapfrog(0.1)
        κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
        mksampler(metric) = HMCSampler(
            κ, metric, StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        )
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DiagEuclideanMetric(d)); metric_strategy = cluster_pooled_metric()
        ).metric isa DiagEuclideanMetric
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DenseEuclideanMetric(d)); metric_strategy = cluster_pooled_metric()
        ).metric isa DenseEuclideanMetric
        @test_throws ErrorException DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(UnitEuclideanMetric(d)); metric_strategy = cluster_pooled_metric()
        )
    end

    @testset "prepare_sampling_metrics! resolves on an unwarmed (astate === nothing) sampler" begin
        # The fixed sampler carries the live adaptive state; before warmup it is `nothing`, and the
        # `Nothing` method must out-specialise the general one (it would read `astate.chain_metrics`).
        s = DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ))
        @test s.astate === nothing
        κ, cm = HMCExt.prepare_sampling_metrics!(s, s.astate, nothing)
        @test κ === s.κ
        @test cm === s.chain_metrics
    end

    @testset "the gate fires on separated modes, not on a unimodal archive" begin
        rng = Xoshiro(7)
        unimodal = [randn(rng, 2) for _ in 1:500]
        @test HMCExt.multimodal_gate(unimodal) == false

        bimodal = vcat(
            [[-8.0, -8.0] .+ 0.3 .* randn(rng, 2) for _ in 1:250],
            [[8.0, 8.0] .+ 0.3 .* randn(rng, 2) for _ in 1:250],
        )
        @test HMCExt.multimodal_gate(bimodal) == true
    end

    @testset "within-cluster pooling strips between-mode inflation" begin
        # Two well-separated modes, each with small within-mode variance. The raw memory
        # (pooled) covariance is dominated by the ~16-unit separation; the within-cluster
        # estimate must instead recover the small within-mode variance.
        rng = Xoshiro(11)
        within_sd = 0.4
        archive = vcat(
            [[-8.0, 5.0] .+ within_sd .* randn(rng, 2) for _ in 1:300],
            [[8.0, -5.0] .+ within_sd .* randn(rng, 2) for _ in 1:300],
        )
        metric = DiagEuclideanMetric(2)

        pooled = HMCExt.memory_inverse_metric(metric, archive, 0.0)
        labels, centers = HMCExt.cluster_archive(archive, 10, Matrix{Float64}(undef, 2, 0))
        @test size(centers, 2) == 2
        within = HMCExt.within_cluster_inverse_metric(metric, archive, labels, centers, 0.0)

        # Within-mode variance ≈ within_sd^2; pooled is inflated by orders of magnitude.
        @test all(isapprox.(within, within_sd^2; atol = 0.15))
        @test all(within .< 0.1 .* pooled)
        @test HMCExt.between_variance_fraction(archive, labels, centers) > 0.9
    end

    @testset "K = 1 reduces to the raw memory covariance" begin
        rng = Xoshiro(3)
        archive = [randn(rng, 2) for _ in 1:400]
        metric = DiagEuclideanMetric(2)
        # Lloyd on K = 1 puts the single centre at the mean, so within == pooled.
        labels1, centers1 = HMCExt.cluster_archive(archive, 1, Matrix{Float64}(undef, 2, 0))
        within = HMCExt.within_cluster_inverse_metric(metric, archive, labels1, centers1, 0.0)
        pooled = HMCExt.memory_inverse_metric(metric, archive, 0.0)
        @test isapprox(within, pooled; atol = 1.0e-8)
    end

    @testset "recovers a unimodal correlated Gaussian (gate stays closed)" begin
        scheme = setup_sampler_scheme(cluster_update())
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

    @testset "the frozen diagonal metric tracks the marginal variances when unimodal" begin
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(cluster_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 300, memory = true, silent = true
        )
        for _ in 1:300
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 300)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DiagEuclideanMetric
        @test isapprox(a.metric.M⁻¹, diag(Σ); atol = 0.3)
        @test 0 < a.integrator.ϵ < 10
    end

    @testset "the metric keeps tracking the archive during sampling" begin
        rng = backwards_compat_rng(77)
        scheme = setup_sampler_scheme(cluster_update(; every = 1))
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
        @test a.metric_steps == steps_after_warmup + 60
        @test a.metric.M⁻¹ != metric_after_warmup
        @test isapprox(a.metric.M⁻¹, diag(Σ); atol = 0.3)
    end

    @testset "running memoryless is rejected on the first warmup step" begin
        rng = backwards_compat_rng(7)
        scheme = setup_sampler_scheme(cluster_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = false, silent = true
        )
        @test_throws ErrorException AbstractMCMC.step_warmup(
            rng, model, scheme, state; num_warmup = 50
        )
    end

    @testset "static parallel tempering is rejected on the first warmup step" begin
        rng = backwards_compat_rng(123)
        ncold, nhot = 6, 4
        scheme = setup_sampler_scheme(cluster_update())
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

    @testset "pure annealing (every chain cools) is accepted" begin
        rng = backwards_compat_rng(125)
        init = [rand(Xoshiro(200 + i), MvNormal(μ, Σ)) for i in 1:24]
        scheme = setup_sampler_scheme(cluster_update())
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

    @testset "dense within-cluster estimate is thread-safe and recovers the target" begin
        dense_update() = DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8; metric = :dense); n_dims = length(μ),
            metric_strategy = cluster_pooled_metric()
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
        @test threaded == serial
    end
end
