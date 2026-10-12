# Tests for the per-cluster (per-mode) metric (Stage 4 of ext/AdvancedHMCExt.jl):
# `per_cluster_metric()` clusters the memory archive and keeps the K cluster covariances
# *separate* — one metric per mode — routing each chain to its current mode's metric. It is the
# heteroscedastic refinement of `cluster_pooled_metric()`: where pooling collapses the K
# within-cluster covariances into one shared metric (correct only when the modes share a shape),
# this keeps them apart so separated modes of different shape each get their own preconditioner.
# Like the pooled strategy it gates to K = 1 (= `memory_metric()`) on a unimodal archive and
# carries the same memory/no-tempering requirements. `CorrelatedGaussianModel`,
# `hmc_adaptive_state`, and `backwards_compat_rng` come from test_hmc.jl / runtests.jl.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag, Diagonal

# Internal helpers (gate, clustering, estimators, assignment) live in the package extension.
const HMCExt = Base.get_extension(DifferentialEvolutionMetropolis, :AdvancedHMCExt)

# A light stand-in for the adaptive state: `per_cluster_inverse_metrics` reads `.metric` (which
# estimator/`renew` to use) and reads/updates `.prev_centers` (the k-means warm start), so it must
# be mutable.
mutable struct FakeAState{M}
    metric::M
    prev_centers::Matrix{Float64}
end
fake_astate(metric) = FakeAState(metric, Matrix{Float64}(undef, size(metric, 1), 0))

@testset "HMC per-cluster metric (AdvancedHMCExt Stage 4)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, raw_model))

    per_cluster_update(; kwargs...) = DifferentialEvolutionMetropolis.setup_hmc_update(
        NUTS(0.8); n_dims = length(μ),
        metric_strategy = per_cluster_metric(; kwargs...)
    )

    @testset "constructor validates its arguments" begin
        @test_throws ErrorException per_cluster_metric(; shrinkage = -0.1)
        @test_throws ErrorException per_cluster_metric(; shrinkage = 1.5)
        @test_throws ErrorException per_cluster_metric(; every = 0)
        @test_throws ErrorException per_cluster_metric(; kmax = 0)
        strat = per_cluster_metric(; shrinkage = 0.1, every = 5, kmax = 4)
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
            mksampler(DiagEuclideanMetric(d)); metric_strategy = per_cluster_metric()
        ).metric isa DiagEuclideanMetric
        @test DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(DenseEuclideanMetric(d)); metric_strategy = per_cluster_metric()
        ).metric isa DenseEuclideanMetric
        @test_throws ErrorException DifferentialEvolutionMetropolis.setup_hmc_update(
            mksampler(UnitEuclideanMetric(d)); metric_strategy = per_cluster_metric()
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

    @testset "K = 1 on a unimodal archive (gate closed) ⇒ memory_metric covariance" begin
        rng = Xoshiro(3)
        archive = [randn(rng, 2) for _ in 1:400]
        astate = fake_astate(DiagEuclideanMetric(2))
        @test HMCExt.multimodal_gate(archive) == false
        metrics, centers = HMCExt.per_cluster_inverse_metrics(per_cluster_metric(), astate, archive)
        @test length(metrics) == 1
        @test size(centers, 2) == 1
        @test metrics[1].M⁻¹ == HMCExt.memory_inverse_metric(astate.metric, archive, 0.0)
    end

    @testset "separated equal-shape modes ⇒ K = 2, each metric the within-mode variance" begin
        rng = Xoshiro(11)
        within_sd = 0.4
        archive = vcat(
            [[-8.0, 5.0] .+ within_sd .* randn(rng, 2) for _ in 1:300],
            [[8.0, -5.0] .+ within_sd .* randn(rng, 2) for _ in 1:300],
        )
        astate = fake_astate(DiagEuclideanMetric(2))
        metrics, centers = HMCExt.per_cluster_inverse_metrics(per_cluster_metric(), astate, archive)
        @test length(metrics) == 2
        @test size(centers, 2) == 2
        # Each separated cluster recovers the small within-mode variance, not the inflated pooled one.
        for m in metrics
            @test all(isapprox.(m.M⁻¹, within_sd^2; atol = 0.15))
        end
    end

    @testset "separated different-shape modes ⇒ each metric recovers its own mode's covariance" begin
        # The case Stage 4 exists for: pooling (Stage 3) would average these two shapes into one
        # compromise; per-cluster keeps a narrow-mode and a wide-mode metric apart.
        rng = Xoshiro(17)
        narrow = [[-8.0, 5.0] .+ [0.3, 0.3] .* randn(rng, 2) for _ in 1:300]
        wide = [[8.0, -5.0] .+ [2.0, 0.2] .* randn(rng, 2) for _ in 1:300]
        archive = vcat(narrow, wide)
        astate = fake_astate(DiagEuclideanMetric(2))
        metrics, centers = HMCExt.per_cluster_inverse_metrics(per_cluster_metric(), astate, archive)
        @test length(metrics) == 2
        small = [m for m in metrics if m.M⁻¹[1] < 1.0]
        large = [m for m in metrics if m.M⁻¹[1] ≥ 1.0]
        @test length(small) == 1 && length(large) == 1
        @test isapprox(small[1].M⁻¹, [0.09, 0.09]; atol = 0.06)
        @test isapprox(large[1].M⁻¹, [4.0, 0.04]; atol = 0.8)
    end

    @testset "nearest_center and assign_chain_metrics! route each chain by current position" begin
        centers = [0.0 10.0; 0.0 10.0]
        @test HMCExt.nearest_center([1.0, 1.0], centers) == 1
        @test HMCExt.nearest_center([9.0, 8.0], centers) == 2

        m1 = AdvancedHMC.renew(DiagEuclideanMetric(2), [1.0, 1.0])
        m2 = AdvancedHMC.renew(DiagEuclideanMetric(2), [4.0, 9.0])
        cm = Vector{DiagEuclideanMetric{Float64, Vector{Float64}}}(undef, 2)
        HMCExt.assign_chain_metrics!(cm, [m1, m2], centers, [[0.0, 0.0], [10.0, 10.0]])
        @test cm[1] === m1
        @test cm[2] === m2

        # Dense: the per-chain slot is a private-scratch copy, so values are copied in (not shared).
        d1 = AdvancedHMC.renew(DenseEuclideanMetric(2), [1.0 0.0; 0.0 1.0])
        d2 = AdvancedHMC.renew(DenseEuclideanMetric(2), [4.0 0.0; 0.0 9.0])
        dcm = [DenseEuclideanMetric(2), DenseEuclideanMetric(2)]
        HMCExt.assign_chain_metrics!(dcm, [d1, d2], centers, [[0.0, 0.0], [10.0, 10.0]])
        @test dcm[1].M⁻¹ == d1.M⁻¹
        @test dcm[2].M⁻¹ == d2.M⁻¹
        @test dcm[1] !== d1   # a copy, not the shared cluster object
    end

    @testset "recovers a unimodal correlated Gaussian (gate stays closed, K = 1)" begin
        scheme = setup_sampler_scheme(per_cluster_update())
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

    @testset "the single cluster metric tracks the marginal variances when unimodal" begin
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(per_cluster_update())
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 300, memory = true, silent = true
        )
        for _ in 1:300
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 300)
        end
        a = hmc_adaptive_state(state)
        @test a.metric isa DiagEuclideanMetric
        @test length(a.cluster_metrics) == 1            # gate stays closed on a unimodal target
        # The single cluster metric carries the estimate (`a.metric` itself is only the seed template).
        @test isapprox(a.cluster_metrics[1].M⁻¹, diag(Σ); atol = 0.3)
        @test 0 < a.integrator.ϵ < 10
    end

    @testset "the cluster metrics keep tracking the archive during sampling" begin
        rng = backwards_compat_rng(77)
        scheme = setup_sampler_scheme(per_cluster_update(; every = 1))
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 50, memory = true, silent = true
        )
        for _ in 1:50
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        end
        a = hmc_adaptive_state(state)
        steps_after_warmup = a.metric_steps
        clusters_after_warmup = a.cluster_metrics
        for _ in 1:60
            _, state = AbstractMCMC.step(rng, model, scheme, state)
        end
        a = hmc_adaptive_state(state)
        @test a.metric_steps == steps_after_warmup + 60
        # `every = 1` recomputes the cluster metrics from the growing archive on every step.
        @test a.cluster_metrics !== clusters_after_warmup
    end

    @testset "the cluster metric set is recomputed on the cadence, not every HMC step" begin
        # With a long `every`, no recompute is due during a short sampling stretch, so the K cluster
        # metrics (and their centres) are held fixed — the set is recomputed only on the cadence,
        # even though chains are relabelled against it every step (see the next testset).
        rng = backwards_compat_rng(31)
        scheme = setup_sampler_scheme(per_cluster_update(; every = 1000))
        _, state = AbstractMCMC.step(
            rng, model, scheme; n_chains = 8, num_warmup = 40, memory = true, silent = true
        )
        for _ in 1:40
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 40)
        end
        a = hmc_adaptive_state(state)
        clusters_before = a.cluster_metrics
        centers_before = a.cluster_centers
        steps_before = a.metric_steps
        for _ in 1:10
            _, state = AbstractMCMC.step(rng, model, scheme, state)
        end
        a = hmc_adaptive_state(state)
        @test a.metric_steps == steps_before + 10        # the cadence counter still advances
        @test a.cluster_metrics === clusters_before      # but no recompute fired
        @test a.cluster_centers === centers_before
    end

    @testset "chains are relabelled to their nearest mode before every HMC step" begin
        # `fill_chain_metrics!` reassigns each chain's carried metric off the fixed centres on every
        # call, so a chain that has moved into another mode picks up that mode's metric without any
        # recompute. Drive it directly with two fixed cluster metrics and a moving population.
        scheme = per_cluster_update(; every = 1000)
        ms = scheme.metric_strategy
        astate = DifferentialEvolutionMetropolis.initialize_adaptive_state(scheme, model, 2)

        m1 = AdvancedHMC.renew(astate.metric, [1.0, 1.0, 1.0])
        m2 = AdvancedHMC.renew(astate.metric, [5.0, 5.0, 5.0])
        astate.cluster_metrics = [m1, m2]
        astate.cluster_centers = [0.0 10.0; 0.0 10.0; 0.0 10.0]

        HMCExt.fill_chain_metrics!(ms, astate, (x = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],))
        @test astate.chain_metrics[1] === m1
        @test astate.chain_metrics[2] === m2

        # No recompute, only the chains moved: the labels (and metrics) follow immediately.
        HMCExt.fill_chain_metrics!(ms, astate, (x = [[9.0, 9.0, 9.0], [1.0, 1.0, 1.0]],))
        @test astate.chain_metrics[1] === m2
        @test astate.chain_metrics[2] === m1

        # Before the first recompute (no cluster metrics yet) it seeds every slot with the lone metric.
        seed = DifferentialEvolutionMetropolis.initialize_adaptive_state(scheme, model, 2)
        HMCExt.fill_chain_metrics!(ms, seed, (x = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],))
        @test all(m -> m === seed.metric, seed.chain_metrics)
    end

    @testset "running memoryless is rejected on the first warmup step" begin
        rng = backwards_compat_rng(7)
        scheme = setup_sampler_scheme(per_cluster_update())
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
        scheme = setup_sampler_scheme(per_cluster_update())
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
        scheme = setup_sampler_scheme(per_cluster_update())
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
    end

    @testset "dense per-cluster estimate is thread-safe and recovers the target" begin
        dense_update() = DifferentialEvolutionMetropolis.setup_hmc_update(
            NUTS(0.8; metric = :dense); n_dims = length(μ),
            metric_strategy = per_cluster_metric()
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
