# Tests for the HMC/NUTS harness (ext/AdvancedHMCExt.jl): an AdvancedHMC sampler run as
# one update inside a composite scheme, sharing a single adapted kernel across all DE
# chains.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag

@testset "HMC harness (AdvancedHMCExt)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    grad_model = ADgradient(:ForwardDiff, raw_model)
    model = AbstractMCMC.LogDensityModel(grad_model)

    @testset "swap update with HMC keeps density caches and samples consistent" begin
        for parallel in (false, true)
            scheme = setup_sampler_scheme(
                setup_hmc_update(NUTS(0.8); n_dims = length(μ)), setup_swap_update(); w = [0.5, 0.5]
            )
            rng = backwards_compat_rng(44)
            _, state = suppress_warnings(MEMORY_HOT_CHAINS_WARNING) do
                AbstractMCMC.step(
                    rng, model, scheme;
                    n_chains = 3, n_hot_chains = 2, memory = true,
                    num_warmup = 20, silent = true
                )
            end
            for _ in 1:10
                p0 = state.memory.fill.position
                draw, state = AbstractMCMC.step_warmup(
                    rng, model, scheme, state;
                    num_warmup = 20, parallel
                )
                @test draw.x == state.x_smpl_view
                @test state.ld ≈ LogDensityProblems.logdensity.(Ref(grad_model), state.x)
                @test state.memory.mem_x[(p0 + 1):state.memory.fill.position] == reverse(state.x)
            end
            for _ in 1:10
                draw, state = AbstractMCMC.step(rng, model, scheme, state; parallel)
                @test draw.x == state.x_smpl_view
                @test state.ld ≈ LogDensityProblems.logdensity.(Ref(grad_model), state.x)
            end
        end
    end

    @testset "a symbol-metric sampler needs n_dims" begin
        # NUTS/HMC/HMCDA store the metric as a symbol, so the dimension must be supplied.
        @test_throws ErrorException DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8))
    end

    @testset "accepts a hand-built HMCSampler (kernel, metric, adaptor)" begin
        # An HMCSampler already carries a sized metric, so n_dims is optional here.
        d = length(μ)
        metric = DiagEuclideanMetric(d)
        integrator = Leapfrog(0.1)
        κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
        adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(HMCSampler(κ, metric, adaptor))
        )
        out = sample(
            backwards_compat_rng(5), model, scheme, 800;
            n_chains = 6, num_warmup = 300, memory = true, progress = false,
            chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.15)
        @test isapprox(cov(flat), Σ; atol = 0.25)
    end

    @testset "gradient requirement fails loudly on order-0 target" begin
        order0 = AbstractMCMC.LogDensityModel(raw_model)
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ)))
        @test_throws ErrorException sample(
            backwards_compat_rng(1), order0, scheme, 50;
            n_chains = 6, num_warmup = 20, memory = false, progress = false
        )
    end

    @testset "internal AdvancedHMC entry points still exist" begin
        # The harness depends on a wider surface than the AbstractMCMC layer; if any
        # of these disappear upstream, fail here rather than deep in a trajectory.
        for f in (
                :make_metric, :make_step_size, :make_integrator, :make_kernel,
                :make_adaptor, :Hamiltonian, :phasepoint, :transition, :update,
            )
            @test isdefined(AdvancedHMC, f)
        end
        for f in (
                :is_in_window, :is_window_end, :reset!, :adapt!,
                :initialize!, :finalize!, :getM⁻¹,
            )
            @test isdefined(AdvancedHMC.Adaptation, f)
        end
    end

    # Run with memory enabled (the package's primary mode: HMC writes into the shared
    # historical pool even though its own kernel does not read it) and with memory
    # disabled, to cover both update_state paths.
    @testset "recovers a unimodal Gaussian (mean & covariance)" begin
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ)))
        out = sample(
            backwards_compat_rng(42), model, scheme, 800;
            n_chains = 6, num_warmup = 300, memory = true, progress = false,
            chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        post_cov = cov(flat)
        @test isapprox(post_mean, μ; atol = 0.15)
        @test isapprox(post_cov, Σ; atol = 0.25)
    end

    @testset "warmup → sampling freezes ϵ and the metric" begin
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ)))
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 6, num_warmup = 200, memory = false, silent = true)
        for _ in 1:200
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 200)
        end
        a = hmc_adaptive_state(state)
        @test a.initialized
        frozen_ϵ = a.integrator.ϵ
        frozen_metric = copy(a.metric.M⁻¹)

        # Adaptation must have actually done something: a sensible step size and a
        # metric that moved away from the identity towards the marginal variances.
        # (Guards against a silent no-op that would still "freeze" at the defaults.)
        @test 0 < frozen_ϵ < 10
        @test frozen_metric != ones(length(μ))
        @test isapprox(frozen_metric, diag(Σ); atol = 0.5)

        # Sampling must not change the adaptive state: the metric/ϵ stay frozen.
        for _ in 1:50
            _, state = AbstractMCMC.step(rng, model, scheme, state)
        end
        a = hmc_adaptive_state(state)
        @test a.integrator.ϵ == frozen_ϵ
        @test a.metric.M⁻¹ == frozen_metric
    end

    @testset "interleaves with DE updates in a composite" begin
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ)),
            setup_de_update();
            w = [0.5, 0.5]
        )
        out = sample(
            backwards_compat_rng(7), model, scheme, 1200;
            n_chains = 6, num_warmup = 400, progress = false,
            chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.2)
        @test isapprox(cov(flat), Σ; atol = 0.3)
    end

    @testset "threaded trajectory loop matches the serial result statistically" begin
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ)))
        out = sample(
            backwards_compat_rng(13), model, scheme, 800;
            n_chains = 6, num_warmup = 300, parallel = true, progress = false,
            chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.15)
        @test all(isfinite, flat)
    end

    @testset "memoryless parallel HMC is not sequential" begin
        hmc = DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8); n_dims = length(μ))
        @test !DifferentialEvolutionMetropolis.cannot_use_parallel(hmc, 1)
        kwargs = (; n_chains = 4, num_warmup = 10, memory = false, parallel = true, silent = true)
        @test_logs AbstractMCMC.step(backwards_compat_rng(1), model, setup_sampler_scheme(hmc); kwargs...)
        @test_logs AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_sampler_scheme(hmc, setup_de_update()); kwargs...
        )
    end

    @testset "a dense metric is thread-safe on the parallel trajectory loop" begin
        # A `DenseEuclideanMetric` carries a `_temp` scratch buffer that `neg_energy` writes
        # every leapfrog step; `run_trajectories!` gives each chain a private-scratch copy so
        # the shared buffer cannot race. With per-chain RNGs the parallel run must reproduce the
        # same-seed serial run exactly — a race on `_temp` would perturb the threaded result.
        runit(parallel) = reshape(
            sample(
                backwards_compat_rng(13), model,
                setup_sampler_scheme(
                    DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = length(μ))
                ),
                800; n_chains = 6, num_warmup = 300, parallel = parallel, memory = true,
                progress = false, chain_type = DEMOutput
            ).samples, :, length(μ)
        )
        serial = runit(false)
        threaded = runit(true)
        @test all(isfinite, threaded)
        @test isapprox(
            vec(sum(serial; dims = 1) ./ size(serial, 1)), μ; atol = 0.2
        )
        @test threaded == serial
    end

    @testset "a symbol integrator errors when never warmed up" begin
        for sampler in (NUTS(0.8), HMC(10), HMCDA(0.8, 1.0)), adapt in (false, true)
            scheme = setup_sampler_scheme(setup_hmc_update(sampler; n_dims = length(μ)))
            _, state = AbstractMCMC.step(
                backwards_compat_rng(1234), model, scheme;
                n_chains = 6, num_warmup = 0, memory = false, adapt, silent = true
            )
            @test_throws ErrorException AbstractMCMC.step(
                backwards_compat_rng(1234), model, scheme, state
            )
        end
    end

    @testset "an explicit step size samples without warmup ($name, adapt = $adapt)" for (name, sampler) in (
                "NUTS" => NUTS(0.8; integrator = Leapfrog(0.3)),
                "NUTS dense" => NUTS(0.8; integrator = Leapfrog(0.3), metric = :dense),
                "HMC" => HMC(0.3, 8),
                "HMCDA" => HMCDA(0.8, 2.0; integrator = Leapfrog(0.3)),
            ), adapt in (false, true)
        hmc = setup_hmc_update(sampler; n_dims = length(μ))
        out = sample(
            backwards_compat_rng(21), model, setup_sampler_scheme(hmc), 1500;
            n_chains = 6, memory = false, adapt, progress = false, chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        @test isapprox(vec(sum(flat; dims = 1) ./ size(flat, 1)), μ; atol = 0.2)
        @test isapprox(cov(flat), Σ; atol = 0.3)
        @test hmc.integrator.ϵ == 0.3
    end

    @testset "an un-warmed update still requires a gradient target" begin
        scheme = setup_sampler_scheme(setup_hmc_update(HMC(0.3, 8); n_dims = length(μ)))
        order0 = AbstractMCMC.LogDensityModel(raw_model)
        @test_throws ErrorException sample(
            backwards_compat_rng(1), order0, scheme, 10;
            n_chains = 6, memory = false, adapt = false, progress = false
        )
    end

    @testset "$name recovers a unimodal Gaussian ($strategy)" for (name, sampler) in (
                "HMC" => HMC(0.3, 8), "HMCDA" => HMCDA(0.8, 3.0),
            ), (strategy, kw) in (("stock", (;)), ("memory_metric", (; metric_strategy = memory_metric())))
        scheme = setup_sampler_scheme(setup_hmc_update(sampler; n_dims = length(μ), kw...))
        out = sample(
            backwards_compat_rng(42), model, scheme, 2000;
            n_chains = 6, num_warmup = 300, memory = true, progress = false,
            chain_type = DEMOutput
        )
        flat = reshape(out.samples, :, length(μ))
        @test isapprox(vec(sum(flat; dims = 1) ./ size(flat, 1)), μ; atol = 0.2)
        @test isapprox(cov(flat), Σ; atol = 0.3)
    end

    @testset "warmup tunes only what each sampler adapts" begin
        function warm(sampler)
            rng = backwards_compat_rng(99)
            scheme = setup_sampler_scheme(setup_hmc_update(sampler; n_dims = length(μ)))
            _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 6, num_warmup = 200, memory = false, silent = true)
            for _ in 1:200
                _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 200)
            end
            return hmc_adaptive_state(state)
        end

        # HMC keeps an explicit step size and the unit metric.
        a = warm(HMC(0.3, 8))
        @test a.integrator.ϵ == 0.3
        @test a.metric.M⁻¹ == ones(length(μ))

        # HMC with a symbol integrator finds a step size once and keeps the unit metric.
        a = warm(HMC(10))
        @test 0 < a.integrator.ϵ < 10
        @test a.metric.M⁻¹ == ones(length(μ))

        # HMCDA adapts the step size by dual averaging but not the metric.
        a = warm(HMCDA(0.8, 2.0))
        @test a.initialized
        @test a.adaptor.state.m == 200
        @test a.integrator.ϵ == a.adaptor.state.ϵ
        @test 0 < a.integrator.ϵ < 10
        @test a.metric.M⁻¹ == ones(length(μ))
    end

    @testset "dual averaging starts from an explicit step size ($name)" for (name, sampler, ssa) in (
            ("NUTS", NUTS(0.8; integrator = Leapfrog(0.37)), a -> a.adaptor.ssa),
            ("HMCDA", HMCDA(0.8, 2.0; integrator = Leapfrog(0.37)), a -> a.adaptor),
        )
        rng = backwards_compat_rng(5)
        scheme = setup_sampler_scheme(setup_hmc_update(sampler; n_dims = length(μ)))
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 6, num_warmup = 50, memory = false, silent = true)
        _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 50)
        @test ssa(hmc_adaptive_state(state)).state.μ ≈ log(10 * 0.37)
    end

    @testset "unsupported adaptors are rejected at construction" begin
        d = length(μ)
        metric = DiagEuclideanMetric(d)
        integrator = Leapfrog(0.1)
        κ = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(10)))
        adaptor = NaiveHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
        @test_throws ErrorException setup_hmc_update(HMCSampler(κ, metric, adaptor))
    end
end
