@testset "Multiple-try" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0, 1.0]))
    sample_kwargs = (progress = false, silent = true, chain_type = DEMOutput)
    initial_state(sampler; kwargs...) = last(
        AbstractMCMC.step(backwards_compat_rng(1), model, sampler; silent = true, kwargs...)
    )

    @testset "setup" begin
        @test isnothing(initial_state(setup_de_update()).multi_try)
        state = suppress_warnings(NO_SWAP_WARNING) do
            initial_state(setup_subspace_sampling(); n_chains = 4, n_hot_chains = 2, memory = false, n_tries = 3)
        end
        @test DEM.n_tries(state.multi_try) == 3
        @test length(state.chain_models) == 18
        @test size(state.multi_try.reference) == (2, 6)
        @test_throws ErrorException initial_state(setup_de_update(); n_tries = 0)
        state = @test_logs (:warn, r"n_tries > 1") initial_state(setup_snooker_update(); n_tries = 3)
        @test isnothing(state.multi_try)
        @test length(state.chain_models) == length(state.x)
        scheme = setup_sampler_scheme(setup_de_update(), setup_snooker_update())
        @test DEM.n_tries(initial_state(scheme; n_tries = 3).multi_try) == 3
        @test !DEM.sampler_supports_multitry(setup_swap_update())
    end

    @testset "threaded matches serial" begin
        updates = (setup_de_update(), setup_subspace_sampling(), setup_sampler_scheme(setup_de_update(), setup_snooker_update()))
        @testset "$(nameof(typeof(update))), memory = $memory" for update in updates, memory in (false, true)
            run(parallel) = suppress_warnings(NO_SWAP_WARNING, MEMORY_HOT_CHAINS_WARNING, MEMORYLESS_PARALLEL_WARNING) do
                sample(
                    backwards_compat_rng(3), model, update, 100;
                    n_chains = 6, n_hot_chains = 2, num_warmup = 50, parallel = parallel, n_tries = 4, memory = memory, sample_kwargs...
                )
            end
            serial = run(false)
            threaded = run(true)
            @test threaded.samples == serial.samples
            @test threaded.ld == serial.ld
        end
    end

    @testset "log density evaluations" begin
        counting = CountingNormalModel([0.0, 0.0], 0)
        n_chains, n_tries = 4, 3
        update = setup_subspace_sampling()
        _, state = AbstractMCMC.step(
            backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), update;
            n_chains = n_chains, memory = false, silent = true, n_tries = n_tries
        )
        for chain_model in state.chain_models
            chain_model.count = 0
        end
        AbstractMCMC.step(backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), update, state)
        @test sum(chain_model.count for chain_model in state.chain_models) == n_chains * (2 * n_tries - 1)
    end

    @testset "zero weight trials leave the chains in place" begin
        _, state = AbstractMCMC.step(
            backwards_compat_rng(6), model, setup_de_update();
            n_chains = 4, memory = false, silent = true, n_tries = 3, initial_position = [zeros(3) for _ in 1:4]
        )
        _, new_state = AbstractMCMC.step(backwards_compat_rng(6), model, setup_de_update(), state)
        @test all(iszero, new_state.x)
        @test all(iszero, new_state.multi_try.selected)
    end

    @testset "weights" begin
        weights = zeros(3)
        rng = backwards_compat_rng(1)
        selected = [DEM.select_trial!(weights, rng, [0.0, -Inf, log(3.0)], 1.0, log(4.0)) for _ in 1:10_000]
        @test !any(==(2), selected)
        @test isapprox(mean(==(3), selected), 0.75; atol = 0.02)
    end

    @testset "Gaussian moments" begin
        @testset "$(nameof(typeof(update))), hot chains = $n_hot, memory = $memory" for update in (setup_de_update(n_dims = 3), setup_subspace_sampling()),
                n_hot in (0, 3), memory in (false, true)
            output = suppress_warnings(NO_SWAP_WARNING, MEMORY_HOT_CHAINS_WARNING) do
                sample(
                    backwards_compat_rng(1234), model, update, 3000;
                    n_chains = 6, n_hot_chains = n_hot, memory = memory, num_warmup = 1000, n_tries = 4, sample_kwargs...
                )
            end
            draws = reshape(output.samples[1001:end, :, :], :, 3)
            @test isapprox(vec(mean(draws; dims = 1)), [-5.0, 5.0, 1.0]; atol = 0.15)
            @test isapprox(vec(var(draws; dims = 1)), ones(3); atol = 0.15)
        end
    end

    @testset "composite with adaptive subspace" begin
        sampler = setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update())
        rng = backwards_compat_rng(4)
        _, state = AbstractMCMC.step(rng, model, sampler; n_chains = 6, silent = true, n_tries = 3)
        @test state.adaptive_state isa DEM.DEMAdaptiveComposite
        for _ in 1:200
            _, state = AbstractMCMC.step_warmup(rng, model, sampler, state)
        end
        adapted = state.adaptive_state.adaptive_states[1]
        @test sum(adapted.L) > 0
        @test !all(==(adapted.p[1]), adapted.p)
        _, state = AbstractMCMC.step(rng, model, sampler, state)
        @test all(isfinite, state.ld)
        @test DEM.n_tries(state.multi_try) == 3
    end

    @testset "templates" begin
        _, state = DREAMz(model, 20; n_burnin = 20, n_tries = 2, save_final_state = true, progress = false, silent = true)
        @test DEM.n_tries(state.multi_try) == 2
    end

    @testset "Float32 positions" begin
        output = sample(
            backwards_compat_rng(5), model, setup_de_update(), 50;
            n_chains = 6, T = Float32, n_tries = 2, sample_kwargs...
        )
        @test eltype(output.samples) == Float32
    end
end
