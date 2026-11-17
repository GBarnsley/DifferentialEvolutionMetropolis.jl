@testset "double ensemble" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
    kwargs = (; memory = false, double_ensemble = true, parallel = true, silent = true)

    @testset "setup" begin
        sample, state = AbstractMCMC.step(backwards_compat_rng(1), model, setup_de_update(); n_chains = 3, kwargs...)
        @test state.memory isa DEM.DEMDoubleEnsemble
        @test length(state.x) == 6
        @test length(sample.x) == 6
        @test DEM.chain_groups(state) == (1:2:6, 2:2:6)
        @test repr(state.memory) == "none (double ensemble)"

        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_sampler_scheme(setup_de_update(), setup_swap_update());
            n_chains = 3, n_hot_chains = 2, kwargs...
        )
        temperature = state.temperature_ladder.temperature
        @test length(state.x) == 10
        @test length(state.x_smpl_view) == 6
        @test temperature[1:2:end] == temperature[2:2:end]
        @test temperature[1:2:end] == [1.0, 1.0, 1.0, 1 + (2 * sqrt(2) - 1) / 2, 2 * sqrt(2)]

        _, state = suppress_warnings(NO_SWAP_WARNING) do
            AbstractMCMC.step(
                backwards_compat_rng(1), model, setup_de_update();
                n_chains = 3, temperature_ladder = [[1.0, 1.0, 1.0, 3.0]], kwargs...
            )
        end
        @test state.temperature_ladder.temperature == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0]

        positions = [fill(Float64(i), 2) for i in 1:8]
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, setup_de_update(); n_chains = 4, initial_position = positions[1:7], kwargs...)
        @test length(state.x) == 8
        @test state.x[1:7] == positions[1:7]
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, setup_de_update(); n_chains = 3, initial_position = positions, kwargs...)
        @test state.x == positions
        @test_throws r"Cannot resolve mismatch" AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_de_update(); n_chains = 3, initial_position = [fill(Float64(i), 2) for i in 1:9], kwargs...
        )
    end

    @testset "errors and warnings" begin
        rng = backwards_compat_rng(1)
        @test_throws r"requires `memory = false`" AbstractMCMC.step(rng, model, setup_de_update(); kwargs..., memory = true)
        @test_throws r"requires at least 4 chains, but only 3" AbstractMCMC.step(rng, model, setup_snooker_update(); n_chains = 3, kwargs...)
        @test_throws r"requires at least 3 chains, but only 2" AbstractMCMC.step(rng, model, setup_de_update(); n_chains = 2, kwargs...)
        @test_logs AbstractMCMC.step(rng, model, setup_snooker_update(); n_chains = 4, kwargs...)
        @test_logs AbstractMCMC.step(rng, model, setup_de_update(); n_chains = 3, kwargs...)
        @test_logs (:warn, DOUBLE_ENSEMBLE_SERIAL_WARNING) AbstractMCMC.step(rng, model, setup_de_update(); kwargs..., parallel = false)
        @test_logs (:warn, r"number of chains should be greater") AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(4))), setup_de_update(); n_chains = 3, kwargs...
        )
        @test_logs (:warn, MEMORYLESS_PARALLEL_WARNING) AbstractMCMC.step(rng, model, setup_de_update(); kwargs..., double_ensemble = false)
    end

    @testset "chains pick from the other ensemble" begin
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, setup_de_update(); n_chains = 4, kwargs...)
        for i in eachindex(state.x)
            state.x[i] .= i
            state.xₚ[i] .= -i
        end
        for i in eachindex(state.x), _ in 1:100
            picked = DEM.pick_chains(state, i, Val(3))
            expected = isodd(i) ? [state.x[j] for j in 2:2:8] : [state.xₚ[j] for j in 1:2:7]
            @test all(p -> count(e -> e === p, expected) == 1, picked)
            @test allunique(objectid.(picked))
        end
    end

    @testset "$name gives the same chains with parallel = true and false" for (name, sampler, extra) in [
            ("DE", setup_de_update(), (;)),
            ("adaptive subspace", setup_subspace_sampling(), (;)),
            ("snooker and swap", setup_sampler_scheme(setup_de_update(), setup_snooker_update(), setup_swap_update()), (; n_hot_chains = 2)),
            ("multi-try", setup_sampler_scheme(setup_de_update(), setup_subspace_sampling()), (; n_tries = 3)),
        ]
        run(parallel) = suppress_warnings(DOUBLE_ENSEMBLE_SERIAL_WARNING) do
            sample(
                backwards_compat_rng(2), model, sampler, 50;
                n_chains = 4, num_warmup = 20, progress = false, chain_type = DEMOutput, kwargs..., parallel, extra...
            )
        end
        output = run(true)
        @test size(output.samples) == (50, 4, 2, 2)
        @test size(output.ld) == (50, 4, 2)
        @test output.samples == run(false).samples
    end

    @testset "output has an ensemble dimension" begin
        run(chain_type) = sample(
            backwards_compat_rng(3), model, setup_de_update(), 20;
            n_chains = 3, n_hot_chains = 1, progress = false, chain_type, kwargs...
        )
        raw = suppress_warnings(NO_SWAP_WARNING) do
            run(Any)
        end
        output = suppress_warnings(NO_SWAP_WARNING) do
            run(DEMOutput)
        end
        @test size(output.samples) == (20, 3, 2, 2)
        @test size(output.ld) == (20, 3, 2)
        @test all(
            output.samples[i, k, e, :] == raw[i].x[2 * (k - 1) + e] && output.ld[i, k, e] == raw[i].ld[2 * (k - 1) + e]
                for i in 1:20, k in 1:3, e in 1:2
        )
        @test repr(output) == "DEMOutput{Float64}(20 iterations, 3 chains, 2 ensembles, 2 parameters)"
        @test occursin("ensembles:   2", repr(MIME("text/plain"), output))

        stacked = suppress_warnings(NO_SWAP_WARNING) do
            sample(
                backwards_compat_rng(3), model, setup_de_update(), MCMCSerial(), 20, 2;
                n_chains = 3, progress = false, chain_type = DEMOutput, kwargs...
            )
        end
        @test size(stacked.samples) == (20, 6, 2, 2)
        @test size(stacked.ld) == (20, 6, 2)
    end

    @testset "templates" begin
        output = deMC(model, 100; n_chains = 4, progress = false, kwargs...)
        @test size(output.samples) == (100, 4, 2, 2)
        output = DREAMz(model, 100; n_chains = 4, progress = false, kwargs...)
        @test size(output.samples) == (100, 4, 2, 2)
        @test_throws r"requires `memory = false`" deMCzs(model, 100; n_chains = 4, progress = false, kwargs..., memory = true)
    end
end
