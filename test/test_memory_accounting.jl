using DifferentialEvolutionMetropolis
using Test
using Random, AbstractMCMC, LogDensityProblems

@testset "Memory Accounting & Ladder Advances" begin
    rng = backwards_compat_rng(1234)
    model = IsotropicNormalModel([-5.0, 5.0])
    wrapped_model = AbstractMCMC.LogDensityModel(model)
    n_chains = 4

    @testset "Memory writes exactly n_chains" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        )

        # Initialize
        _, state0 = AbstractMCMC.step(rng, wrapped_model, spl; n_chains = n_chains, num_warmup = 10, memory = true, adapt = true, silent = true)

        # Take a step_warmup
        p0 = state0.memory.fill.position
        _, state1 = AbstractMCMC.step_warmup(rng, wrapped_model, spl, state0; num_warmup = 10)
        p1 = state1.memory.fill.position

        @test p1 - p0 == n_chains

        # Check no duplicates in the newly written block
        # The new block is written at slots p0+1 to p1
        new_block = state1.memory.mem_x[(p0 + 1):p1]
        @test length(unique(new_block)) == n_chains

        # Take a post-warmup step
        _, state2 = AbstractMCMC.step(rng, wrapped_model, spl, state1)
        p2 = state2.memory.fill.position

        @test p2 - p1 == n_chains

        # Check no duplicates in this block either
        new_block2 = state2.memory.mem_x[(p1 + 1):p2]
        @test length(unique(new_block2)) == n_chains
    end

    @testset "Refill conversion conversion" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        )
        # N₀ = 4, memory_size = 2, total_memory_size = 8 > N₀ (prevents crash)
        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 2,
            N₀ = 4,
            memory_size = 2,
            memory_refill = true,
            adapt = true,
            silent = true
        )

        @test isa(state.memory, DifferentialEvolutionMetropolis.DifferentialEvolutionMemoryFill)
        @test state.memory.fill.position == 4

        # Taking 1 step: position goes from 4 to 8, which hits the length 8, and converts to Refill.
        _, state = AbstractMCMC.step_warmup(rng, wrapped_model, spl, state; num_warmup = 2)
        @test isa(state.memory, DifferentialEvolutionMetropolis.DifferentialEvolutionMemoryRefill)
    end

    @testset "Temperature ladder advances" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        )

        # Initialize with annealing = true, annealing_steps = 10
        _, state0 = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            adapt = true,
            annealing = true,
            annealing_steps = 10,
            silent = true
        )

        # Initially, length of ladder decreases by 1 across one step_warmup
        l0 = length(state0.temperature_ladder.temperature_ladder)

        _, state1 = AbstractMCMC.step_warmup(rng, wrapped_model, spl, state0; num_warmup = 10)
        l1 = length(state1.temperature_ladder.temperature_ladder)
        @test l0 - l1 == 1

        # Across one post-warmup step, length of ladder decreases by exactly 1
        _, state2 = AbstractMCMC.step(rng, wrapped_model, spl, state1)
        l2 = length(state2.temperature_ladder.temperature_ladder)
        @test l1 - l2 == 1

        # fix_sampler_state leaves it unchanged
        fixed_spl, fixed_state = fix_sampler_state(spl, state1)
        @test length(fixed_state.temperature_ladder.temperature_ladder) == l1
    end

    @testset "memory_size too small errors" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0)
        )

        # This should error because default N₀ is 2 * n_chains = 8.
        # With num_warmup = 1, memory_size defaults to 2 * num_warmup = 2.
        # total_memory_size = 2 * 4 = 8.
        # N₀ + n_chains = 8 + 4 = 12.
        # Since 8 < 12, it must error.
        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 1,
            memory = true,
            adapt = true,
            silent = true
        )

        # User-set memory_size = 1 also errors
        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            memory_size = 1,
            adapt = true,
            silent = true
        )

        # Large enough memory_size (e.g. 3) should pass
        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            memory_size = 3,
            adapt = true,
            silent = true
        )
        @test isa(state.memory, DifferentialEvolutionMetropolis.DifferentialEvolutionMemoryFill)
        @test length(state.memory.mem_x) == 12
    end

    @testset "memoryless chain count validation" begin
        # Memoryless snooker with n_chains = 3 should error
        snooker_spl = setup_sampler_scheme(setup_snooker_update())
        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, snooker_spl;
            n_chains = 3,
            memory = false,
            adapt = false,
            silent = true
        )

        # Memoryless snooker with n_chains = 4 should sample fine
        _, state_snooker = AbstractMCMC.step(
            rng, wrapped_model, snooker_spl;
            n_chains = 4,
            memory = false,
            adapt = false,
            silent = true
        )
        @test length(state_snooker.x) == 4

        # Memoryless DE with n_chains = 3 should still work
        de_spl = setup_sampler_scheme(setup_de_update())
        _, state_de = AbstractMCMC.step(
            rng, wrapped_model, de_spl;
            n_chains = 3,
            memory = false,
            adapt = false,
            silent = true
        )
        @test length(state_de.x) == 3
    end

    @testset "Float32 initial positions work correctly" begin
        spl = setup_sampler_scheme(setup_de_update())
        init_pos = [Float32[1.0, 2.0], Float32[3.0, 4.0], Float32[5.0, 6.0]]
        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = 3,
            initial_position = init_pos,
            memory = false,
            adapt = false,
            silent = true
        )
        @test eltype(state.x[1]) == Float32
    end
end
