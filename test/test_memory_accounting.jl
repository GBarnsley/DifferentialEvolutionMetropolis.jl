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
            setup_subspace_sampling(; γ = 1.0)
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
            setup_subspace_sampling(; γ = 1.0)
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

        @test isa(state.memory, DifferentialEvolutionMetropolis.DEMMemoryFill)
        @test state.memory.fill.position == 4

        # Taking 1 step: position goes from 4 to 8, which hits the length 8, and converts to Refill.
        _, state = AbstractMCMC.step_warmup(rng, wrapped_model, spl, state; num_warmup = 2)
        @test isa(state.memory, DifferentialEvolutionMetropolis.DEMMemoryRefill)
    end

    @testset "Temperature ladder advances" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(; γ = 1.0)
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
        fixed_spl, fixed_state = DifferentialEvolutionMetropolis.fix_sampler_state(spl, state1)
        @test length(fixed_state.temperature_ladder.temperature_ladder) == l1
    end

    @testset "memory_size allocation" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(; γ = 1.0)
        )

        # Default N₀ = 10 * dimension = 20 plus memory_size = 2 * num_warmup positions per chain.
        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 1,
            memory = true,
            adapt = true,
            silent = true
        )
        @test length(state.memory.mem_x) == 20 + 2 * n_chains

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            memory_size = 3,
            adapt = true,
            silent = true
        )
        @test isa(state.memory, DifferentialEvolutionMetropolis.DEMMemoryFill)
        @test length(state.memory.mem_x) == 20 + 3 * n_chains

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            memory = true,
            N₀ = 5000,
            adapt = true,
            silent = true
        )
        @test length(state.memory.mem_x) == 5000 + 1001 * n_chains

        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            memory = true,
            memory_size = 0,
            adapt = true,
            silent = true
        )
    end

    @testset "memory_size too small errors with refill" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(; γ = 1.0)
        )

        # Default N₀ = 20 and memory_size = 2 * num_warmup = 2 give 2 * 4 = 8 < N₀ + n_chains = 24.
        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 1,
            memory = true,
            memory_refill = true,
            adapt = true,
            silent = true
        )

        @test_throws ErrorException AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            memory_refill = true,
            memory_size = 1,
            adapt = true,
            silent = true
        )

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 10,
            memory = true,
            memory_refill = true,
            memory_size = 6,
            adapt = true,
            silent = true
        )
        @test isa(state.memory, DifferentialEvolutionMetropolis.DEMMemoryFill)
        @test length(state.memory.mem_x) == 24
    end

    @testset "memory extends by memory_size per chain" begin
        spl = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_subspace_sampling(; γ = 1.0)
        )
        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains,
            num_warmup = 2,
            N₀ = 4,
            memory_size = 1,
            adapt = true,
            silent = true
        )
        @test length(state.memory.mem_x) == 4 + n_chains

        _, state = AbstractMCMC.step_warmup(rng, wrapped_model, spl, state; num_warmup = 2)
        @test length(state.memory.mem_x) == 4 + 2 * n_chains
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

    @testset "Default N₀ keeps all supplied initial positions" begin
        spl = setup_sampler_scheme(setup_subspace_sampling())
        init = [randn(rng, 2) for _ in 1:24]

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains, initial_position = init, silent = true
        )
        @test state.memory.fill.position == 24
        @test state.memory.mem_x[1:24] == init

        _, state_small = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains, initial_position = init[1:5], silent = true
        )
        @test state_small.memory.fill.position == 10 * 2

        _, state_set = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = n_chains, initial_position = init, N₀ = 8, silent = true
        )
        @test state_set.memory.fill.position == 8
        @test state_set.memory.mem_x[1:8] == init[17:24]
    end

    @testset "Memory size check before sampling" begin
        DEM = DifferentialEvolutionMetropolis
        target(C, k) = ceil(Int, C / (1 - 0.99^(1 / k)) - C)
        de = setup_de_update(; n_dims = 2)
        subspace = setup_subspace_sampling(; δ = DiscreteUniform(1, 5))

        @test DEM.memory_picks(de) == 3
        @test DEM.memory_picks(setup_snooker_update()) == 4
        @test DEM.memory_picks(subspace) == 6
        @test DEM.memory_picks(setup_subspace_sampling(; δ = Categorical([0.1, 0.1, 0.8]))) == 6
        @test DEM.memory_picks(setup_subspace_sampling(; δ = truncated(Poisson(3); lower = 1))) == 6
        @test DEM.memory_picks(setup_sampler_scheme(de, subspace)) == 6
        @test DEM.target_memory_size(de, n_chains) == target(n_chains, 3)

        A = DEM.target_memory_size(de, n_chains)
        N₀ = 8
        for thin in (0, 3)
            D = max(thin, 1) * cld(A - N₀, n_chains)
            @test_logs (:warn, Regex("at least $D warm-up steps")) DEM.check_memory_size!(
                String[], de, n_chains, N₀, D - 1, thin, false, 10^6
            )
            @test_logs min_level = Logging.Warn DEM.check_memory_size!(
                String[], de, n_chains, N₀, D, thin, false, 10^6
            )
        end

        log = @test_logs min_level = Logging.Warn DEM.check_memory_size!(String[], de, n_chains, N₀, 0, 0, false, 10^6)
        @test log == ["   Suggested minimum memory size before sampling is $A positions (N₀ = $N₀)."]

        @test_logs (:warn, MEMORY_REFILL_WARNING) DEM.check_memory_size!(String[], de, n_chains, N₀, 10^6, 0, true, A - 1)
        @test_logs min_level = Logging.Warn DEM.check_memory_size!(String[], de, n_chains, N₀, 10^6, 0, true, A)

        @test_logs (:warn, MEMORY_SIZE_WARNING) AbstractMCMC.step(
            rng, wrapped_model, de; n_chains, N₀, num_warmup = 10, silent = true
        )
        @test_logs min_level = Logging.Warn AbstractMCMC.step(
            rng, wrapped_model, de; n_chains, N₀ = A, num_warmup = 10, memory_size = 1001, silent = true
        )
        @test_logs (:warn, MEMORY_REFILL_WARNING) AbstractMCMC.step(
            rng, wrapped_model, de; n_chains, memory_refill = true, memory_size = 10, silent = true
        )
    end
end
