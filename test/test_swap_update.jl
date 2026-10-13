@testset "Swap update" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([0.0]))
    swap = setup_swap_update()
    function swap_state(ladder; memory = false, sampler = setup_de_update(n_dims = 1))
        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, sampler;
            n_chains = 3, n_hot_chains = 2, temperature_ladder = ladder,
            initial_position = [[10.0], [1.0], [2.0], [0.0], [3.0]],
            memory = memory, adapt = false, silent = true
        )
        return state
    end
    # Which original chain's position each chain holds after a pass, compared by identity.
    function swapped_pairs(state, original)
        pairs = Set{Tuple{Int, Int}}()
        for i in eachindex(original)
            j = findfirst(x -> x === state.xₚ[i], original)
            i == j || push!(pairs, minmax(i, j))
        end
        return pairs
    end

    @testset "acceptance ratio and cached densities" begin
        state = swap_state([[1.0, 1.0, 1.0, 2.0, 4.0]])
        @test DEM.propose_swap!(backwards_compat_rng(2), state, 1, 4)
        @test state.xₚ[1] == [0.0]
        @test !DEM.propose_swap!(backwards_compat_rng(2), state, 1, 4)
        @test state.ldₚ == LogDensityProblems.logdensity.(Ref(model.logdensity), state.xₚ)
        @test state.xₚ_smpl_view[1] === state.xₚ[1]
    end

    @testset "only adjacent current temperatures are paired" begin
        # Custom annealing schedule with current temperature order (3, 1, 5, 2, 4).
        ladder = [[2.0, 8.0, 1.0, 16.0, 4.0], [1.0, 1.0, 1.0, 2.0, 4.0]]
        pass_pairs = (Set([(1, 3), (2, 5)]), Set([(1, 5), (2, 4)]))
        for pass in 1:2, seed in 1:10
            s = swap_state(ladder)
            fill!(s.ldₚ, 0.0)
            original = copy(s.xₚ)
            DEM.swap_pass!(backwards_compat_rng(seed), s, s.temperature_ladder.temperature, pass)
            @test swapped_pairs(s, original) == pass_pairs[pass]
        end
        # Tied temperatures (3, 5) < (1, 2) < 4: one random chain per temperature.
        ladder = [[2.0, 2.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 2.0, 4.0]]
        neighbours = Set([(1, 3), (1, 5), (2, 3), (2, 5), (1, 4), (2, 4)])
        swapped = Set{Tuple{Int, Int}}()
        for seed in 1:80
            s = swap_state(ladder)
            fill!(s.ldₚ, 0.0)
            original = copy(s.xₚ)
            DEM.swap_pass!(backwards_compat_rng(seed), s, s.temperature_ladder.temperature, isodd(seed) ? 1 : 2)
            union!(swapped, swapped_pairs(s, original))
        end
        @test swapped == neighbours
        # Sorted parallel tempering ladder: cold chains 1:3, then 4 and 5.
        s = swap_state([[1.0, 1.0, 1.0, 2.0, 4.0]])
        fill!(s.ldₚ, 0.0)
        original = copy(s.xₚ)
        DEM.swap_pass!(backwards_compat_rng(3), s, s.temperature_ladder.temperature, 2)
        @test swapped_pairs(s, original) == Set([(4, 5)])
    end

    @testset "every adjacent pair is proposed each step" begin
        s = swap_state([[1.0, 1.0, 1.0, 2.0, 4.0]]; sampler = swap)
        x0 = copy.(s.x)
        fill!(s.ld, 0.0)
        _, s = AbstractMCMC.step(backwards_compat_rng(1), model, swap, s)
        @test sort(s.x) == sort(x0)
        @test count(s.x .!= x0) == 3
    end

    @testset "no swaps without distinct temperatures" begin
        for ladder in ([[8.0, 8.0, 8.0, 8.0, 8.0], [1.0, 1.0, 1.0, 2.0, 4.0]],)
            s = swap_state(ladder; sampler = swap)
            x0 = copy.(s.x)
            _, s = AbstractMCMC.step(backwards_compat_rng(2), model, swap, s)
            @test s.x == x0
        end
        _, s = AbstractMCMC.step(
            backwards_compat_rng(1), model, swap;
            n_chains = 3, memory = false, adapt = false, silent = true
        )
        x0 = copy.(s.x)
        _, s = AbstractMCMC.step(backwards_compat_rng(2), model, swap, s)
        @test s.x == x0
    end

    @testset "swaps do not allocate" begin
        swap_allocations(r, s) = @allocated DifferentialEvolutionMetropolis.swap_adjacent_temperatures!(r, s, s.temperature_ladder)
        for ladder in (
                [[1.0, 1.0, 1.0, 2.0, 4.0]],
                [[3.0, 3.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 2.0, 4.0]],
                [[2.0, 8.0, 1.0, 16.0, 4.0], [1.0, 1.0, 1.0, 2.0, 4.0]],
                [[2.0, 2.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 2.0, 4.0]],
            )
            s = swap_state(ladder)
            r = backwards_compat_rng(5)
            swap_allocations(r, s)
            @test swap_allocations(r, s) == 0
        end
    end

    @testset "samples, caches and memory after swaps" begin
        sampler = setup_sampler_scheme(setup_de_update(n_dims = 1), swap; w = [0.5, 0.5])
        for memory in (false, true), parallel in (false, true)
            s = swap_state([[1.0, 1.0, 1.0, 2.0, 4.0]]; memory, sampler)
            rng = backwards_compat_rng(123)
            for _ in 1:10
                p0 = memory ? s.memory.fill.position : 0
                draw, s = AbstractMCMC.step(rng, model, sampler, s; parallel)
                @test draw.x == s.x_smpl_view
                @test draw.ld == s.ld_smpl_view
                @test s.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), s.x)
                @test parent(s.x_smpl_view) === s.x
                if memory
                    @test s.memory.mem_x[(p0 + 1):s.memory.fill.position] == reverse(s.x)
                end
            end
        end
    end

    @testset "adaptive composite warmup advances annealing once" begin
        spl = setup_sampler_scheme(setup_subspace_sampling(), swap; w = [0.5, 0.5])
        schedule = [[3.0, 3.0, 3.0, 4.0, 5.0], [2.0, 2.0, 2.0, 3.0, 4.5], [1.0, 1.0, 1.0, 2.0, 4.0]]
        _, s = AbstractMCMC.step(
            backwards_compat_rng(3), model, spl;
            n_chains = 3, n_hot_chains = 2, memory = true,
            temperature_ladder = schedule, num_warmup = 5, silent = true
        )
        for k in 2:3
            p0 = s.memory.fill.position
            draw, s = AbstractMCMC.step_warmup(backwards_compat_rng(k), model, spl, s)
            @test s.temperature_ladder.temperature == schedule[k]
            @test draw.x == s.x_smpl_view
            @test s.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), s.x)
            @test s.memory.mem_x[(p0 + 1):s.memory.fill.position] == reverse(s.x)
        end
    end

    @testset "swap update and hot chain detection" begin
        de = setup_de_update(n_dims = 1)
        @test !DEM.has_swap_update(de)
        @test DEM.has_swap_update(swap)
        @test DEM.has_swap_update(setup_sampler_scheme(de, swap))
        @test !DEM.has_swap_update(setup_sampler_scheme(de, setup_snooker_update()))
        @test !DEM.has_hot_chains(DEM.DifferentialEvolutionNullTemperatureLadder{Float64}())
        @test DEM.has_hot_chains(swap_state([[1.0, 1.0, 1.0, 2.0, 4.0]]).temperature_ladder)
    end

    @testset "setup warnings" begin
        de = setup_de_update(n_dims = 1)
        kw = (n_chains = 3, memory = false, silent = true)
        @test_logs (:warn, r"Parallel tempering without a swap update") AbstractMCMC.step(
            backwards_compat_rng(1), model, de; n_hot_chains = 2, kw...
        )
        @test_logs (:warn, r"no hot chains") AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_sampler_scheme(de, swap); kw...
        )
        @test_logs AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_sampler_scheme(de, swap); n_hot_chains = 2, kw...
        )
    end

    @testset "templates" begin
        scheme = setup_sampler_scheme(setup_de_update(), setup_snooker_update(); w = [3.0, 1.0])
        with_swaps = DEM.add_swap_update(scheme, 2, true, 0.2)
        @test with_swaps.updates[end] isa DEM.DifferentialEvolutionSwapSampler
        @test with_swaps.update_weights ≈ [0.6, 0.2, 0.2]
        @test DEM.add_swap_update(scheme, 2, false, 0.2) === scheme
        @test (@test_logs (:warn, r"without hot chains") DEM.add_swap_update(scheme, 0, true, 0.2)) === scheme
        @test (@test_logs (:warn, r"probability = 0") DEM.add_swap_update(scheme, 2, true, 0.0)) === scheme
        @test_throws ErrorException DEM.add_swap_update(scheme, 2, true, 1.0)
        @test_throws ErrorException DEM.add_swap_update(scheme, 2, true, -0.1)

        fixed_scheme = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n = 3)
        fixed_with_swaps = DEM.add_swap_update(fixed_scheme, 2, true, 0.2)
        @test fixed_with_swaps.updates[1] === fixed_scheme
        @test fixed_with_swaps.updates[2] isa DEM.DifferentialEvolutionSwapSampler
        @test fixed_with_swaps.update_weights ≈ [0.8, 0.2]

        kw = (n_chains = 3, n_burnin = 10, memory = false, silent = true, progress = false)
        @test_logs deMC(model, 10; n_hot_chains = 2, kw...)
        @test_logs (:warn, r"Parallel tempering without a swap update") deMC(
            model, 10; n_hot_chains = 2, use_parallel_tempering_swaps = false, kw...
        )
        @test_logs (:warn, r"without hot chains") deMC(model, 10; use_parallel_tempering_swaps = true, kw...)
        @test_logs (:warn, r"probability = 0") match_mode = :any deMCzs(
            model, 10; n_hot_chains = 2, parallel_tempering_swap_probability = 0, kw...
        )
        output = DREAMz(model, 10; n_hot_chains = 2, kw...)
        @test size(output.samples) == (10, 3, 1)
    end
end
