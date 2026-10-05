@testset "Replica exchange" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([0.0]))
    sampler = setup_de_update(n_dims = 1)
    function exchange_state(ladder; memory = false)
        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, sampler;
            n_chains = 3, n_hot_chains = 2, temperature_ladder = ladder,
            initial_position = [[10.0], [1.0], [2.0], [0.0], [3.0]],
            memory = memory, adapt = false, silent = true
        )
        return state
    end
    state = exchange_state([[1.0, 1.0, 1.0, 2.0, 4.0]])
    @test DEM.exchange_pair!(backwards_compat_rng(2), state, 1, 4)
    @test state.xₚ[1] == [0.0]
    @test !DEM.exchange_pair!(backwards_compat_rng(2), state, 1, 4)
    @test state.ldₚ == LogDensityProblems.logdensity.(Ref(model.logdensity), state.xₚ)
    @test state.xₚ_smpl_view[1] === state.xₚ[1]
    @test state.x[1] == [10.0] # only accepted proposal buffers exchanged

    # A custom annealing schedule reverses hot-chain order relative to final order.
    # Equal densities force acceptance; any changed identities must be neighbours
    # in CURRENT temperature order (3,1,5,2,4), never in final index order.
    ladder = [[2.0, 8.0, 1.0, 16.0, 4.0], [1.0, 1.0, 1.0, 2.0, 4.0]]
    neighbours = Set([(1, 3), (1, 5), (2, 5), (2, 4)])
    for seed in 1:40
        s = exchange_state(ladder)
        fill!(s.ldₚ, 0.0)
        original = copy(s.xₚ)
        DEM.replica_exchange!(backwards_compat_rng(seed), s)
        for i in eachindex(original)
            j = findfirst(x -> x === s.xₚ[i], original)
            @test i == j || minmax(i, j) in neighbours
        end
    end
    for ladder in (
            [[1.0, 1.0, 1.0, 2.0, 4.0]],
            [[8.0, 8.0, 8.0, 8.0, 8.0], [1.0, 1.0, 1.0, 2.0, 4.0]],
        )
        s = exchange_state(ladder)
        original = copy(s.xₚ)
        DEM.replica_exchange!(backwards_compat_rng(2), s; enabled = false)
        @test s.xₚ == original
        if length(ladder) > 1
            DEM.replica_exchange!(backwards_compat_rng(2), s)
            @test s.xₚ == original
        end
    end
    # The public step actually enables exchange by default, with an explicit opt-out.
    changed = false
    for seed in 1:20
        a = exchange_state([[1.0, 1.0, 1.0, 2.0, 4.0]])
        b = deepcopy(a)
        da, _ = AbstractMCMC.step(backwards_compat_rng(seed), model, sampler, a)
        db, _ = AbstractMCMC.step(backwards_compat_rng(seed), model, sampler, b; replica_exchange = false)
        changed |= da.x != db.x
    end
    @test changed
    # Samples, caches, cold views and archive must all reflect post-exchange positions.
    for memory in (false, true), parallel in (false, true)
        s = exchange_state([[1.0, 1.0, 1.0, 2.0, 4.0]]; memory)
        for _ in 1:5
            p0 = memory ? s.memory.fill.position : 0
            draw, s = AbstractMCMC.step(backwards_compat_rng(123), model, sampler, s; parallel)
            @test draw.x == s.x_smpl_view
            @test draw.ld == s.ld_smpl_view
            @test s.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), s.x)
            @test parent(s.x_smpl_view) === s.x
            if memory
                @test s.memory.mem_x[(p0 + 1):s.memory.fill.position] == reverse(s.x)
            end
        end
    end
    # Adaptive and composite warmup exchange exactly once before archive write and
    # advance the annealing schedule exactly once.
    for spl in (
            setup_subspace_sampling(),
            setup_sampler_scheme(setup_subspace_sampling(), setup_de_update()),
        )
        schedule = [[3.0, 3.0, 3.0, 4.0, 5.0], [1.0, 1.0, 1.0, 2.0, 4.0]]
        _, s = AbstractMCMC.step(
            backwards_compat_rng(3), model, spl;
            n_chains = 3, n_hot_chains = 2, memory = true,
            temperature_ladder = schedule, num_warmup = 5, silent = true
        )
        p0 = s.memory.fill.position
        draw, s = AbstractMCMC.step_warmup(backwards_compat_rng(4), model, spl, s)
        @test s.temperature_ladder.temperature == schedule[2]
        @test draw.x == s.x_smpl_view
        @test s.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), s.x)
        @test s.memory.mem_x[(p0 + 1):s.memory.fill.position] == reverse(s.x)
    end
    # Untempered and pure-annealing sweeps consume exactly the old RNG stream.
    for ladder in ([[1.0, 1.0, 1.0]], [[4.0, 4.0, 4.0], [1.0, 1.0, 1.0]])
        _, s = AbstractMCMC.step(
            backwards_compat_rng(3), model, sampler;
            n_chains = 3, memory = false, temperature_ladder = ladder,
            adapt = false, silent = true
        )
        a, b = deepcopy(s), deepcopy(s)
        da, a = AbstractMCMC.step(backwards_compat_rng(4), model, sampler, a)
        db, b = AbstractMCMC.step(backwards_compat_rng(4), model, sampler, b; replica_exchange = false)
        @test da.x == db.x
        @test a.ld == b.ld
    end
end
