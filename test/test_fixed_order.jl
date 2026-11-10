@testset "Fixed Order Sampler" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
    # Proposals from this update round back to the current position, so chains only move on secondary iterations
    still_update = setup_de_update(; γ = 1.0e-300, β = Normal(0.0, 1.0e-300))
    # Steps `scheme` `n` times (warm-up for the first `n_warmup`) and records whether any chain moved
    function record_moves!(rng, scheme, state, n; n_warmup = 0)
        moved = Bool[]
        for it in 1:n
            x_old = copy.(state.x)
            step_function = it <= n_warmup ? AbstractMCMC.step_warmup : AbstractMCMC.step
            _, state = step_function(rng, model, scheme, state)
            push!(moved, state.x != x_old)
        end
        return moved, state
    end

    @testset "Setup" begin
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n = 10)
        @test scheme isa DEM.DEMCompositeSamplerFixedOrder
        @test scheme.n == 10
        @test scheme.updates[2] isa DEM.DEMSnookerSampler
        @test setup_fixed_order_scheme(setup_de_update(), setup_snooker_update()).n == 1
        @test_throws ErrorException setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n = 0)
        @test DEM.chains_required(setup_fixed_order_scheme(setup_de_update(), setup_snooker_update())) ==
            max(DEM.chains_required(setup_de_update()), DEM.chains_required(setup_snooker_update()))
    end

    @testset "setup_sampler_scheme picks the scheme type from w" begin
        de, snooker, subspace = setup_de_update(), setup_snooker_update(), setup_subspace_sampling()
        @test setup_sampler_scheme(de, snooker; w = 5) isa DEM.DEMCompositeSamplerFixedOrder
        @test setup_sampler_scheme(de, snooker; w = 5).n == 5
        @test setup_sampler_scheme(de, snooker; w = [1, 2]) isa DEM.DEMCompositeSamplerRandom
        @test setup_sampler_scheme(de, snooker, subspace) isa DEM.DEMCompositeSamplerRandom
        @test setup_random_order_scheme(de, snooker; w = [1, 2]).update_weights == [1, 2]
        @test_throws "requires exactly two updates" setup_sampler_scheme(de, snooker, subspace; w = 5)
        @test_throws "requires exactly two updates" setup_sampler_scheme(de; w = 5)
        @test_throws TypeError setup_sampler_scheme(de, snooker; w = 0.5)
    end

    @testset "Show" begin
        de, snooker = setup_de_update(), setup_snooker_update()
        scheme = setup_fixed_order_scheme(de, snooker; n = 3)
        @test repr(scheme) == "Fixed order composite sampler(3 × " * repr(de) * ", 1 × " * repr(snooker) * ")"
        @test sprint(show, MIME"text/plain"(), scheme) ==
            "Fixed order composite sampler, cycle of 4 iterations\n       3 × " * repr(de) * "\n       1 × " * repr(snooker)
    end

    @testset "Cycle order" begin
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n = 3)
        order_state = DEM.initialize_order_state(scheme)
        @test [DEM.select_update!(backwards_compat_rng(1), scheme, order_state) for _ in 1:8] == [1, 1, 1, 2, 1, 1, 1, 2]
        @test order_state.iteration == 8
    end

    @testset "Chains only move on secondary iterations (adapt = $adapt)" for adapt in (true, false)
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(still_update, setup_de_update(); n = 3)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false, adapt = adapt)
        @test state.adaptive_state isa DEM.DEMAdaptiveStatic
        moved, state = record_moves!(rng, scheme, state, 12; n_warmup = 6)
        @test moved == repeat([false, false, false, true], 3)
        @test state.order_state.iteration == 12
    end

    @testset "Multiple-try (adapt = $adapt)" for adapt in (true, false)
        rng = backwards_compat_rng(1234)
        @test DEM.sampler_supports_multitry(setup_fixed_order_scheme(setup_snooker_update(), setup_de_update()))
        @test !DEM.sampler_supports_multitry(setup_fixed_order_scheme(setup_snooker_update(), setup_swap_update()))
        scheme = setup_fixed_order_scheme(still_update, setup_subspace_sampling(); n = 3)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false, adapt = adapt, n_tries = 3)
        @test DEM.n_tries(state.multi_try) == 3
        moved, state = record_moves!(rng, scheme, state, 8; n_warmup = 4)
        @test moved == repeat([false, false, false, true], 2)
        @test state.order_state.iteration == 8
    end

    @testset "Adaptive updates only adapt when used" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_subspace_sampling(); n = 2)
        _, state = AbstractMCMC.step(rng, model, scheme; memory = true)
        subspace_state = state.adaptive_state.adaptive_states[2]
        @test subspace_state isa DEM.DEMAdaptiveSubspace
        for _ in 1:2
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        end
        @test subspace_state.var_count == 0
        _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        @test subspace_state.var_count > 0
    end

    @testset "Nested random schemes pick updates with the product of their weights (adapt = $adapt)" for adapt in (true, false)
        # The subspace update is the only one that moves, picked with probability 1/2 × 1/2
        rng = backwards_compat_rng(1234)
        inner = setup_sampler_scheme(still_update, setup_subspace_sampling())
        scheme = setup_sampler_scheme(inner, still_update)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false, adapt = adapt)
        @test state.adaptive_state isa (adapt ? DEM.DEMAdaptiveComposite : DEM.DEMAdaptiveStatic)
        moved, _ = record_moves!(rng, scheme, state, 2000; n_warmup = 1000)
        @test isapprox(mean(moved), 0.25; atol = 0.04)
    end

    @testset "Adaptive updates inside nested random schemes adapt" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_sampler_scheme(
            setup_sampler_scheme(still_update, setup_subspace_sampling()), setup_snooker_update()
        )
        _, state = AbstractMCMC.step(rng, model, scheme; memory = true)
        subspace_state = state.adaptive_state.adaptive_states[1].adaptive_states[2]
        @test subspace_state isa DEM.DEMAdaptiveSubspace
        for _ in 1:50
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        end
        @test subspace_state.var_count > 0
        @test state.adaptive_state.adaptive_states[1].adaptive_states[2] === subspace_state
    end

    @testset "Random inside fixed order inside random (adapt = $adapt)" for adapt in (true, false)
        rng = backwards_compat_rng(1234)
        # Both branches of the inner random scheme hold still, so chains only move on the subspace iterations
        inner_random = setup_sampler_scheme(still_update, still_update)
        inner_fixed = setup_fixed_order_scheme(inner_random, setup_subspace_sampling(); n = 1)
        scheme = setup_sampler_scheme(inner_fixed)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false, adapt = adapt)
        @test state.adaptive_state isa (adapt ? DEM.DEMAdaptiveComposite : DEM.DEMAdaptiveStatic)
        moved, state = record_moves!(rng, scheme, state, 6; n_warmup = 3)
        @test moved == [false, true, false, true, false, true]
        @test state.order_state.order_states[1].iteration == 6
    end

    @testset "Nested fixed order schemes keep their own position" begin
        rng = backwards_compat_rng(1234)
        # Outer cycle: inner, inner, still. Inner cycle: still, DE. So DE runs on the 2nd of every 3 iterations
        inner = setup_fixed_order_scheme(still_update, setup_de_update(); n = 1)
        scheme = setup_fixed_order_scheme(inner, still_update; n = 2)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false)
        moved, state = record_moves!(rng, scheme, state, 9; n_warmup = 4)
        @test moved == repeat([false, true, false], 3)
        @test state.order_state.iteration == 9
        @test state.order_state.order_states[1].iteration == 6

        outer_fixed = setup_fixed_order_scheme(
            setup_fixed_order_scheme(setup_de_update(), setup_subspace_sampling(); n = 2),
            setup_sampler_scheme(setup_snooker_update(), setup_subspace_sampling());
            n = 3
        )
        samples = sample(rng, model, outer_fixed, 50; num_warmup = 50, progress = false)
        @test length(samples) == 50
        @test all(s -> all(x -> all(isfinite, x), s.x), samples)
    end

    @testset "Each sample call starts a new cycle" begin
        inner = setup_fixed_order_scheme(still_update, setup_de_update(); n = 2)
        scheme = setup_sampler_scheme(setup_fixed_order_scheme(inner, still_update))
        draws(seed) = sample(backwards_compat_rng(seed), model, scheme, 7; num_warmup = 5, progress = false)
        @test [s.x for s in draws(42)] == [s.x for s in draws(42)]

        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, scheme; n_chains = 20, memory = false)
        fixed_order_state = state.order_state.order_states[1]
        @test fixed_order_state.iteration == 0
        @test fixed_order_state.order_states[1].iteration == 0

        # resuming from a state keeps the position in the cycle
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, scheme, state)
        sample(backwards_compat_rng(1), model, scheme, 3; initial_state = state, progress = false)
        @test fixed_order_state.iteration == 4
    end

    @testset "fix_sampler_state keeps the position in the cycle" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(still_update, setup_subspace_sampling(); n = 2)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false)
        _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        fixed_scheme, fixed_state = fix_sampler_state(scheme, state)
        @test fixed_state.order_state === state.order_state
        @test fixed_state.adaptive_state isa DEM.DEMAdaptiveStatic
        moved, _ = record_moves!(rng, fixed_scheme, fixed_state, 3)
        @test moved == [false, true, false]
    end

    @testset "fix_sampler_state fixes updates inside nested schemes" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(
            still_update, setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update())
        )
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = true)
        subspace_state = state.adaptive_state.adaptive_states[2].adaptive_states[1]
        fixed_scheme, fixed_state = fix_sampler_state(scheme, state)
        @test fixed_scheme.updates[2].updates[1].cr_spl === subspace_state.cr_spl
        @test fixed_state.adaptive_state isa DEM.DEMAdaptiveStatic
        moved, _ = record_moves!(rng, fixed_scheme, fixed_state, 4)
        @test moved == [false, true, false, true]
    end

    @testset "Type stability (adapt = $adapt)" for adapt in (true, false)
        rng = backwards_compat_rng(1234)
        de, snooker, subspace = setup_de_update, setup_snooker_update, setup_subspace_sampling
        schemes = [
            setup_fixed_order_scheme(de(), subspace(); n = 2),
            setup_sampler_scheme(de(), subspace(), snooker()),
            setup_fixed_order_scheme(
                setup_fixed_order_scheme(de(), subspace()), setup_sampler_scheme(snooker(), subspace())
            ),
            setup_fixed_order_scheme(de(), setup_sampler_scheme(snooker(), subspace())),
            setup_fixed_order_scheme(setup_fixed_order_scheme(de(), subspace()), snooker()),
            setup_sampler_scheme(setup_fixed_order_scheme(de(), subspace()), snooker()),
            setup_fixed_order_scheme(setup_fixed_order_scheme(de(), de()), de()),
            setup_sampler_scheme(setup_sampler_scheme(de(), subspace()), snooker()),
            setup_sampler_scheme(setup_sampler_scheme(de(), subspace()), setup_sampler_scheme(snooker(), subspace())),
            setup_sampler_scheme(setup_sampler_scheme(de(), snooker()), setup_sampler_scheme(snooker(), de())),
            setup_sampler_scheme(setup_sampler_scheme(de(), de()), de()),
        ]
        for scheme in schemes
            _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 10, memory = false, adapt = adapt)
            @inferred AbstractMCMC.step(rng, model, scheme, state)
            @inferred AbstractMCMC.step_warmup(rng, model, scheme, state)
        end
    end

    @testset "Templates accept an integer instead of a probability" begin
        rng = backwards_compat_rng(1234)
        out = deMCzs(model, 100; n_burnin = 100, p_snooker = 10, rng = rng, progress = false)
        @test all(isfinite, out.samples)

        out = deMC(model, 100; n_burnin = 100, p_γ₂ = 9, rng = rng, progress = false)
        @test all(isfinite, out.samples)

        out = DREAMz(model, 100; n_burnin = 100, p_γ₂ = 4, rng = rng, progress = false, save_final_state = true)
        @test out[2].adaptive_state isa DEM.DEMAdaptiveComposite

        out = deMCzs(model, 100; n_burnin = 100, p_snooker = 0.1, rng = rng, progress = false)
        @test all(isfinite, out.samples)
        out = deMC(model, 100; n_burnin = 100, p_γ₂ = 0.1f0, γ₂ = 1.0f0, rng = rng, progress = false)
        @test all(isfinite, out.samples)

        out = suppress_warnings(MEMORY_HOT_CHAINS_WARNING) do
            deMCzs(model, 10; n_burnin = 10, p_snooker = 4, n_hot_chains = 2, rng = rng, progress = false, save_final_state = true)
        end
        @test all(isfinite, out[1].samples)
    end
end
