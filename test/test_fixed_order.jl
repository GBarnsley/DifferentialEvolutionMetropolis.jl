@testset "Fixed Order Sampler" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
    # Proposals from this update round back to the current position, so chains only move on secondary iterations
    still_update = setup_de_update(γ = 1.0e-300, β = Normal(0.0, 1.0e-300))

    @testset "Setup" begin
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n_main = 10)
        @test scheme isa DEM.DifferentialEvolutionFixedOrderSampler
        @test scheme.n_main == 10
        @test scheme.updates[2] isa DEM.DifferentialEvolutionSnookerSampler
        @test setup_fixed_order_scheme(setup_de_update(), setup_snooker_update()).n_main == 1
        @test_throws ErrorException setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n_main = 0)
        @test DEM.chains_required(setup_fixed_order_scheme(setup_de_update(), setup_snooker_update())) ==
            max(DEM.chains_required(setup_de_update()), DEM.chains_required(setup_snooker_update()))
    end

    @testset "Show" begin
        de, snooker = setup_de_update(), setup_snooker_update()
        scheme = setup_fixed_order_scheme(de, snooker; n_main = 3)
        @test repr(scheme) == "Fixed order sampler(3 × " * repr(de) * ", 1 × " * repr(snooker) * ")"
        @test sprint(show, MIME"text/plain"(), scheme) ==
            "Fixed order sampler, cycle of 4 iterations\n       3 × " * repr(de) * "\n       1 × " * repr(snooker)
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, scheme; memory = false)
        @test occursin("adaptive state: fixed order(0 iterations; static, static)", sprint(show, MIME"text/plain"(), state))
    end

    @testset "Cycle order" begin
        adaptive_state = DEM.DifferentialEvolutionAdaptiveFixedOrder{Float64}(
            (DEM.DifferentialEvolutionAdaptiveStatic{Float64}(), DEM.DifferentialEvolutionAdaptiveStatic{Float64}()), 0
        )
        @test [DEM.next_update!(adaptive_state, 3) for _ in 1:8] == [1, 1, 1, 2, 1, 1, 1, 2]
        @test adaptive_state.iteration == 8
    end

    @testset "Chains only move on secondary iterations (adapt = $adapt)" for adapt in (true, false)
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(still_update, setup_de_update(); n_main = 3)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false, adapt = adapt)
        @test state.adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder
        moved = Bool[]
        for it in 1:12
            x_old = copy.(state.x)
            step_function = it <= 6 ? AbstractMCMC.step_warmup : AbstractMCMC.step
            _, state = step_function(rng, model, scheme, state)
            push!(moved, state.x != x_old)
        end
        @test moved == repeat([false, false, false, true], 3)
        @test state.adaptive_state.iteration == 12
    end

    @testset "Adaptive updates only adapt when used" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_subspace_sampling(); n_main = 2)
        _, state = AbstractMCMC.step(rng, model, scheme; memory = true)
        subspace_state = state.adaptive_state.adaptive_states[2]
        @test subspace_state isa DEM.DifferentialEvolutionAdaptiveSubspace
        for _ in 1:2
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        end
        @test subspace_state.var_count == 0
        _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        @test subspace_state.var_count > 0
    end

    @testset "Nesting with composite schemes" begin
        rng = backwards_compat_rng(1234)
        inner_fixed = setup_fixed_order_scheme(still_update, setup_de_update(); n_main = 1)
        outer_composite = setup_sampler_scheme(inner_fixed, setup_snooker_update(); w = [1.0, 0.0])
        for adapt in (true, false)
            _, state = AbstractMCMC.step(rng, model, outer_composite; n_chains = 20, memory = false, adapt = adapt)
            @test state.adaptive_state isa DEM.DifferentialEvolutionAdaptiveComposite
            moved = Bool[]
            for it in 1:4
                x_old = copy.(state.x)
                step_function = it <= 2 ? AbstractMCMC.step_warmup : AbstractMCMC.step
                _, state = step_function(rng, model, outer_composite, state)
                push!(moved, state.x != x_old)
            end
            @test moved == [false, true, false, true]
        end

        outer_fixed = setup_fixed_order_scheme(
            setup_fixed_order_scheme(setup_de_update(), setup_subspace_sampling(); n_main = 2),
            setup_sampler_scheme(setup_snooker_update(), setup_subspace_sampling());
            n_main = 3
        )
        samples = sample(rng, model, outer_fixed, 50; num_warmup = 50, progress = false)
        @test length(samples) == 50
        @test all(s -> all(x -> all(isfinite, x), s.x), samples)
    end

    @testset "fix_sampler_state keeps the cycle position" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(still_update, setup_subspace_sampling(); n_main = 2)
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 20, memory = false)
        _, state = AbstractMCMC.step_warmup(rng, model, scheme, state)
        fixed_scheme, fixed_state = fix_sampler_state(scheme, state)
        @test fixed_state.adaptive_state.iteration == 1
        @test all(s -> s isa DEM.DifferentialEvolutionAdaptiveStatic, fixed_state.adaptive_state.adaptive_states)
        moved = Bool[]
        for _ in 1:3
            x_old = copy.(fixed_state.x)
            _, fixed_state = AbstractMCMC.step(rng, model, fixed_scheme, fixed_state)
            push!(moved, fixed_state.x != x_old)
        end
        @test moved == [false, true, false]
    end

    @testset "A state without a cycle position errors" begin
        rng = backwards_compat_rng(1234)
        scheme = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update())
        _, state = AbstractMCMC.step(rng, model, setup_de_update(); memory = false)
        @test_throws ErrorException AbstractMCMC.step(rng, model, scheme, state)
    end

    @testset "Templates accept an integer instead of a probability" begin
        rng = backwards_compat_rng(1234)
        out = deMCzs(model, 100; n_burnin = 100, p_snooker = 10, rng = rng, progress = false, save_final_state = true)
        @test out[2].adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder
        @test all(isfinite, out[1].samples)

        out = deMC(model, 100; n_burnin = 100, p_γ₂ = 9, rng = rng, progress = false, save_final_state = true)
        @test out[2].adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder

        out = DREAMz(model, 100; n_burnin = 100, p_γ₂ = 4, rng = rng, progress = false, save_final_state = true)
        @test out[2].adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder

        out = deMCzs(model, 100; n_burnin = 100, p_snooker = 0.1, rng = rng, progress = false, save_final_state = true)
        @test !(out[2].adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder)
        out = deMCzs(model, 100; n_burnin = 100, p_snooker = 0, rng = rng, progress = false, save_final_state = true)
        @test !(out[2].adaptive_state isa DEM.DifferentialEvolutionAdaptiveFixedOrder)
    end
end
