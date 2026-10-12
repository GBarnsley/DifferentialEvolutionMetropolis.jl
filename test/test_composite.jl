@testset "Composite Sampler" begin
    @testset "Composite Setup" begin
        # Test single update
        single_sampler = setup_sampler_scheme(setup_de_update())
        @test isa(single_sampler, DifferentialEvolutionMetropolis.DifferentialEvolutionCompositeSampler)
        @test length(single_sampler.updates) == 1
        @test length(single_sampler.update_weights) == 1
        @test single_sampler.update_weights[1] == 1.0

        # Test multiple updates with equal weights
        multi_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update(),
            setup_subspace_sampling()
        )
        @test isa(multi_sampler, DifferentialEvolutionMetropolis.DifferentialEvolutionCompositeSampler)
        @test length(multi_sampler.updates) == 3
        @test length(multi_sampler.update_weights) == 3
        @test all(multi_sampler.update_weights .== 1.0)

        # Test with custom weights
        weighted_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update(),
            w = [0.7, 0.3]
        )
        @test isa(weighted_sampler, DifferentialEvolutionMetropolis.DifferentialEvolutionCompositeSampler)
        @test length(weighted_sampler.updates) == 2
        @test weighted_sampler.update_weights == [0.7, 0.3]

        # Test error cases
        @test_throws ErrorException setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update(),
            w = [0.5, 0.3, 0.2]  # Wrong number of weights
        )

        @test_throws ErrorException setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update(),
            w = [-0.1, 0.5]  # Negative weights
        )
    end

    @testset "Sample using regular Composite (non-adaptive)" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # DE + Snooker composite
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update()
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = false, adapt = false
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress = false,
            adapt = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample using memory Composite (non-adaptive)" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # DE + Subspace + Snooker composite with weights
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_subspace_sampling(),
            setup_snooker_update(),
            w = [0.5, 0.3, 0.2]
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = true, adapt = false
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress = false,
            adapt = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample using Composite with adaptive subspace" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # Adaptive subspace + non-adaptive DE
        de_sampler = setup_sampler_scheme(
            setup_subspace_sampling(),
            setup_de_update(),
            w = [0.6, 0.4]
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = true, adapt = true
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveComposite)
        @test length(initial_state.adaptive_state.adaptive_states) == 2
        @test isa(
            initial_state.adaptive_state.adaptive_states[1],
            DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace
        )
        @test isa(
            initial_state.adaptive_state.adaptive_states[2],
            DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic
        )
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result,
            initial_state = AbstractMCMC.step_warmup(
            rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveComposite)
        @test length(initial_state.adaptive_state.adaptive_states) == 2
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            num_warmup = 100,
            progress = false,
            adapt = true
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample using Composite with all non-adaptive" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # All non-adaptive samplers should result in static adaptive state
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_snooker_update(),
            w = [0.7, 0.3]
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = true, adapt = true
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            num_warmup = 100,
            progress = false,
            adapt = true
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Check composite adaptation works as intended" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        n_cr = 5
        its = 100

        # Create composite with adaptive subspace sampler
        de_sampler = setup_sampler_scheme(
            setup_subspace_sampling(n_cr = n_cr),
            setup_de_update(),
            w = [0.8, 0.2]
        )

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; adapt = true)
        states = Vector{typeof(initial_state)}(undef, its + 1)
        states[1] = initial_state
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, states[i - 1]
            )
            states[i] = state
        end

        # Check that we have composite adaptive states
        @test all(
            isa(state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveComposite)
                for state in states
        )
        @test all(length(state.adaptive_state.adaptive_states) == 2 for state in states)

        # Check that the subspace sampler (first component) is adapting
        subspace_adaptive_states = [
            state.adaptive_state.adaptive_states[1]
                for state in states
        ]
        @test all(
            isa(state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace)
                for state in subspace_adaptive_states
        )

        # Attempts for subspace sampler
        L_values = cat([state.L for state in subspace_adaptive_states]..., dims = 2)
        @test any(L_values .> 0)
        @test size(L_values, 1) == n_cr
        @test all(diff(L_values, dims = 2) .≥ 0)

        # Jump distances for subspace sampler
        Δ_values = cat([state.Δ for state in subspace_adaptive_states]..., dims = 2)
        @test size(Δ_values, 1) == n_cr
        @test all(diff(Δ_values, dims = 2) .≥ 0)
        @test all(Δ_values .≥ 0)

        # Variance tracking for subspace sampler
        var_counts = [state.var_count for state in subspace_adaptive_states]
        @test all(diff(var_counts) .≥ 0)  # Should be non-decreasing

        var_means = cat([state.var_mean for state in subspace_adaptive_states]..., dims = 2)
        @test size(var_means, 1) == LogDensityProblems.dimension(model)

        var_m2 = cat([state.var_m2 for state in subspace_adaptive_states]..., dims = 2)
        @test size(var_m2, 1) == LogDensityProblems.dimension(model)
        @test all(var_m2 .≥ 0)

        # Check that the DE sampler (second component) remains static
        de_adaptive_states = [state.adaptive_state.adaptive_states[2] for state in states]
        @test all(
            isa(state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveStatic)
                for state in de_adaptive_states
        )

        # Test switching to non-adaptive mode
        states_noadapt = Vector{typeof(initial_state)}(undef, its + 1)
        states_noadapt[1] = states[end]
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, states_noadapt[i - 1]
            )
            states_noadapt[i] = state
        end

        # Check that adaptation parameters don't change in non-adaptive mode
        subspace_states_noadapt = [
            state.adaptive_state.adaptive_states[1]
                for state in states_noadapt
        ]
        L_values_noadapt = cat([state.L for state in subspace_states_noadapt]..., dims = 2)
        @test L_values_noadapt[:, 1] == L_values_noadapt[:, end]
        crs_noadapt = [state.cr_spl for state in subspace_states_noadapt]
        @test crs_noadapt[1] == crs_noadapt[end]
    end

    @testset "Composite with single sampler behaves like individual sampler" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # Test single DE sampler in composite
        individual_sampler = setup_de_update()
        composite_sampler = setup_sampler_scheme(setup_de_update())

        # Compare initialization
        _,
            individual_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            individual_sampler; memory = false, adapt = false
        )
        rng = backwards_compat_rng(1234)  # Reset RNG for fair comparison
        _,
            composite_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            composite_sampler; memory = false, adapt = false
        )

        @test typeof(individual_state) == typeof(composite_state)
        @test length(individual_state.x) == length(composite_state.x)
        @test length(individual_state.ld) == length(composite_state.ld)
    end
end
