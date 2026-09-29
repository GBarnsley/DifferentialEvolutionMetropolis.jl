@testset "Adaptive Subspace Update" begin
    @testset "Sample using regular Subspace" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling()

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = false, adapt = true
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace)
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
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace)
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

    @testset "Sample using memory Subspace" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling()

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = true, adapt = true
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace)
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
        @test isa(initial_state.adaptive_state, DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveSubspace)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
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

    @testset "check adaption works as intended" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        n_cr = 5
        its = 1000
        de_sampler = setup_subspace_sampling(n_cr = n_cr)

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; adapt = true)
        states = Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionState}(undef, its + 1)
        states[1] = deepcopy(initial_state)
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, states[i - 1]
            )
            states[i] = deepcopy(state)
        end

        #attempts
        L_values = cat([state.adaptive_state.L for state in states]..., dims = 2)
        @test any(L_values .> 0)
        @test size(L_values, 1) == n_cr
        @test all(diff(L_values, dims = 2) .≥ 0)
        n_chains = length(initial_state.x)
        skipped_its = cld(de_sampler.min_variance_count, n_chains)
        @test sum(L_values[:, end]) == ((its - skipped_its) * n_chains)
        @test all(L_values[:, skipped_its] .== 0)
        #probabilities stay above the floor
        DifferentialEvolutionMetropolis.adapted_cr_probabilities!(
            states[end].adaptive_state, de_sampler.cr_uniform_weight
        )
        p = states[end].adaptive_state.p
        @test sum(p) ≈ 1.0
        @test all(p .≥ de_sampler.cr_uniform_weight / n_cr)
        #jump distances
        Δ_values = cat([state.adaptive_state.Δ for state in states]..., dims = 2)
        @test size(Δ_values, 1) == n_cr
        @test all(diff(Δ_values, dims = 2) .≥ 0)
        @test all(Δ_values .≥ 0)
        #var counts
        var_counts = [state.adaptive_state.var_count for state in states[1:(end - 1)]]
        update_size = unique(diff(var_counts))
        @test length(update_size) == 1
        @test update_size[1] == length(initial_state.x)
        @test var_counts[end] == (its * length(initial_state.x))
        #var means seems ok
        var_means = cat([state.adaptive_state.var_mean for state in states]..., dims = 2)
        @test size(var_means, 1) == LogDensityProblems.dimension(model)

        var_m2 = cat([state.adaptive_state.var_m2 for state in states]..., dims = 2)
        @test size(var_m2, 1) == LogDensityProblems.dimension(model)

        states_2 = Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionState}(undef, its + 1)
        states_2[1] = states[end]
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, states_2[i - 1])
            states_2[i] = deepcopy(state)
        end
        L_values = cat([state.adaptive_state.L for state in states_2]..., dims = 2)
        @test L_values[:, 1] == L_values[:, end]
        crs = [state.adaptive_state.cr_spl for state in states_2]
        @test crs[1] == crs[end]

        states_noadapt = Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionState}(undef, its + 1)
        states_noadapt[1] = initial_state
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, states_noadapt[i - 1]
            )
            states_noadapt[i] = deepcopy(state)
        end
        L_values = cat([state.adaptive_state.L for state in states_noadapt]..., dims = 2)
        @test L_values[:, 1] == L_values[:, end]
        crs = [state.adaptive_state.cr_spl for state in states_noadapt]
        @test crs[1] == crs[end]

        new_sampler, initial_state2 = fix_sampler_state(de_sampler, states[end])
        states_noadapt = Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionState}(undef, its + 1)
        states_noadapt[1] = initial_state2
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model),
                new_sampler, states_noadapt[i - 1]
            )
            states_noadapt[i] = deepcopy(state)
        end
        @test isa(states_noadapt[end], DifferentialEvolutionMetropolis.DifferentialEvolutionState)

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; adapt = false)
        states_noadapt = Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionState}(undef, its + 1)
        states_noadapt[1] = initial_state
        for i in 2:(its + 1)
            sample_result,
                state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, states_noadapt[i - 1]
            )
            states_noadapt[i] = deepcopy(state)
        end
        @test isa(states_noadapt[end], DifferentialEvolutionMetropolis.DifferentialEvolutionState)
    end
    @testset "adaption ignores hot chains" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        n_chains = 4
        n_hot_chains = 3
        its = 100
        de_sampler = setup_subspace_sampling()

        _, state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model), de_sampler;
            adapt = true, n_chains = n_chains, n_hot_chains = n_hot_chains
        )
        for _ in 1:its
            _, state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, state
            )
        end
        skipped_its = cld(de_sampler.min_variance_count, n_chains)
        @test sum(state.adaptive_state.L) == (its - skipped_its) * n_chains
        @test state.adaptive_state.var_count == its * n_chains
    end

    @testset "adaption with an unused crossover value" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        de_sampler = setup_subspace_sampling(n_cr = 3)
        _, state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model), de_sampler; adapt = true
        )
        state.adaptive_state.L .= [10, 10, 10]
        state.adaptive_state.Δ .= [0.0, 1.0, 3.0]
        for w in (0.0, 0.05, 0.5)
            DifferentialEvolutionMetropolis.adapted_cr_probabilities!(state.adaptive_state, w)
            @test state.adaptive_state.p ≈ (1 - w) .* [0.0, 0.25, 0.75] .+ w / 3
        end
    end

    @testset "user-set adaptation options" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        its = 50
        de_sampler = setup_subspace_sampling(n_cr = 3, cr_uniform_weight = 0.3, min_variance_count = 40)
        @test de_sampler.cr_uniform_weight == 0.3
        @test de_sampler.min_variance_count == 40
        _, state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model), de_sampler; adapt = true
        )
        n_chains = length(state.x)
        for _ in 1:its
            _, state = AbstractMCMC.step_warmup(
                rng, AbstractMCMC.LogDensityModel(model), de_sampler, state
            )
        end
        @test sum(state.adaptive_state.L) == (its - cld(40, n_chains)) * n_chains
        DifferentialEvolutionMetropolis.adapted_cr_probabilities!(state.adaptive_state, 0.3)
        @test all(state.adaptive_state.p .≥ 0.1 - 1.0e-12)
        fixed = DifferentialEvolutionMetropolis.fix_sampler(de_sampler, state.adaptive_state)
        @test fixed.cr_uniform_weight == 0.3
        @test fixed.min_variance_count == 40
        @test_throws ErrorException setup_subspace_sampling(cr_uniform_weight = 1.5)
        @test_throws ErrorException setup_subspace_sampling(cr_uniform_weight = -0.1)
        @test_throws ErrorException setup_subspace_sampling(min_variance_count = 1)
    end

    @testset "warnings" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        # Test warning when sampler has fixed crossover probability (n_cr = 0)
        @test_logs (
            :warn, "sampler already has a fixed crossover probability, cannot adapt.",
        ) begin
            de_sampler_fixed = setup_subspace_sampling(cr = 0.5)  # Fixed cr means n_cr = 0
            AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler_fixed; adapt = true)
        end

        # Test warning when only one crossover probability (n_cr = 1)
        @test_logs (:warn, "Only one crossover probability, cannot adapt.") begin
            de_sampler_one_cr = setup_subspace_sampling(n_cr = 1)
            AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler_one_cr; adapt = true)
        end
    end
end
