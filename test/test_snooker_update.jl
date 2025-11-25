@testset "Snooker Update" begin
    @testset "Snooker Setup" begin
        double_dist = setup_snooker_update(
            γ = truncated(Normal(0.8, 1.2); lower = 0.1)
        )
        @test isa(double_dist.γ_spl, Truncated{Normal{Float64}})

        single_dist = setup_snooker_update(
            γ = 0.5
        )
        @test isa(single_dist.γ_spl, Dirac)
        det = setup_snooker_update()
        @test isa(det.γ_spl, Dirac)
        ran = setup_snooker_update(
            deterministic_γ = false
        )
        @test isa(ran.γ_spl, Uniform)
    end

    @testset "Snooker validation errors" begin
        # Test γ validation
        @test_throws ErrorException setup_snooker_update(γ = -1.0)
        @test_throws ErrorException setup_snooker_update(γ = Uniform(-1.0, 0.5))
        @test_throws ErrorException setup_snooker_update(γ = truncated(Normal(0.0, 1.0), upper = 0.0))
        @test_throws ErrorException setup_snooker_update(γ = Dirac(0.0))  # exactly 0 should error

        # Valid cases should not error
        @test_nowarn setup_snooker_update(γ = 1.0)
        @test_nowarn setup_snooker_update(γ = Uniform(0.1, 2.0))
        @test_nowarn setup_snooker_update(γ = truncated(Normal(1.0, 0.5), lower = 0.1))

        # Disable checks should allow invalid distributions
        @test_nowarn setup_snooker_update(γ = -1.0, check_args = false)
        @test_nowarn setup_snooker_update(γ = Dirac(0.0), check_args = false)
    end

    @testset "Sample using regular Snooker" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_snooker_update()

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = false)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
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
            progress = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample using memory Snooker" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_snooker_update(
            deterministic_γ = false
        )

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = true)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
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
            progress = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end
end
