@testset "Differential Evolution Update" begin
    @testset "deMC setup" begin
        double_dist = setup_de_update(
            γ = truncated(Normal(0.8, 1.2), lower = 0.0),
            β = Uniform(-1.0e-4, 1.0e-4)
        )
        @test isa(double_dist.γ_spl, Truncated{Normal{Float64}})
        @test isa(double_dist.β_spl, Uniform)
        @test double_dist == setup_de_update(
            γ = truncated(Normal(0.8, 1.2), lower = 0.0),
            β = Uniform(-1.0e-4, 1.0e-4),
            n_dims = 10
        )

        single_dist = setup_de_update(
            γ = 0.5,
            β = Normal(0.0, 1.0e-4)
        )
        @test isa(single_dist.γ_spl, Dirac)
        @test isa(single_dist.β_spl, Distributions.Normal)
        @test single_dist == setup_de_update(
            γ = 0.5,
            β = Normal(0.0, 1.0e-4),
            n_dims = 10
        )
        det = setup_de_update(
            n_dims = 10
        )
        @test isa(det.γ_spl, Dirac)
        @test isa(det.β_spl, Uniform)
        ran = setup_de_update()
        @test isa(ran.γ_spl, Uniform)
        @test isa(ran.β_spl, Uniform)
    end

    @testset "deMC validation errors" begin
        # Test γ validation
        @test_throws ErrorException setup_de_update(γ = -1.0)
        @test_throws ErrorException setup_de_update(γ = 0.0)
        @test_throws ErrorException setup_de_update(γ = Uniform(-1.0, 0.5))
        @test_throws ErrorException setup_de_update(γ = truncated(Normal(0.0, 1.0), upper = 0.0))

        # Test β validation (noise should be centered around 0)
        @test_throws ErrorException setup_de_update(β = Normal(1.0, 0.01))  # not centered at 0
        @test_throws ErrorException setup_de_update(β = Uniform(0.0, 1.0))  # not centered at 0
        @test_throws ErrorException setup_de_update(β = Gumbel(0, 1.0))  # not symmetric

        # Valid cases should not error
        @test_nowarn setup_de_update(γ = 1.0, β = Normal(0.0, 0.01))
        @test_nowarn setup_de_update(γ = Uniform(0.1, 2.0), β = Uniform(-0.01, 0.01))

        # Disable checks should allow invalid distributions
        @test_nowarn setup_de_update(γ = -1.0, check_args = false)
        @test_nowarn setup_de_update(β = Normal(1.0, 0.01), check_args = false)
    end

    @testset "Sample using regular deMC" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_de_update()

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

    @testset "Sample using memory deMC" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_de_update(
            n_dims = LogDensityProblems.dimension(model)
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
