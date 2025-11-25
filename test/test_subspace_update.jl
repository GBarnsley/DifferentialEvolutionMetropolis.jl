@testset "Subspace Update" begin
    @testset "Subspace Setup" begin
        dist = setup_subspace_sampling(
            γ = nothing,
            δ = 1
        )
        @test isa(dist.δ_spl, Dirac)
        @test isa(dist.cr_spl, Distributions.DiscreteNonParametricSampler)
        dist = setup_subspace_sampling(
            γ = 1.0,
            δ = truncated(Poisson(0.5), lower = 1),
            cr = 0.5
        )
        @test isa(dist.γ, Real)
        @test isa(dist.cr_spl, Dirac)
        @test isa(dist.δ_spl, Truncated{Poisson{Float64}})
    end

    @testset "Subspace validation errors" begin
        # Test cr validation
        @test_throws ErrorException setup_subspace_sampling(cr = 0.0)  # should be > 0
        @test_throws ErrorException setup_subspace_sampling(cr = -0.1)  # negative
        @test_throws ErrorException setup_subspace_sampling(cr = 1.1)  # > 1
        @test_throws ErrorException setup_subspace_sampling(cr = Uniform(-0.1, 0.5))  # minimum ≤ 0
        @test_throws ErrorException setup_subspace_sampling(cr = Uniform(0.5, 1.1))  # maximum > 1

        # Test δ validation
        @test_throws ErrorException setup_subspace_sampling(δ = 0)  # should be > 0
        @test_throws ErrorException setup_subspace_sampling(δ = -1)  # negative
        @test_throws ErrorException setup_subspace_sampling(δ = Poisson(0.0))  # minimum ≤ 0

        # Test γ validation (when provided as Real)
        @test_throws ErrorException setup_subspace_sampling(γ = 0.0)  # should be > 0
        @test_throws ErrorException setup_subspace_sampling(γ = -1.0)  # negative

        # Test ϵ validation (noise should be centered around 0)
        @test_throws ErrorException setup_subspace_sampling(ϵ = Normal(1.0, 0.01))  # not centered at 0
        @test_throws ErrorException setup_subspace_sampling(ϵ = Uniform(0.0, 1.0))  # not centered at 0
        @test_throws ErrorException setup_subspace_sampling(ϵ = Exponential(1.0))  # not symmetric

        # Test e validation (noise should be centered around 0)
        @test_throws ErrorException setup_subspace_sampling(e = Normal(1.0, 0.01))  # not centered at 0
        @test_throws ErrorException setup_subspace_sampling(e = Uniform(0.0, 1.0))  # not centered at 0
        @test_throws ErrorException setup_subspace_sampling(e = Exponential(1.0))  # not symmetric

        # Valid cases should not error
        @test_nowarn setup_subspace_sampling(cr = 0.5, δ = 2, γ = 1.0)
        @test_nowarn setup_subspace_sampling(cr = Uniform(0.1, 0.9), δ = DiscreteUniform(1, 5))
        @test_nowarn setup_subspace_sampling(ϵ = Normal(0.0, 0.01), e = Uniform(-0.01, 0.01))

        # Disable checks should allow invalid distributions
        @test_nowarn setup_subspace_sampling(cr = 0.0, check_args = false)
        @test_nowarn setup_subspace_sampling(δ = 0, check_args = false)
        @test_nowarn setup_subspace_sampling(γ = -1.0, check_args = false)
        @test_nowarn setup_subspace_sampling(ϵ = Normal(1.0, 0.01), check_args = false)
    end

    @testset "Sample using regular Subspace" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(
            cr = DiscreteNonParametric((1:5) ./ 5, repeat([1 / 5], 5))
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = false, adapt = false
        )

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
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress = false,
            adapt = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Sample which will likely fail to pick a dimension atleast once" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(cr = 0.1)

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = false, adapt = false
        )

        @test isa(sample_result, DifferentialEvolutionMetropolis.DifferentialEvolutionSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DifferentialEvolutionState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})
    end

    @testset "Sample using memory Subspace" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_subspace_sampling(
            cr = DiscreteNonParametric((1:5) ./ 5, repeat([1 / 5], 5)),
            γ = 1.0
        )

        sample_result,
            initial_state = AbstractMCMC.step(
            rng, AbstractMCMC.LogDensityModel(model),
            de_sampler; memory = true, adapt = false
        )

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
            progress = false,
            adapt = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end
end
