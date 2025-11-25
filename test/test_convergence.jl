@testset "Rhat Convergence" begin
    rng = backwards_compat_rng(1234)

    de_sampler = setup_de_update()

    @testset "Should pass" begin
        model = IsotropicNormalModel([-5.0, 5.0])
        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            r̂_stopping_criteria;
            progress = false
        )
        @test (length(samples) % (1000 - 1)) == 0
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end

    @testset "Should hit max" begin
        max_its = 5000
        model = IsotropicNormalModel([-5.0, 5.0])
        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            r̂_stopping_criteria;
            n_chains = 3,
            progress = false,
            maximum_iterations = max_its,
            maximum_R̂ = 1.0
        )
        @test length(samples) == (max_its - 1)
        @test all(isa(x, DifferentialEvolutionMetropolis.DifferentialEvolutionSample) for x in samples)
    end
end
