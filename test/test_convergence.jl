using MCMCDiagnosticTools

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

@testset "Float32 default stopping threshold" begin
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(Float32, 2)))
    result = deMCzs(
        model, 4, 1; rng = backwards_compat_rng(42), T = Float32,
        n_chains = 4, warmup_epochs = 0, silent = true, progress = false
    )
    @test eltype(result.samples) === Float32
    @test all(isfinite, result.ld)
end
