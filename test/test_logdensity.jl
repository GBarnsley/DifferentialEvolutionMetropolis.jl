@testset "Untransformed LogDensityProblems target" begin
    target = IsotropicNormalModel([-5.0, 5.0])
    wrapped = AbstractMCMC.LogDensityModel(target)
    sampler = setup_de_update()
    options = (;
        n_chains = 4, num_warmup = 10, silent = true, progress = false,
        chain_type = DifferentialEvolutionOutput,
    )
    reference = sample(backwards_compat_rng(123), wrapped, sampler, 10; options...)
    result = sample(backwards_compat_rng(123), target, sampler, 10; options...)
    @test result.samples == reference.samples
    @test result.ld == reference.ld
    @test all(isfinite, result.ld)
end
