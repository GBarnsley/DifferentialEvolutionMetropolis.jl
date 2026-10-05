@testset "Log-density input forms" begin
    target = IsotropicNormalModel([-5.0, 5.0])
    wrapped = AbstractMCMC.LogDensityModel(target)
    f = x -> LogDensityProblems.logdensity(target, x)
    options = (; n_chains = 8, n_burnin = 10, silent = true, progress = false)
    for template in (deMC, deMCzs, DREAMz)
        reference = template(wrapped, 10; rng = backwards_compat_rng(123), options...)
        for model in (target,)
            result = template(model, 10; rng = backwards_compat_rng(123), options...)
            @test result.samples == reference.samples
            @test result.ld == reference.ld
            @test all(isfinite, result.ld)
            stopped = template(
                model, 10, 2; rng = backwards_compat_rng(123),
                warmup_epochs = 1, n_chains = 8, silent = true, progress = false
            )
            expected = template(
                wrapped, 10, 2; rng = backwards_compat_rng(123),
                warmup_epochs = 1, n_chains = 8, silent = true, progress = false
            )
            @test stopped.samples == expected.samples
            @test stopped.ld == expected.ld
        end
        @test_throws ArgumentError template(f, 10; options...)
        @test_throws ArgumentError template(f, 10, 2; n_chains = 8)

    end

    sampler = setup_de_update()
    opts = (;
        n_chains = 4, num_warmup = 10, silent = true, progress = false,
        chain_type = DifferentialEvolutionOutput,
    )
    reference = sample(backwards_compat_rng(123), wrapped, sampler, 10; opts...)
    for model in (target,)
        result = sample(backwards_compat_rng(123), model, sampler, 10; opts...)
        @test result.samples == reference.samples
        @test result.ld == reference.ld
    end
    threaded = sample(backwards_compat_rng(123), target, sampler, 10; parallel = true, opts...)
    @test threaded.samples == reference.samples
    @test threaded.ld == reference.ld
    typed = deMC(target, 10; T = Float32, rng = backwards_compat_rng(123), options...)
    @test typed isa DifferentialEvolutionOutput{Float32}
    @test all(isfinite, typed.ld)
    @test sample(target, sampler, 2; opts...) isa DifferentialEvolutionOutput
    @test_throws ArgumentError sample(f, sampler, 2; opts...)
    adapter = DifferentialEvolutionMetropolis.as_logdensity_model(target)
    @test adapter.logdensity === target
    @test LogDensityProblems.dimension(adapter.logdensity) == 2
    @test LogDensityProblems.capabilities(adapter.logdensity) == LogDensityProblems.LogDensityOrder{0}()
    @test DifferentialEvolutionMetropolis.as_logdensity_model(wrapped) === wrapped
end
