struct CallableNormal{M}
    model::M
end
(f::CallableNormal)(x) = LogDensityProblems.logdensity(f.model, x)

@testset "Log-density input forms" begin
    target = IsotropicNormalModel([-5.0, 5.0])
    wrapped = AbstractMCMC.LogDensityModel(target)
    f = x -> LogDensityProblems.logdensity(target, x)
    options = (; n_chains = 8, n_burnin = 10, silent = true, progress = false)
    for template in (deMC, deMCzs, DREAMz)
        reference = template(wrapped, 10; rng = backwards_compat_rng(123), options...)
        for (model, extra) in ((target, (;)), (f, (; n_dims = 2)), (CallableNormal(target), (; n_dims = 2)))
            result = template(model, 10; rng = backwards_compat_rng(123), options..., extra...)
            @test result.samples == reference.samples
            @test result.ld == reference.ld
            @test all(isfinite, result.ld)
            stopped = template(
                model, 10, 2; rng = backwards_compat_rng(123),
                warmup_epochs = 1, n_chains = 8, silent = true, progress = false, extra...
            )
            expected = template(
                wrapped, 10, 2; rng = backwards_compat_rng(123),
                warmup_epochs = 1, n_chains = 8, silent = true, progress = false
            )
            @test stopped.samples == expected.samples
            @test stopped.ld == expected.ld
        end
        @test_throws ArgumentError template(f, 10)
        @test_throws ArgumentError template(f, 10; n_dims = 0)
        @test_throws ArgumentError template(f, 10; n_dims = 2.0)
        @test_throws ArgumentError template(target, 10; n_dims = 3)
        @test_throws ArgumentError template(wrapped, 10; n_dims = 3)
    end

    sampler = setup_de_update()
    opts = (;
        n_chains = 4, num_warmup = 10, silent = true, progress = false,
        chain_type = DifferentialEvolutionOutput,
    )
    reference = sample(backwards_compat_rng(123), wrapped, sampler, 10; opts...)
    for model in (target, f)
        extra = model === f ? (; n_dims = 2) : (;)
        result = sample(backwards_compat_rng(123), model, sampler, 10; opts..., extra...)
        @test result.samples == reference.samples
        @test result.ld == reference.ld
    end
    threaded = sample(backwards_compat_rng(123), f, sampler, 10; n_dims = 2, parallel = true, opts...)
    @test threaded.samples == reference.samples
    @test threaded.ld == reference.ld
    typed = deMC(f, 10; n_dims = 2, T = Float32, rng = backwards_compat_rng(123), options...)
    @test typed isa DifferentialEvolutionOutput{Float32}
    @test all(isfinite, typed.ld)
    # The default-RNG AbstractMCMC entry point also forwards n_dims.
    @test sample(f, sampler, 2; n_dims = 2, opts...) isa DifferentialEvolutionOutput
    @test_throws ArgumentError sample(f, sampler, 2; opts...)
    adapter = DifferentialEvolutionMetropolis.as_logdensity_model(f; n_dims = 2)
    @test LogDensityProblems.dimension(adapter.logdensity) == 2
    @test LogDensityProblems.capabilities(adapter.logdensity) == LogDensityProblems.LogDensityOrder{0}()
    @test DifferentialEvolutionMetropolis.as_logdensity_model(wrapped) === wrapped
end
