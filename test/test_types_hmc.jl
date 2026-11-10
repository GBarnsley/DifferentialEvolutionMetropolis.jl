@testset "Non-Float64 element types (HMC)" begin
    for T in (Float32, BigFloat)
        ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel(T[-5, 5]))
        initial_position = [randn(backwards_compat_rng(1), T, 2) for _ in 1:4]

        @testset "$T HMC type warning" begin
            model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, ld.logdensity))
            mismatched = setup_hmc_update(NUTS(0.8); n_dims = 2)
            @test_logs (:error, r"HMC metric element type Float64 does not match") DifferentialEvolutionMetropolis.initialize_adaptive_state(mismatched, model, 4, T)
            matched = setup_hmc_update(NUTS(T(0.8)); n_dims = 2)
            @test_logs DifferentialEvolutionMetropolis.initialize_adaptive_state(matched, model, 4, T)
        end

        @testset "$T HMC scheme $(keys(hmc_kw))" for hmc_kw in ((;), (; metric_strategy = memory_metric()))
            model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, ld.logdensity))
            scheme = setup_sampler_scheme(
                setup_de_update(),
                setup_hmc_update(NUTS(T(0.8)); n_dims = 2, hmc_kw...)
            )
            result = sample(
                backwards_compat_rng(1234), model, scheme, 50;
                num_warmup = 50, initial_position = initial_position, n_chains = 4,
                silent = true, progress = false, chain_type = DEMOutput
            )
            @test result isa DEMOutput{T}
            @test all(isfinite, result.samples)
        end
    end
end
