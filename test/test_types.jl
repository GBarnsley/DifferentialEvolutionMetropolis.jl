# Positions, log densities and adaptive states follow the element type of `initial_position`.
@testset "Non-Float64 element types" begin
    for T in (Float32, BigFloat)
        ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel(T[-5, 5]))
        initial_position = [randn(backwards_compat_rng(1), T, 2) for _ in 1:4]
        @testset "$T $(nameof(template)) $(keys(kw))" for template in (deMC, deMCzs, DREAMz),
                kw in ((;), (; n_hot_chains = 2, memory = false), (; annealing = true))

            result = template(
                ld, 50;
                initial_position = initial_position, n_chains = 4, n_burnin = 50,
                rng = backwards_compat_rng(1234), silent = true, kw...
            )
            @test result isa DifferentialEvolutionOutput{T}
            @test all(isfinite, result.samples)
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
                silent = true, progress = false, chain_type = DifferentialEvolutionOutput
            )
            @test result isa DifferentialEvolutionOutput{T}
            @test all(isfinite, result.samples)
        end

        @testset "$T typed update parameters" begin
            scheme = setup_sampler_scheme(
                setup_de_update(γ = T(0.5), β = Normal(zero(T), T(1.0e-6))),
                setup_snooker_update(γ = Uniform(T(0.8), T(1.2))),
                setup_subspace_sampling(γ = T(1), cr = DiscreteNonParametric(T[0.5, 1], T[0.5, 0.5]))
            )
            result = sample(
                backwards_compat_rng(1234), ld, scheme, 50;
                num_warmup = 50, initial_position = initial_position, n_chains = 4,
                silent = true, progress = false, chain_type = DifferentialEvolutionOutput
            )
            @test result isa DifferentialEvolutionOutput{T}
            @test all(isfinite, result.samples)
        end
    end
end
