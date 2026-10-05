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

@testset "Explicit position type" begin
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
    for T in (Float32, BigFloat)
        @testset "$T $(nameof(template)) explicit initialization" for template in (deMC, deMCzs, DREAMz),
                initial_position in (nothing, [[-5.0, 5.0] for _ in 1:2], [[-5.0, 5.0] for _ in 1:8])
            result, state = template(
                model, 10; T = T, initial_position = initial_position,
                n_chains = 4, n_burnin = 10, rng = backwards_compat_rng(1234),
                silent = true, progress = false, save_final_state = true
            )
            @test result isa DifferentialEvolutionOutput{T}
            @test eltype(state.x[1]) === T
            @test eltype(state.ld) === T
            @test all(isfinite, result.samples)
        end
        @testset "$T explicit type with tempering and annealing" for kw in
            ((; n_hot_chains = 2, memory = false), (; annealing = true))
            result = DREAMz(
                model, 10; T = T, n_chains = 4, n_burnin = 10,
                rng = backwards_compat_rng(1234), silent = true, progress = false, kw...
            )
            @test result isa DifferentialEvolutionOutput{T}
            @test all(isfinite, result.samples)
        end
    end
    _, state = AbstractMCMC.step(
        backwards_compat_rng(1234), model, setup_de_update(); T = Float64,
        initial_position = [Float32[-5, 5] for _ in 1:4], silent = true
    )
    @test eltype(state.x[1]) === Float64
end
