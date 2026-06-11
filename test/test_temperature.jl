@testset "Temperature Functionality" begin
    # Set up common test model
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
    n_dims = 2

    @testset "Parallel Tempering Basic Functionality" begin
        rng = backwards_compat_rng(1234)

        # Test basic parallel tempering with DE sampler
        @testset "DE with Parallel Tempering" begin
            result = deMC(
                ld, 100;
                n_chains = 3,
                n_hot_chains = 2,
                max_temp_pt = 5.0,
                memory = false,
                rng = rng,
                save_burnt = false
            )

            # Should only return samples from cold chains
            @test size(result.samples, 2) == 3  # Only cold chains in output
            @test size(result.samples, 3) == n_dims
            @test size(result.ld, 2) == 3

            # Test with memory
            result_mem = deMC(
                ld, 50;
                n_chains = 3,
                n_hot_chains = 2,
                memory = true,
                rng = rng
            )
            @test size(result_mem.samples, 2) == 3
        end

        @testset "deMCzs with Parallel Tempering" begin
            result = deMCzs(
                ld, 100;
                n_chains = 3,
                n_hot_chains = 2,
                epoch_limit = 2,
                α = 1.2,
                max_temp_pt = 4.0,
                rng = rng
            )

            @test size(result.samples, 2) == 3  # Only cold chains
            @test size(result.ld, 2) == 3
        end

        @testset "DREAMz with Parallel Tempering" begin
            result = DREAMz(
                ld, 100;
                n_chains = 3,
                n_hot_chains = 2,
                epoch_limit = 2,
                α = 0.8,
                max_temp_pt = 6.0,
                rng = rng
            )

            @test size(result.samples, 2) == 3
            @test size(result.ld, 2) == 3
        end
    end

    @testset "Simulated Annealing Functionality" begin
        rng = backwards_compat_rng(5678)

        @testset "DE with Simulated Annealing" begin
            # Test basic annealing functionality - should complete successfully
            result = deMC(
                ld, 50;
                n_burnin = 25,
                annealing = true,
                max_temp_sa = 3.0,
                α = 1.0,
                memory = false,
                n_chains = 4,  # Ensure enough chains
                rng = rng,
                save_burnt = false
            )
            @test isa(result, DifferentialEvolutionOutput)
            @test size(result.samples, 2) == 4  # Only cold chains
        end

        @testset "deMCzs with Simulated Annealing" begin
            result = deMCzs(
                ld, 50;
                warmup_epochs = 2,
                epoch_limit = 2,
                annealing = true,
                max_temp_sa = 3.0,
                n_chains = 4,
                rng = rng
            )

            @test isa(result, DifferentialEvolutionOutput)
            @test size(result.samples, 3) == n_dims
            @test size(result.samples, 2) == 4  # Only cold chains
        end

        @testset "DREAMz with Simulated Annealing" begin
            result = DREAMz(
                ld, 50;
                warmup_epochs = 2,
                epoch_limit = 2,
                annealing = true,
                max_temp_sa = 3.0,
                n_chains = 4,
                rng = rng
            )

            @test isa(result, DifferentialEvolutionOutput)
            @test size(result.samples, 3) == n_dims
            @test size(result.samples, 2) == 4  # Only cold chains
        end
    end

    @testset "Combined Parallel Tempering and Simulated Annealing" begin
        rng = backwards_compat_rng(9999)

        # Test combining both features - should complete successfully
        result = DREAMz(
            ld, 30;
            n_chains = 3,
            n_hot_chains = 2,
            annealing = true,
            warmup_epochs = 2,
            epoch_limit = 2,
            max_temp_pt = 3.0,
            max_temp_sa = 4.0,
            α = 1.0,
            rng = rng
        )
        @test isa(result, DifferentialEvolutionOutput)
    end

    @testset "Parameter Edge Cases and Validation" begin
        rng = backwards_compat_rng(1111)

        @testset "Zero Hot Chains" begin
            # Should work normally without parallel tempering
            result = deMC(
                ld, 50;
                n_hot_chains = 0,
                max_temp_pt = 5.0,
                rng = rng
            )
            @test isa(result, DifferentialEvolutionOutput)
        end

        @testset "Different Temperature Spacing" begin
            # Test very close temperature spacing
            result_close = deMC(
                ld, 30;
                n_chains = 2,
                n_hot_chains = 2,
                α = 0.3,  # Very close spacing
                max_temp_pt = 2.0,
                memory = false,
                rng = rng
            )
            @test isa(result_close, DifferentialEvolutionOutput)

            # Test wide temperature spacing
            result_wide = deMC(
                ld, 30;
                n_chains = 2,
                n_hot_chains = 2,
                α = 2.0,  # Wide spacing
                max_temp_pt = 10.0,
                memory = false,
                rng = rng
            )
            @test isa(result_wide, DifferentialEvolutionOutput)
        end

        @testset "Custom Temperature Ladder" begin
            # Test providing custom temperature ladder
            custom_ladder = [[1.0, 1.0, 3.0, 6.0]]
            result = deMC(
                ld, 30;
                n_chains = 2,
                n_hot_chains = 2,
                temperature_ladder = custom_ladder,
                memory = false,
                rng = rng
            )
            @test isa(result, DifferentialEvolutionOutput)
            @test size(result.samples, 2) == 2  # Only cold chains
        end
    end

    @testset "Temperature Integration with Memory" begin
        rng = backwards_compat_rng(2222)

        # Test warning when using hot chains with memory
        @test_logs (:warn, r"Memory-based samplers.*hot chains") deMCzs(
            ld, 50;
            n_chains = 2,
            n_hot_chains = 3,
            memory = true,
            epoch_limit = 2,
            rng = rng
        )
    end

    @testset "Temperature with Different Samplers" begin
        rng = backwards_compat_rng(3333)

        # Test parallel tempering works with different update types
        @testset "Subspace Sampling with Temperature" begin
            result = DREAMz(
                ld, 50;
                n_chains = 2,
                n_hot_chains = 2,
                n_cr = 2,
                epoch_limit = 2,
                rng = rng,
                memory = false
            )
            @test isa(result, DifferentialEvolutionOutput)
        end

        @testset "Snooker Updates with Temperature" begin
            result = deMCzs(
                ld, 50;
                n_chains = 2,
                n_hot_chains = 2,
                p_snooker = 0.3,
                epoch_limit = 2,
                rng = rng,
                memory = false
            )
            @test isa(result, DifferentialEvolutionOutput)
        end
    end

    @testset "Performance and Parallel Execution" begin
        rng = backwards_compat_rng(4444)

        # Test parallel execution with temperature
        result = DREAMz(
            ld, 30;
            n_chains = 2,
            n_hot_chains = 2,
            parallel = true,
            epoch_limit = 2,
            rng = rng,
            memory = false
        )
        @test isa(result, DifferentialEvolutionOutput)
        @test size(result.samples, 2) == 2
    end

    @testset "Hastings offset is correctly tempered" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        wrapped_model = AbstractMCMC.LogDensityModel(model)

        spl = setup_sampler_scheme(setup_de_update())

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = 2,
            n_hot_chains = 2,
            max_temp_pt = 4.0,
            memory = false,
            adapt = false,
            silent = true
        )

        # Set temperature of chain 4 to exactly 2.0
        state.temperature_ladder.temperature[4] = 2.0

        # Test values:
        # ld = -10.0, ldₚ = -8.0 -> Δld = 2.0
        # offset = -1.5
        # Correct threshold: Δld/T + offset = 2.0/2.0 - 1.5 = -0.5
        # Incorrect threshold (tempered offset): (Δld + offset)/T = (2.0 - 1.5)/2.0 = 0.25
        state.ld[4] = -10.0
        state.ldₚ[4] = -8.0
        offset = -1.5

        # Find a seed for backwards_compat_rng such that the next rand() value u is > 0.6065.
        # u = 0.8 is great because log(0.8) ≈ -0.223 > -0.5.
        # We find a seed that gives u ∈ (0.65, 0.9).
        seed = 1
        while true
            u = rand(backwards_compat_rng(seed))
            if 0.65 < u < 0.9
                break
            end
            seed += 1
        end
        state.rngs[4] = backwards_compat_rng(seed)

        # Call update_chain! and assert rejection
        accepted = DifferentialEvolutionMetropolis.update_chain!(wrapped_model.logdensity, state, offset, 4)
        @test accepted == false
    end

    @testset "NaN comparison fall through to acceptance" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])
        wrapped_model = AbstractMCMC.LogDensityModel(model)

        spl = setup_sampler_scheme(setup_de_update())

        _, state = AbstractMCMC.step(
            rng, wrapped_model, spl;
            n_chains = 4,
            memory = false,
            adapt = false,
            silent = true
        )

        state.ld[4] = -Inf
        state.ldₚ[4] = -Inf

        accepted = DifferentialEvolutionMetropolis.update_chain!(wrapped_model.logdensity, state, 0.0, 4)
        @test accepted == true
    end
end
