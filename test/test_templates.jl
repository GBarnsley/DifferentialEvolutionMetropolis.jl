@testset "templates" begin
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))

    @testset "chain_type outputs" begin
        # Test default output (should be Array)
        result_default = deMC(ld, 50; memory = false)
        @test isa(result_default, DifferentialEvolutionOutput)
        @test isa(result_default.samples, Array{Float64, 3})

        # Test explicit Array output
        result_array = deMC(ld, 50; memory = false, chain_type = DifferentialEvolutionOutput)
        @test isa(result_array, DifferentialEvolutionOutput)
        @test isa(result_array.samples, Array{Float64, 3})

        # Test raw samples output (default AbstractMCMC behavior)
        result_any = deMC(ld, 50; memory = false, chain_type = Any)
        @test isa(result_any, Vector)
        @test length(result_any) == 50
        @test isa(result_any[1], DifferentialEvolutionMetropolis.DifferentialEvolutionSample)

        # Test with save_final_state = true
        result_with_state = deMC(
            ld, 50; memory = false, chain_type = DifferentialEvolutionOutput,
            save_final_state = true
        )
        @test isa(result_with_state, Tuple)
        @test length(result_with_state) == 2
        @test isa(result_with_state[1], DifferentialEvolutionOutput)  # processed samples
        @test isa(result_with_state[2], DifferentialEvolutionMetropolis.DifferentialEvolutionState)  # final state

        # Test array dimensions are correct
        n_dims = LogDensityProblems.dimension(ld.logdensity)
        n_chains = 2 * n_dims
        @test size(result_array.samples, 2) == n_chains
        @test size(result_array.samples, 3) == n_dims
        @test size(result_array.ld, 2) == n_chains

        #test with meta chains
        n_meta_chains = 3
        n_samples = 10000
        n_burnin = 10000
        n_dims = LogDensityProblems.dimension(ld.logdensity)
        n_total_chains = n_meta_chains * (2 * n_dims)

        meta_any = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples + n_burnin, n_meta_chains;
            chain_type = Any, num_warmup = n_burnin, discard_initial = 0
        )
        @test length(meta_any) == 3
        @test all([length(meta_any[i]) == n_samples + n_burnin for i in 1:n_meta_chains])

        meta_array = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples + n_burnin, n_meta_chains;
            chain_type = DifferentialEvolutionOutput, num_warmup = n_burnin, discard_initial = 0
        )
        @test size(meta_array.samples) == (n_samples + n_burnin, n_total_chains, n_dims)
        @test size(meta_array.ld) == (n_samples + n_burnin, n_total_chains)

        meta_array = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples + n_burnin, n_meta_chains;
            chain_type = DifferentialEvolutionOutput, num_warmup = n_burnin, discard_initial = 0, save_final_state = true
        )
        @test size(meta_array[1].samples) == (n_samples + n_burnin, n_total_chains, n_dims)
        @test size(meta_array[1].ld) == (n_samples + n_burnin, n_total_chains)
        @test isa(meta_array[2], Vector{<:DifferentialEvolutionMetropolis.DifferentialEvolutionState})
    end

    @testset "non memory runs" begin
        deMC(ld, 100; memory = false, save_burnt = true)
    end
    @testset "memory runs" begin
        deMCzs(ld, 1000; thinning = 2, memory = true, save_burnt = true)
    end
    @testset "thin memory" begin
        deMC(ld, 100; memory = true, memory_thin_interval = 5)
    end
    @testset "non-refill memory" begin
        deMC(ld, 100; memory = true, memory_size = 50, memory_refill = true)
    end
    @testset "parameter simplifying" begin
        deMC(ld, 100; memory = false, γ₁ = 0.5, γ₂ = 0.5)
    end
    @testset "annealing and parallel tempering" begin
        deMC(ld, 100; memory = false, annealing = true)
    end
    @testset "warnings" begin
        #check for warning
        ld_wide = AbstractMCMC.LogDensityModel(
            IsotropicNormalModel(
                [
                    -5.0, 5.0, 0.0, 0.0, 0.0,
                ]
            )
        )
        @test_logs (:warn,) deMC(ld_wide, 1000; thinning = 2, n_chains = 4)
    end
    @testset "DREAMz noise distributions" begin
        DREAMz(ld, 100; memory = false, ϵ = Uniform(-1.0e-4, 1.0e-4), e = Normal(0.0, 1.0e-2), δ = 2)
        DREAMz(ld, 100; memory = false, ϵ = Laplace(0.0, 1.0e-6), e = Logistic(0.0, 1.0e-2), δ = Categorical([0.5, 0.5]))
        deMC(ld, 100; memory = false, β = Uniform(-1.0e-4, 1.0e-4))
        deMCzs(ld, 100; memory = false, β = Uniform(-1.0e-4, 1.0e-4))
    end
    @testset "DREAMz crossover and γ₂ weight types" begin
        suppress_warnings(FIXED_CROSSOVER_WARNING) do
            DREAMz(ld, 100; memory = false, γ₂ = 1, cr₁ = 0.5, cr₂ = Beta(2, 2))
            DREAMz(ld, 100; memory = false, cr₁ = Uniform(0.1, 0.9), p_γ₂ = 1 // 5)
        end
        @test_throws TypeError DREAMz(ld, 100; memory = false, p_γ₂ = nothing)
    end
    @testset "initial_position building" begin
        n_dims = LogDensityProblems.dimension(ld.logdensity)

        disable_logging(Logging.Debug)
        @test_logs match_mode = :any (
            :info,
            "   Initial position is smaller than the requested (or required) n_chains (including hot chains). Expanding initial position.",
        ) deMC(
            ld, 100; initial_position = [
                randn(n_dims)
                    for _ in 2:(n_dims * 2)
            ]
        )
        @test_logs match_mode = :any (
            :info,
            "   Done!",
        ) deMC(
            ld, 100; initial_position = [randn(n_dims) for _ in 1:(n_dims * 2)], n_chains = n_dims * 2, memory = false
        )
        @test_logs match_mode = :any (
            :info,
            "   Assuming initial position size is n_chains. Ignoring extra positions.",
        ) deMC(
            ld, 100; initial_position = [randn(n_dims) for _ in 1:(n_dims * 2)], n_chains = n_dims * 2 - 1, n_hot_chains = 0, memory = false
        )
        @test_logs match_mode = :any (
            :info,
            "   Initial position is larger than requested number of chains. Shrinking initial position appending the rest to initial memory.",
        ) deMC(
            ld, 100; initial_position = [randn(n_dims) for _ in 0:(n_dims * 2)], memory = true
        )
        @test_throws ErrorException deMC(
            ld, 100; initial_position = [
                randn(n_dims - 1)
                    for _ in 1:(n_dims * 2)
            ]
        )

        @test_throws ErrorException deMC(
            ld, 100; initial_position = [randn(n_dims) for _ in 1:10],
            n_chains = 5, memory = false, n_hot_chains = 2
        )
        @test_logs match_mode = :any (
            :info, "   Initial memory size greater than N₀, truncating memory.",
        ) deMC(
            ld, 100; initial_position = [randn(n_dims) for _ in 0:10], N₀ = 5, memory = true
        )

        @test_logs match_mode = :any (
            :info, "   Consider tailoring memory_size keyword argument to control memory usage!",
        ) deMC(
            ld, 100; memory = true, memory_refill = true
        )

        @test_logs match_mode = :any (
            :info, "   Consider tailoring memory_size keyword argument to control memory usage!",
        ) deMC(
            ld, 100; n_burnin = 0, memory = true, memory_refill = true
        )
        disable_logging(Logging.Info)
    end
end
