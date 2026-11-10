@testset "Differential Evolution Update" begin
    @testset "deMC setup" begin
        double_dist = setup_de_update(;
            γ = truncated(Normal(0.8, 1.2); lower = 0.0),
            β = Uniform(-1.0e-4, 1.0e-4)
        )
        @test isa(double_dist.γ_spl, Truncated{Normal{Float64}})
        @test isa(double_dist.β_spl, Uniform)
        @test double_dist == setup_de_update(;
            γ = truncated(Normal(0.8, 1.2); lower = 0.0),
            β = Uniform(-1.0e-4, 1.0e-4),
            n_dims = 10
        )

        single_dist = setup_de_update(;
            γ = 0.5,
            β = Normal(0.0, 1.0e-4)
        )
        @test isa(single_dist.γ_spl, Dirac)
        @test isa(single_dist.β_spl, Distributions.Normal)
        @test single_dist == setup_de_update(;
            γ = 0.5,
            β = Normal(0.0, 1.0e-4),
            n_dims = 10
        )
        det = setup_de_update(;
            n_dims = 10
        )
        @test isa(det.γ_spl, Dirac)
        @test isa(det.β_spl, Normal)
        ran = setup_de_update()
        @test isa(ran.γ_spl, Uniform)
        @test ran.β_spl == Normal(0.0, 1.0e-12)
    end

    @testset "deMC validation errors" begin
        # Test γ validation
        @test_throws ErrorException setup_de_update(; γ = -1.0)
        @test_throws ErrorException setup_de_update(; γ = 0.0)
        @test_throws ErrorException setup_de_update(; γ = Uniform(-1.0, 0.5))
        @test_throws ErrorException setup_de_update(; γ = truncated(Normal(0.0, 1.0); upper = 0.0))

        # Test β validation (noise should be centered around 0)
        @test_throws ErrorException setup_de_update(; β = Normal(1.0, 0.01))  # not centered at 0
        @test_throws ErrorException setup_de_update(; β = Uniform(0.0, 1.0))  # not centered at 0
        @test_throws ErrorException setup_de_update(; β = Gumbel(0, 1.0))  # not symmetric

        # Valid cases should not error
        @test_nowarn setup_de_update(; γ = 1.0, β = Normal(0.0, 0.01))
        @test_nowarn setup_de_update(; γ = Uniform(0.1, 2.0), β = Uniform(-0.01, 0.01))

        # Disable checks should allow invalid distributions
        @test_nowarn setup_de_update(; γ = -1.0, check_args = false)
        @test_nowarn setup_de_update(; β = Normal(1.0, 0.01), check_args = false)
    end

    @testset "Sample using regular deMC" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_de_update()

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = false)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DEMSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DEMState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DEMSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DEMState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DEMSample) for x in samples)
    end

    @testset "memoryless chains are updated in turn" begin
        model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([0.0]))
        _, state = AbstractMCMC.step(backwards_compat_rng(1), model, setup_de_update(; n_dims = 1); n_chains = 3, memory = false, silent = true)
        for i in 1:3
            picked = DifferentialEvolutionMetropolis.pick_chains(state, i, Val(2))
            expected = [j < i ? state.xₚ[j] : state.x[j] for j in 1:3 if j != i]
            @test all(p -> count(e -> e === p, expected) == 1, picked)
            @test picked[1] !== picked[2]
        end
    end

    @testset "parallel without memory" begin
        rng = backwards_compat_rng(1)
        model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
        kwargs = (; n_chains = 4, memory = false, silent = true)
        DEM = DifferentialEvolutionMetropolis
        @test DEM.cannot_use_parallel(setup_de_update(), 1)
        @test DEM.cannot_use_parallel(setup_swap_update(), 1)
        @test DEM.cannot_use_parallel(setup_sampler_scheme(setup_de_update(), setup_swap_update()), 1)

        @test_logs (:warn, MEMORYLESS_PARALLEL_WARNING) AbstractMCMC.step(
            rng, model, setup_de_update(); parallel = true, kwargs...
        )
        @test_logs (:warn, MEMORYLESS_PARALLEL_WARNING) match_mode = :any deMC(
            model, 10; parallel = true, progress = false, silent = true
        )
        @test_logs AbstractMCMC.step(
            rng, model, setup_sampler_scheme(setup_de_update(), setup_snooker_update()); parallel = true, n_tries = 2, kwargs...
        )
        @test_logs AbstractMCMC.step(rng, model, setup_de_update(); parallel = true, n_tries = 2, kwargs...)
        @test_logs AbstractMCMC.step(rng, model, setup_de_update(); parallel = true, n_chains = 4, silent = true)

        @testset "$(nameof(typeof(sampler))) runs the chains in turn" for sampler in (setup_de_update(), setup_subspace_sampling())
            run(parallel) = suppress_warnings(MEMORYLESS_PARALLEL_WARNING) do
                sample(
                    backwards_compat_rng(2), model, sampler, 20;
                    num_warmup = 10, parallel, progress = false, chain_type = DEMOutput, kwargs...
                )
            end
            @test run(true).samples == run(false).samples
        end
    end

    @testset "Sample using memory deMC" begin
        rng = backwards_compat_rng(1234)
        model = IsotropicNormalModel([-5.0, 5.0])

        de_sampler = setup_de_update(;
            n_dims = LogDensityProblems.dimension(model)
        )

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler; memory = true)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DEMSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DEMState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        sample_result,
            initial_state = AbstractMCMC.step(rng, AbstractMCMC.LogDensityModel(model), de_sampler, initial_state)

        @test isa(sample_result, DifferentialEvolutionMetropolis.DEMSample)
        @test length(sample_result.x) == LogDensityProblems.dimension(model) * 2
        @test isa(initial_state, DifferentialEvolutionMetropolis.DEMState)
        @test length(initial_state.x) == LogDensityProblems.dimension(model) * 2
        @test length(initial_state.x[1]) == LogDensityProblems.dimension(model)
        @test length(initial_state.ld) == LogDensityProblems.dimension(model) * 2
        @test all(isfinite, initial_state.ld)
        @test all([all(isfinite, x) for x in initial_state.x])
        @test isa(initial_state.x[1], Vector{Float64})

        samples = sample(
            rng,
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            100;
            progress = false
        )
        @test length(samples) == 100
        @test all(isa(x, DifferentialEvolutionMetropolis.DEMSample) for x in samples)
    end
end
