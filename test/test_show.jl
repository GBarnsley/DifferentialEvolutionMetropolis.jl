plain(x) = sprint(show, MIME"text/plain"(), x)

@testset "show methods" begin
    de = setup_de_update(γ = 1.0)
    snooker = setup_snooker_update()
    subspace = setup_subspace_sampling()
    subspace_fixed = setup_subspace_sampling(γ = 0.5, cr = 0.3)

    @testset "samplers" begin
        @test repr(de) == "DE update(γ = 1.0::Float64)"
        @test plain(de) == "Differential evolution update\n  γ: 1.0::Float64\n  β: Uniform{Float64}(a=-0.0001, b=0.0001)"
        @test repr(snooker) == "Snooker update(γ = 1.68291::Float64)"
        @test plain(snooker) == "Snooker update\n  γ: 1.68291::Float64"
        @test repr(subspace) == "Adaptive subspace update(γ = 2.38 / sqrt(2δd), cr = adaptive over 3 values)"
        @test startswith(plain(subspace), "Adaptive subspace update\n")
        @test occursin("DiscreteNonParametric(support = [0.333333, 0.666667, 1.0])", plain(subspace))
        @test repr(subspace_fixed) == "Subspace update(γ = 0.5, cr = 0.3::Float64)"
        @test startswith(plain(subspace_fixed), "Subspace update\n")
        @test occursin("n_cr: 0", plain(subspace_fixed))

        composite = setup_sampler_scheme(de, snooker; w = [3.0, 1.0])
        @test repr(composite) == "Composite sampler(" * repr(de) * ", " * repr(snooker) * ")"
        @test plain(composite) == "Composite sampler with 2 updates\n    0.75 × " * repr(de) * "\n    0.25 × " * repr(snooker)
        @test sprint(show, MIME"text/plain"(), composite; context = :compact => true) == "Composite sampler with 2 updates"
    end

    @testset "state" begin
        model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([0.0, 0.0]))
        scheme = setup_sampler_scheme(de, subspace)
        sample, state = AbstractMCMC.step(backwards_compat_rng(1), model, scheme; n_chains = 4)
        @test repr(sample) == "DifferentialEvolutionSample(4 chains, 2 parameters)"
        @test repr(state.temperature_ladder) == "none"
        @test repr(state.memory) == "8 stored positions (filled every iteration, grows when full)"
        @test repr(state) == "DifferentialEvolutionState{Float64}(4 chains, 2 parameters)"
        text = plain(state)
        @test occursin("chains:         4\n", text)
        @test occursin("adaptive state: composite(static, adaptive subspace (3 crossover probabilities))", text)
        @test occursin("temperature:    none", text)
        @test occursin("memory:         8 stored positions (filled every iteration, grows when full)", text)

        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, de; n_chains = 4, n_hot_chains = 2, memory = false
        )
        text = plain(state)
        @test occursin("chains:         6 (4 cold)", text)
        @test occursin("temperature:    parallel tempering (2 hot chains)", text)
        @test occursin("memory:         none", text)

        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, de; n_chains = 4, memory = false, annealing = true, num_warmup = 10
        )
        @test occursin("temperature:    annealing (10 steps remaining)", plain(state))

        rng = backwards_compat_rng(1)
        _, state = AbstractMCMC.step(
            rng, model, de; n_chains = 4, memory_size = 3, memory_refill = true, memory_thin_interval = 2
        )
        @test occursin("memory:         8 stored positions (filled every 2 iterations, refills when full)", plain(state))
        for _ in 1:4
            _, state = AbstractMCMC.step(rng, model, de, state)
        end
        @test occursin("memory:         12 stored positions (full, refilled every 2 iterations)", plain(state))
    end

    @testset "output" begin
        output = DifferentialEvolutionOutput(zeros(5, 3, 2), fill(-1.0, 5, 3))
        @test repr(output) == "DifferentialEvolutionOutput{Float64}(5 iterations, 3 chains, 2 parameters)"
        @test plain(output) == "DifferentialEvolutionOutput{Float64}\n  iterations:  5\n  chains:      3\n  parameters:  2\n  log density: mean -1.0, range (-1.0, -1.0)"
        @test !occursin("log density", plain(DifferentialEvolutionOutput(zeros(0, 3, 2), zeros(0, 3))))
    end

    @testset "HMC" begin
        hmc = setup_hmc_update(NUTS(0.8); n_dims = 2, metric_strategy = memory_metric())
        @test repr(hmc) == "HMC update(DiagEuclideanMetric, memory_metric(shrinkage = 0.0, every = 100))"
        @test startswith(plain(hmc), "HMC update\n  trajectory:")
        @test repr(cluster_pooled_metric()) == "cluster_pooled_metric(shrinkage = 0.0, every = 100, kmax = 10)"
        @test repr(per_cluster_metric(kmax = 3)) == "per_cluster_metric(shrinkage = 0.0, every = 100, kmax = 3)"
        @test repr(setup_hmc_update(NUTS(0.8); n_dims = 2)) == "HMC update(DiagEuclideanMetric, stock adaptor)"

        model = AbstractMCMC.LogDensityModel(
            ADgradient(:ForwardDiff, CorrelatedGaussianModel(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])))
        )
        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), model, setup_sampler_scheme(hmc, de); n_chains = 4
        )
        @test occursin("adaptive state: composite(HMC (not initialised, 0 HMC steps), static)", plain(state))
    end
end
