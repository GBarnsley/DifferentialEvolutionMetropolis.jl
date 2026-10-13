mutable struct CountingNormalModel
    mean::Vector{Float64}
    count::Int
end
LogDensityProblems.dimension(model::CountingNormalModel) = length(model.mean)
function LogDensityProblems.logdensity(model::CountingNormalModel, x::AbstractVector{<:Real})
    model.count += 1
    return -sum(abs2, x .- model.mean) / 2
end
LogDensityProblems.capabilities(::CountingNormalModel) = LogDensityProblems.LogDensityOrder{0}()

struct BimodalModel
    separation::Float64
end
LogDensityProblems.dimension(::BimodalModel) = 2
function LogDensityProblems.logdensity(model::BimodalModel, x::AbstractVector{<:Real})
    a = -sum(abs2, x .- model.separation) / 2
    b = -sum(abs2, x .+ model.separation) / 2
    return max(a, b) + log1p(exp(-abs(a - b)))
end
LogDensityProblems.capabilities(::BimodalModel) = LogDensityProblems.LogDensityOrder{0}()

@testset "Multi-try update" begin
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0, 1.0]))
    sample_kwargs = (progress = false, silent = true, chain_type = DifferentialEvolutionOutput)

    @testset "setup" begin
        @test setup_multi_try_update(setup_de_update()).n_tries == 5
        @test setup_multi_try_update(setup_subspace_sampling(); n_tries = 2).n_tries == 2
        @test setup_multi_try_update(setup_subspace_sampling(γ = 0.5, cr = 0.3)) isa DifferentialEvolutionMetropolis.DifferentialEvolutionMultiTrySampler
        @test_throws ErrorException setup_multi_try_update(setup_de_update(); n_tries = 0)
        @test_throws ErrorException setup_multi_try_update(setup_snooker_update())
        @test_throws ErrorException setup_multi_try_update(setup_swap_update())
        @test_throws ErrorException setup_multi_try_update(setup_sampler_scheme(setup_de_update()))
        @test DifferentialEvolutionMetropolis.chains_required(setup_multi_try_update(setup_de_update())) == 3
        @test setup_multi_try_update(setup_de_update()) == setup_multi_try_update(setup_de_update())
    end

    @testset "n_tries = 1 matches the wrapped update" begin
        updates = (setup_de_update(), setup_subspace_sampling(), setup_subspace_sampling(γ = 0.5, cr = 0.3))
        @testset "$(nameof(typeof(update))), memory = $memory, hot chains = $n_hot" for update in updates,
                (memory, n_hot) in ((false, 0), (true, 0), (false, 3))
            run(sampler) = sample(
                backwards_compat_rng(11), model, sampler, 200;
                n_chains = 6, n_hot_chains = n_hot, memory = memory, num_warmup = 100, sample_kwargs...
            )
            reference = run(update)
            multi_try = run(setup_multi_try_update(update; n_tries = 1))
            @test multi_try.samples == reference.samples
            @test multi_try.ld == reference.ld
        end
    end

    @testset "threaded matches serial" begin
        @testset "$(nameof(typeof(update)))" for update in (setup_de_update(), setup_subspace_sampling())
            sampler = setup_multi_try_update(update; n_tries = 4)
            run(parallel) = sample(
                backwards_compat_rng(3), model, sampler, 100;
                n_chains = 6, n_hot_chains = 2, num_warmup = 50, parallel = parallel, sample_kwargs...
            )
            serial = run(false)
            threaded = run(true)
            @test threaded.samples == serial.samples
            @test threaded.ld == serial.ld
        end
    end

    @testset "model copies" begin
        sampler = setup_multi_try_update(setup_de_update(); n_tries = 3)
        sample(backwards_compat_rng(1), model, sampler, 5; n_chains = 4, parallel = true, sample_kwargs...)
        @test length(sampler.scratch.models) == 12
        @test sampler.scratch.model_source === model.logdensity
        other = AbstractMCMC.LogDensityModel(IsotropicNormalModel([1.0, 2.0, 3.0]))
        sample(backwards_compat_rng(1), other, sampler, 5; n_chains = 4, parallel = true, sample_kwargs...)
        @test sampler.scratch.model_source === other.logdensity
        @test all(m -> m.mean == [1.0, 2.0, 3.0], sampler.scratch.models)
    end

    @testset "log density evaluations" begin
        counting = CountingNormalModel([0.0, 0.0], 0)
        n_chains, n_tries = 4, 3
        sampler = setup_multi_try_update(setup_subspace_sampling(); n_tries = n_tries)
        _, state = AbstractMCMC.step(
            backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), sampler;
            n_chains = n_chains, memory = false, silent = true
        )
        counting.count = 0
        AbstractMCMC.step(backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), sampler, state)
        @test counting.count == n_chains * (2 * n_tries - 1)
    end

    @testset "Gaussian moments" begin
        @testset "$(nameof(typeof(update))), hot chains = $n_hot" for update in (setup_de_update(n_dims = 3), setup_subspace_sampling()),
                n_hot in (0, 3)
            output = sample(
                backwards_compat_rng(1234), model, setup_multi_try_update(update; n_tries = 4), 3000;
                n_chains = 6, n_hot_chains = n_hot, num_warmup = 1000, sample_kwargs...
            )
            draws = reshape(output.samples[1001:end, :, :], :, 3)
            @test isapprox(vec(mean(draws; dims = 1)), [-5.0, 5.0, 1.0]; atol = 0.15)
            @test isapprox(vec(var(draws; dims = 1)), ones(3); atol = 0.15)
        end
    end

    @testset "bimodal target" begin
        bimodal = AbstractMCMC.LogDensityModel(BimodalModel(2.5))
        sampler = setup_sampler_scheme(
            setup_multi_try_update(setup_de_update(n_dims = 2); n_tries = 4), setup_snooker_update(); w = [0.9, 0.1]
        )
        output = sample(
            backwards_compat_rng(7), bimodal, sampler, 6000;
            n_chains = 4, num_warmup = 1000, sample_kwargs...
        )
        draws = output.samples[1001:end, :, 1]
        @test isapprox(mean(draws .> 0), 0.5; atol = 0.1)
    end

    @testset "composite with adaptive subspace" begin
        sampler = setup_sampler_scheme(
            setup_multi_try_update(setup_subspace_sampling(); n_tries = 3), setup_snooker_update()
        )
        rng = backwards_compat_rng(4)
        _, state = AbstractMCMC.step(rng, model, sampler; n_chains = 6, silent = true)
        @test state.adaptive_state isa DifferentialEvolutionMetropolis.DifferentialEvolutionAdaptiveComposite
        for _ in 1:200
            _, state = AbstractMCMC.step_warmup(rng, model, sampler, state)
        end
        adapted = state.adaptive_state.adaptive_states[1]
        @test sum(adapted.L) > 0
        @test !all(==(adapted.p[1]), adapted.p)
        _, state = AbstractMCMC.step(rng, model, sampler, state)
        @test all(isfinite, state.ld)
    end

    @testset "Float32 positions" begin
        output = sample(
            backwards_compat_rng(5), model, setup_multi_try_update(setup_de_update(); n_tries = 2), 50;
            n_chains = 6, T = Float32, sample_kwargs...
        )
        @test eltype(output.samples) == Float32
    end
end
