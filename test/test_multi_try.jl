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

struct OriginOnlyModel end
LogDensityProblems.dimension(::OriginOnlyModel) = 2
LogDensityProblems.logdensity(::OriginOnlyModel, x::AbstractVector{<:Real}) = all(iszero, x) ? 0.0 : -Inf
LogDensityProblems.capabilities(::OriginOnlyModel) = LogDensityProblems.LogDensityOrder{0}()

@testset "Multiple-try" begin
    DEM = DifferentialEvolutionMetropolis
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0, 1.0]))
    sample_kwargs = (progress = false, silent = true, chain_type = DifferentialEvolutionOutput)
    initial_state(sampler; kwargs...) = last(
        AbstractMCMC.step(backwards_compat_rng(1), model, sampler; silent = true, kwargs...)
    )

    @testset "setup" begin
        @test DEM.n_tries(initial_state(setup_de_update()).multi_try) == 1
        state = initial_state(setup_subspace_sampling(); n_chains = 4, n_hot_chains = 2, memory = false, n_tries = 3)
        @test DEM.n_tries(state.multi_try) == 3
        @test length(state.chain_models) == 18
        @test size(state.multi_try.reference) == (2, 6)
        @test_throws ErrorException initial_state(setup_de_update(); n_tries = 0)
        state = @test_logs (:warn, r"n_tries > 1") initial_state(setup_snooker_update(); n_tries = 3)
        @test DEM.n_tries(state.multi_try) == 1
        @test length(state.chain_models) == length(state.x)
        scheme = setup_sampler_scheme(setup_de_update(), setup_snooker_update())
        @test DEM.n_tries(initial_state(scheme; n_tries = 3).multi_try) == 3
        @test !DEM.supports_multi_try(setup_swap_update())
    end

    @testset "threaded matches serial" begin
        @testset "$(nameof(typeof(update)))" for update in (setup_de_update(), setup_subspace_sampling())
            run(parallel) = sample(
                backwards_compat_rng(3), model, update, 100;
                n_chains = 6, n_hot_chains = 2, num_warmup = 50, parallel = parallel, n_tries = 4, sample_kwargs...
            )
            serial = run(false)
            threaded = run(true)
            @test threaded.samples == serial.samples
            @test threaded.ld == serial.ld
        end
    end

    @testset "log density evaluations" begin
        counting = CountingNormalModel([0.0, 0.0], 0)
        n_chains, n_tries = 4, 3
        update = setup_subspace_sampling()
        _, state = AbstractMCMC.step(
            backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), update;
            n_chains = n_chains, memory = false, silent = true, n_tries = n_tries
        )
        counting.count = 0
        AbstractMCMC.step(backwards_compat_rng(2), AbstractMCMC.LogDensityModel(counting), update, state)
        @test counting.count == n_chains * (2 * n_tries - 1)
    end

    @testset "zero weight trials leave the chains in place" begin
        origin = AbstractMCMC.LogDensityModel(OriginOnlyModel())
        _, state = AbstractMCMC.step(
            backwards_compat_rng(6), origin, setup_de_update();
            n_chains = 4, memory = false, silent = true, n_tries = 3,
            initial_position = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        x = deepcopy(state.x)
        _, new_state = AbstractMCMC.step(backwards_compat_rng(6), origin, setup_de_update(), state)
        @test new_state.x[1:2] == x[1:2]
        @test all(iszero, state.multi_try.selected[1:2])
    end

    @testset "weights" begin
        lds = [0.0 -Inf; log(3.0) -Inf]
        @test DEM.tempered_log_sum(lds, 1, 1.0) ≈ log(4.0)
        @test DEM.tempered_log_sum(lds, 1, 2.0) ≈ log(1 + sqrt(3.0))
        @test DEM.tempered_log_sum(lds, 2, 1.0) == -Inf
        @test DEM.log_add_exp(log(2.0), log(3.0)) ≈ log(5.0)
        @test DEM.log_add_exp(-Inf, -Inf) == -Inf
        # weights that sum below one fall back to the last trial with a positive weight
        @test DEM.select_trial(backwards_compat_rng(1), reshape([0.0, -Inf], 2, 1), 1, 1.0, 10.0) == 1
    end

    @testset "Gaussian moments" begin
        @testset "$(nameof(typeof(update))), hot chains = $n_hot, memory = $memory" for update in (setup_de_update(n_dims = 3), setup_subspace_sampling()),
                n_hot in (0, 3), memory in (false, true)
            output = sample(
                backwards_compat_rng(1234), model, update, 3000;
                n_chains = 6, n_hot_chains = n_hot, memory = memory, num_warmup = 1000, n_tries = 4, sample_kwargs...
            )
            draws = reshape(output.samples[1001:end, :, :], :, 3)
            @test isapprox(vec(mean(draws; dims = 1)), [-5.0, 5.0, 1.0]; atol = 0.15)
            @test isapprox(vec(var(draws; dims = 1)), ones(3); atol = 0.15)
        end
    end

    @testset "bimodal target" begin
        bimodal = AbstractMCMC.LogDensityModel(BimodalModel(2.5))
        sampler = setup_sampler_scheme(setup_de_update(n_dims = 2), setup_snooker_update(); w = [0.9, 0.1])
        output = sample(
            backwards_compat_rng(7), bimodal, sampler, 6000;
            n_chains = 4, num_warmup = 1000, n_tries = 4, sample_kwargs...
        )
        draws = output.samples[1001:end, :, 1]
        @test isapprox(mean(draws .> 0), 0.5; atol = 0.1)
    end

    @testset "composite with adaptive subspace" begin
        sampler = setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update())
        rng = backwards_compat_rng(4)
        _, state = AbstractMCMC.step(rng, model, sampler; n_chains = 6, silent = true, n_tries = 3)
        @test state.adaptive_state isa DEM.DifferentialEvolutionAdaptiveComposite
        for _ in 1:200
            _, state = AbstractMCMC.step_warmup(rng, model, sampler, state)
        end
        adapted = state.adaptive_state.adaptive_states[1]
        @test sum(adapted.L) > 0
        @test !all(==(adapted.p[1]), adapted.p)
        _, state = AbstractMCMC.step(rng, model, sampler, state)
        @test all(isfinite, state.ld)
        @test DEM.n_tries(state.multi_try) == 3
    end

    @testset "templates" begin
        _, state = DREAMz(model, 20; n_burnin = 20, n_tries = 2, save_final_state = true, progress = false, silent = true)
        @test DEM.n_tries(state.multi_try) == 2
    end

    @testset "Float32 positions" begin
        output = sample(
            backwards_compat_rng(5), model, setup_de_update(), 50;
            n_chains = 6, T = Float32, n_tries = 2, sample_kwargs...
        )
        @test eltype(output.samples) == Float32
    end
end
