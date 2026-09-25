using DifferentialEvolutionMetropolis
using Test
using Random, AbstractMCMC, LogDensityProblems, Distributions, LinearAlgebra, Logging
using Pathfinder, ForwardDiff

struct PathfinderBimodalModel{D}
    dist::D
end
LogDensityProblems.dimension(model::PathfinderBimodalModel) = length(model.dist)
LogDensityProblems.logdensity(model::PathfinderBimodalModel, x) = logpdf(model.dist, x)
LogDensityProblems.capabilities(::Type{<:PathfinderBimodalModel}) = LogDensityProblems.LogDensityOrder{0}()

@testset "Pathfinder initialisation" begin
    model = PathfinderBimodalModel(MixtureModel([MvNormal([-6.0, 0.0], I), MvNormal([6.0, 0.0], I)]))
    wrapped_model = AbstractMCMC.LogDensityModel(model)
    spl = setup_sampler_scheme(setup_de_update(), setup_snooker_update())
    n_chains = 4

    pf_left = pathfinder(model; rng = backwards_compat_rng(1), init = [-5.0, 0.0])
    pf_right = pathfinder(model; rng = backwards_compat_rng(2), init = [5.0, 0.0])
    mpf = with_logger(NullLogger()) do
        multipathfinder(model, 50; nruns = 4, rng = backwards_compat_rng(3))
    end

    initialise(ip; kwargs...) = last(
        AbstractMCMC.step(
            backwards_compat_rng(4), wrapped_model, spl;
            n_chains = n_chains, initial_position = ip, silent = true, kwargs...
        )
    )

    @testset "Single PathfinderResult" begin
        state = initialise(pf_left; N₀ = 20)
        @test length(state.x) == n_chains
        @test state.memory.fill.position == 20
        @test all(x -> x[1] < 0, state.memory.mem_x[1:20])
        @test isempty(intersect(state.x, state.memory.mem_x[1:20]))

        state_default = initialise(pf_left)
        @test state_default.memory.fill.position == 2 * n_chains
        @test state_default.memory.mem_x[1:n_chains] == state_default.x

        state_memoryless = initialise(pf_left; memory = false)
        @test length(state_memoryless.x) == n_chains
    end

    @testset "Reproducible for a fixed rng" begin
        for ip in (pf_left, (pf_left, pf_right), mpf)
            state_a = initialise(ip; N₀ = 12)
            state_b = initialise(ip; N₀ = 12)
            @test state_a.x == state_b.x
            @test state_a.memory.mem_x[1:12] == state_b.memory.mem_x[1:12]
        end
    end

    @testset "Tuple or vector of PathfinderResults" begin
        for ip in ((pf_left, pf_right), [pf_left, pf_right])
            state = initialise(ip; N₀ = 40)
            @test sign.(first.(state.x)) == [-1.0, 1.0, -1.0, 1.0]
            mem = state.memory.mem_x[1:40]
            @test any(x -> x[1] < 0, mem) && any(x -> x[1] > 0, mem)
        end

        state_unstratified = last(
            AbstractMCMC.step(
                backwards_compat_rng(4), wrapped_model, spl;
                n_chains = 20, initial_position = (pf_left, pf_right),
                stratify_initial_position = false, silent = true
            )
        )
        @test length(state_unstratified.x) == 20
        @test sign.(first.(state_unstratified.x)) != repeat([-1.0, 1.0], 10)

        @test_throws ArgumentError initialise(typeof(pf_left)[])
    end

    @testset "MultiPathfinderResult" begin
        component_signs = sign.(first.(mean.(components(mpf.fit_distribution))))
        state = initialise(mpf; N₀ = 40)
        @test sign.(first.(state.x)) == component_signs
        draws = collect(eachcol(mpf.draws))
        @test all(x -> x in draws, state.memory.mem_x[1:40])

        state_topped_up = initialise(mpf; N₀ = 60)
        mem = state_topped_up.memory.mem_x[1:60]
        @test count(x -> x in draws, mem) == 50

        state_unstratified = initialise(mpf; N₀ = 40, stratify_initial_position = false)
        @test length(state_unstratified.x) == n_chains
    end

    @testset "Errors" begin
        model_3d = PathfinderBimodalModel(MvNormal(zeros(3), I))
        pf_3d = pathfinder(model_3d; rng = backwards_compat_rng(5))
        @test_throws ErrorException initialise(pf_3d)
        @test_throws ArgumentError initialise((1, 2))
    end

    @testset "Templates" begin
        result = deMCzs(
            wrapped_model, 100;
            initial_position = (pf_left, pf_right), n_chains = n_chains, N₀ = 20,
            rng = backwards_compat_rng(6), silent = true, progress = false
        )
        @test size(result.samples) == (100, n_chains, 2)
    end
end
