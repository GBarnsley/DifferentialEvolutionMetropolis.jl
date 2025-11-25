using DifferentialEvolutionMetropolis
using Test
using LogDensityProblems, Random, Distributions, AbstractMCMC, StatsBase
using Aqua
using Logging

# Disable all @info messages globally
disable_logging(Logging.Info)

function backwards_compat_rng(seed)
    if VERSION < v"1.7"
        rng = Random.MersenneTwister(seed)
    else
        rng = Random.Xoshiro(seed)
    end
    return rng
end

@testset "DifferentialEvolutionMetropolis.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(DifferentialEvolutionMetropolis)
    end

    #define some common problems
    struct IsotropicNormalModel{M <: AbstractVector{<:Real}}
        "mean of the isotropic Gaussian"
        mean::M
    end
    function LogDensityProblems.dimension(model::IsotropicNormalModel{<:AbstractVector{<:Real}})
        return length(model.mean)
    end
    function LogDensityProblems.logdensity(model::IsotropicNormalModel, x::AbstractVector{<:Real})
        return - sum(abs2, x .- model.mean) / 2
    end
    LogDensityProblems.capabilities(model::IsotropicNormalModel) = LogDensityProblems.LogDensityOrder{0}()

    struct BendyBananaModel{T <: Real}
        σ::T
    end
    function LogDensityProblems.dimension(model::BendyBananaModel{<:Real})
        return 2
    end
    function LogDensityProblems.logdensity(model::BendyBananaModel, x::AbstractVector{<:Real})
        return logpdf(Normal(0, 1), x[1]) + logpdf(Normal(x[1], model.σ), x[2])
    end
    LogDensityProblems.capabilities(model::BendyBananaModel) = LogDensityProblems.LogDensityOrder{0}()

    include("test_differential_evolution_update.jl")
    include("test_snooker_update.jl")
    include("test_subspace_update.jl")
    include("test_subspace_adaptive_update.jl")
    include("test_composite.jl")
    include("test_rng.jl")
    include("test_temperature.jl")
    include("test_correct.jl")

    @testset "lack of Diagnostics Tools" begin
        @test_logs match_mode = :any (
            :error, "Please load MCMCDiagnosticTools.jl to use `r̂_stopping_criteria`",
        ) sample(
            backwards_compat_rng(1234),
            AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0])),
            setup_de_update(),
            r̂_stopping_criteria;
            progress = false
        )
    end

    using MCMCDiagnosticTools, MCMCChains, FlexiChains
    include("test_templates.jl")
    include("test_convergence.jl")
    #include("test_diagnostics.jl")

    #if VERSION ≥ v"1.11"
    #    @testset "JET" begin
    #        JET.test_package(DifferentialEvolutionMetropolis)
    #    end
    #end
end
