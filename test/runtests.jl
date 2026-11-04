using DifferentialEvolutionMetropolis
using Test
using LogDensityProblems, Random, Distributions, AbstractMCMC, StatsBase
using Aqua
using Logging

# Disable all @info messages globally
disable_logging(Logging.Info)
# Progress bars print one line per update in non-terminal logs such as CI
AbstractMCMC.setprogress!(false; silent = true)

function backwards_compat_rng(seed)
    if VERSION < v"1.7"
        rng = Random.MersenneTwister(seed)
    else
        rng = Random.Xoshiro(seed)
    end
    return rng
end

const NO_SWAP_WARNING = r"Parallel tempering without a swap update"
const MEMORY_HOT_CHAINS_WARNING = r"Memory-based samplers do not typically require hot chains"
const SINGLE_WORKER_WARNING = r"Only a single (thread|process) available"
const FIXED_CROSSOVER_WARNING = r"sampler already has a fixed crossover probability"

struct WarningFilter{L <: AbstractLogger} <: AbstractLogger
    logger::L
    patterns::Vector{Regex}
end
Logging.min_enabled_level(f::WarningFilter) = Logging.min_enabled_level(f.logger)
Logging.shouldlog(f::WarningFilter, args...) = Logging.shouldlog(f.logger, args...)
Logging.catch_exceptions(f::WarningFilter) = Logging.catch_exceptions(f.logger)
function Logging.handle_message(f::WarningFilter, level, message, args...; kwargs...)
    level == Logging.Warn && any(p -> occursin(p, string(message)), f.patterns) && return nothing
    return Logging.handle_message(f.logger, level, message, args...; kwargs...)
end

"""
    suppress_warnings(f, patterns::Regex...)

Run `f()` dropping only the warnings whose message matches one of `patterns`, for tests whose
setup deliberately triggers a warning that is not under test. Other log messages pass through.
"""
suppress_warnings(f, patterns::Regex...) =
    with_logger(f, WarningFilter(current_logger(), collect(patterns)))

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
    include("test_scratch.jl")
    include("test_composite.jl")
    include("test_rng.jl")
    include("test_threadsafe.jl")
    include("test_parallel.jl")
    include("test_swapping.jl")
    include("test_temperature.jl")
    include("test_swap_update.jl")
    include("test_hmc.jl")
    include("test_hmc_memory_metric.jl")
    include("test_hmc_cluster_pooled_metric.jl")
    include("test_hmc_per_cluster_metric.jl")
    include("test_types.jl")
    include("test_show.jl")
    include("test_memory_accounting.jl")
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
    include("test_logdensity.jl")
    include("test_convergence.jl")
    include("test_pathfinder.jl")
    #include("test_diagnostics.jl")

    #if VERSION ≥ v"1.11"
    #    @testset "JET" begin
    #        JET.test_package(DifferentialEvolutionMetropolis)
    #    end
    #end
end
