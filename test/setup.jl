using DifferentialEvolutionMetropolis
using Test
using LogDensityProblems, Random, Distributions, AbstractMCMC, StatsBase
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

struct UnitIntervalModel end
LogDensityProblems.dimension(::UnitIntervalModel) = 1
LogDensityProblems.logdensity(::UnitIntervalModel, x::AbstractVector{<:Real}) = 0 <= x[1] <= 1 ? 0.0 : -Inf
LogDensityProblems.capabilities(::UnitIntervalModel) = LogDensityProblems.LogDensityOrder{0}()

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

plain(x) = sprint(show, MIME"text/plain"(), x)
compact(x) = sprint(show, x; context = :compact => true)

mutable struct CountingNaNAtModel
    count::Int
    nan_at::Int
    value::Float64
end
LogDensityProblems.dimension(::CountingNaNAtModel) = 1
LogDensityProblems.capabilities(::CountingNaNAtModel) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(model::CountingNaNAtModel, x)
    model.count += 1
    return model.count == model.nan_at ? NaN : model.value
end

struct IntervalModel
    outside::Float64
end
LogDensityProblems.dimension(::IntervalModel) = 1
LogDensityProblems.capabilities(::IntervalModel) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(model::IntervalModel, x) = 0 <= x[1] <= 1 ? 0.0 : model.outside
