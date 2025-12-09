module DifferentialEvolutionMetropolis
export setup_de_update, setup_snooker_update, setup_subspace_sampling, setup_sampler_scheme
export step, step_warmup, fix_sampler, fix_sampler_state
export DifferentialEvolutionOutput
export deMC, deMCzs, DREAMz
export r̂_stopping_criteria

import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution,
    ContinuousUnivariateDistribution, DiscreteNonParametricSampler
import Distributions: Sampleable, Discrete, Continuous, Univariate, sampler
import Distributions: Dirac, Uniform, DiscreteUniform, Normal, DiscreteNonParametric
import Distributions

import LogDensityProblems: logdensity, dimension
import StatsBase: wsample
import StatsBase
import LinearAlgebra: norm, normalize, dot
import Random: AbstractRNG, default_rng
import Random
import AbstractMCMC: LogDensityModel, AbstractSampler, step, step_warmup, sample, bundle_samples, chainsstack
import AbstractMCMC

abstract type AbstractDifferentialEvolutionSampler <: AbstractSampler end

abstract type AbstractDifferentialEvolutionAdaptiveState{T} end

abstract type AbstractDifferentialEvolutionMemory{T} end

abstract type AbstractDifferentialEvolutionTemperatureLadder{T} end

struct DifferentialEvolutionState{
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
        M <: AbstractDifferentialEvolutionMemory{T},
        V <: AbstractVector{T}, VV <: AbstractVector{V},
    }
    "current position"
    x::VV
    "log density at current position"
    ld::V
    "preallocated next position"
    xₚ::VV
    "preallocated next log densities"
    ldₚ::V
    "random states"
    rngs::Vector{<:AbstractRNG}
    "struct for holding the status of the adaptive scheme"
    adaptive_state::A
    "temperature ladder"
    temperature_ladder::L
    "memory structure"
    memory::M
end

"""
    DifferentialEvolutionOutput{T <: Real}

Container for differential evolution MCMC sampling results.

# Fields
- `samples::Array{T, 3}`: Three-dimensional array of parameter samples with dimensions
  (iterations, chains, parameters). Each sample represents a point in parameter space
  from the MCMC chain.
- `ld::Matrix{T}`: Two-dimensional matrix of log-density values with dimensions
  (iterations, chains). Contains the log-probability density evaluated at each
  corresponding sample point.

# Type Parameters
- `T <: Real`: Numeric type for the samples and log-density values (typically `Float64`).

# Examples
```julia
# Access samples from the output
output = sample(model, sampler, n_samples)
parameter_samples = output.samples  # Shape: (n_samples, n_chains, n_params)
log_densities = output.ld          # Shape: (n_samples, n_chains)

# Extract samples for a specific chain
chain_1_samples = output.samples[:, 1, :]  # All samples from chain 1
```
"""
struct DifferentialEvolutionOutput{T <: Real}
    samples::Array{T, 3}
    ld::Matrix{T}
end

include("docs.jl")
include("temperature.jl")
include("memory.jl")
include("fast_sample.jl")
include("chains.jl")
include("differential_evolution_update.jl")
include("snooker_update.jl")
include("subspace_update.jl")
include("subspace_adaptive_update.jl")
include("composite_sampler.jl")
include("utilities.jl")
include("convergence.jl")
include("templates.jl")


if !isdefined(Base, :get_extension)
    #backwards compat for julia < 1.9
    include("../ext/MCMCDiagnosticToolsExt.jl")
    include("../ext/MCMCChainsExt.jl")
    include("../ext/FlexiChainsExt.jl")
end

end
