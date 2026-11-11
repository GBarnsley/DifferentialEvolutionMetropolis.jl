module DifferentialEvolutionMetropolis
export setup_de_update, setup_snooker_update, setup_subspace_sampling, setup_sampler_scheme, setup_random_order_scheme, setup_fixed_order_scheme, setup_swap_update, setup_hmc_update, memory_metric, cluster_pooled_metric, per_cluster_metric
export step, step_warmup
export DEMOutput
export deMC, deMCzs, DREAMz
export r̂_stopping_criteria

import Distributions: UnivariateDistribution, DiscreteUnivariateDistribution,
    ContinuousUnivariateDistribution
import Distributions: Sampleable, Discrete, Continuous, Univariate, sampler
import Distributions: Dirac, Uniform, DiscreteUniform, Normal, DiscreteNonParametric, median
import Distributions
import AliasTables: AliasTable, set_weights!

import LogDensityProblems: logdensity, dimension
import LogExpFunctions: logsumexp, logaddexp
import StatsBase: wsample
import StatsBase
import LinearAlgebra: norm, normalize!, dot
import Random: AbstractRNG, Xoshiro, default_rng, rand!, seed!, rand
import AbstractMCMC: LogDensityModel, step, step_warmup, sample, bundle_samples, chainsstack
import AbstractMCMC

"""
    AbstractDEMSamplerExecutionMode

Supertype indicating the execution mode of an [`DifferentialEvolutionMetropolis.AbstractDEMSampler`](@ref), declaring whether chains can update in parallel with multithreading.
"""
abstract type AbstractDEMSamplerExecutionMode end

"""
    DEMSamplerSerial

Execution mode of an update whose proposals read other chains, so without memory the chains must update in turn (unless multiple tries are used).
"""
struct DEMSamplerSerial <: AbstractDEMSamplerExecutionMode end

"""
    DEMSamplerMultiThread

Execution mode of an update whose chains can update in parallel, even without memory.
"""
struct DEMSamplerMultiThread <: AbstractDEMSamplerExecutionMode end

"""
    AbstractDEMSamplerTryMode

Supertype indicating if an [`DifferentialEvolutionMetropolis.AbstractDEMSampler`](@ref), supports multiple tries.
"""
abstract type AbstractDEMSamplerTryMode end

"""
    DEMSamplerSingleTry

Does not support multiple tries.
"""
struct DEMSamplerSingleTry <: AbstractDEMSamplerTryMode end

"""
    DEMSamplerMultipleTry

Try mode of an update that supports `n_tries > 1`, using the `offset` returned by [`DifferentialEvolutionMetropolis.proposal!`](@ref) as the log Hastings correction of each trial.
"""
struct DEMSamplerMultipleTry <: AbstractDEMSamplerTryMode end

"""
    AbstractDEMMetaSampler

Supertype of every update and sampler scheme in the package. Includes both actual updates [`DifferentialEvolutionMetropolis.AbstractDEMSampler`](@ref) and composite schemes.
"""
abstract type AbstractDEMMetaSampler <: AbstractMCMC.AbstractSampler end

"""
    AbstractDEMSampler{E <: AbstractDEMSamplerExecutionMode, T <: AbstractDEMSamplerTryMode}

Supertype of a single update. A struct inheriting from this type and implementing [`DifferentialEvolutionMetropolis.proposal!`](@ref), will function as an update in this package.
"""
abstract type AbstractDEMSampler{
    E <: AbstractDEMSamplerExecutionMode,
    T <: AbstractDEMSamplerTryMode,
} <: AbstractDEMMetaSampler end

abstract type AbstractDEMCompositeSampler <: AbstractDEMMetaSampler end

"""
    chains_required(sampler::AbstractDEMMetaSampler)

Minimum number of chains `sampler` needs when sampling without memory.

Define a method for a custom update that needs a different number, e.g. `chains_required(::MyUpdate) = 1` for a random walk.
"""
chains_required(::AbstractDEMSampler) = 3  # conservative fallback, COV_EXCL_LINE

# Number of memory positions an update draws, used to give the suggested memory size before sampling
memory_picks(sampler::AbstractDEMSampler) = chains_required(sampler)

"""
    proposal!(xₚ, x, state::DEMState, sampler::AbstractDEMSampler, current_state::Int)

Write a proposal for chain `current_state`, made from position `x`, into `xₚ`.

Every update must implement this.
Use `state.rngs[current_state]` for random draws, as chains may be updated in parallel.

# Returns
- A named tuple with at least `offset`, the log Hastings correction `log(q(x | xₚ) / q(xₚ | x))` (`0.0` for a symmetric proposal, `-Inf` to reject the proposal).
  It may be conditional on draws that do not depend on `x`, such as the chains picked by `pick_chains`.
"""
function proposal! end

"""
    AbstractDEMAdaptiveState{T}

Supertype of the adaptive state of a sampler, held in [`DifferentialEvolutionMetropolis.DEMState`](@ref), with `T` the position element type.
"""
abstract type AbstractDEMAdaptiveState{T} end

abstract type AbstractDEMMemory{T} end

abstract type AbstractDEMTemperatureLadder{T} end

"""
    DEMState

State of all chains, passed between steps and to [`DifferentialEvolutionMetropolis.proposal!`](@ref).

Fields a custom update may read:
- `x`: current position of each chain.
- `ld`: log density at each current position.
- `rngs`: per-chain random number generators.
- `adaptive_state`: the [`DifferentialEvolutionMetropolis.AbstractDEMAdaptiveState`](@ref) of the sampler.

Draw other chains or memory positions with [`DifferentialEvolutionMetropolis.pick_chains`](@ref).
The remaining fields are preallocated buffers and should not be modified.
"""
struct DEMState{
        T <: Real, A <: AbstractDEMAdaptiveState{T},
        L <: AbstractDEMTemperatureLadder{T},
        M <: AbstractDEMMemory{T},
        Model, B,
        V <: AbstractVector{T}, VV <: AbstractVector{V},
        R <: AbstractRNG, V1 <: SubArray, V2 <: SubArray, O,
    }
    "current position"
    x::VV
    "log density at current position"
    ld::V
    "random states"
    rngs::Vector{R}
    "struct for holding the status of the adaptive scheme"
    adaptive_state::A
    "temperature ladder"
    temperature_ladder::L
    "memory structure"
    memory::M
    "per-chain model copies for thread-safe parallel logdensity evaluation"
    chain_models::Vector{Model}
    "preallocated multiple-try trials, `nothing` if multiple-try is disabled"
    multi_try::B
    "preallocated next position"
    xₚ::VV
    "preallocated next log densities"
    ldₚ::V
    "views of x and ld that actually sample from (i.e. the cold chains if parallel tempering)"
    x_smpl_view::V1
    ld_smpl_view::V2
    xₚ_smpl_view::V1
    ldₚ_smpl_view::V2
    "position of each fixed order scheme in its cycle, `nothing` if the sampler has no fixed order scheme"
    order_state::O
end

# Reseed a chain rng from `rng`, Xoshiro states are set directly as `seed!` allocates while hashing the seed
reseed!(chain_rng::AbstractRNG, rng::AbstractRNG) = seed!(chain_rng, rand(rng, UInt))
function reseed!(chain_rng::Xoshiro, rng::AbstractRNG)
    return copy!(chain_rng, Xoshiro(rand(rng, UInt64), rand(rng, UInt64), rand(rng, UInt64), rand(rng, UInt64)))
end

function DEMState(
        rng::R,
        x::VV,
        ld::V,
        adaptive_state::A,
        temperature_ladder::L,
        memory::M,
        chain_models::Vector{Model},
        n_tries::Int,
        order_state
    ) where {
        T <: Real, A <: AbstractDEMAdaptiveState{T},
        L <: AbstractDEMTemperatureLadder{T},
        M <: AbstractDEMMemory{T},
        V <: AbstractVector{T}, VV <: AbstractVector{V},
        R <: AbstractRNG, Model,
    }

    rngs = [reseed!(copy(rng), rng) for _ in 1:length(x)]
    xₚ = copy.(x)
    ldₚ = copy(ld)
    x_smpl_view, ld_smpl_view = setup_view(x, ld, temperature_ladder)
    xₚ_smpl_view, ldₚ_smpl_view = setup_view(xₚ, ldₚ, temperature_ladder)
    return DEMState(
        x,
        ld,
        rngs,
        adaptive_state,
        temperature_ladder,
        memory,
        chain_models,
        multi_try_buffers(x, n_tries),
        xₚ,
        ldₚ,
        x_smpl_view,
        ld_smpl_view,
        xₚ_smpl_view,
        ldₚ_smpl_view,
        order_state
    )
end

"""
    DEMOutput{T <: Real}

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
struct DEMOutput{T <: Real}
    samples::Array{T, 3}
    ld::Matrix{T}
end

include("docs.jl")
include("hmc.jl")
include("temperature.jl")
include("memory.jl")
include("fast_sample.jl")
include("scratch.jl")
include("chains.jl")
include("differential_evolution_update.jl")
include("snooker_update.jl")
include("subspace_update.jl")
include("subspace_adaptive_update.jl")
include("composite_sampler.jl")
include("swap_update.jl")
include("multi_try.jl")
include("utilities.jl")
include("convergence.jl")
include("templates.jl")
include("show.jl")


if !isdefined(Base, :get_extension)
    #backwards compat for julia < 1.9
    include("../ext/MCMCDiagnosticToolsExt.jl")
    include("../ext/MCMCChainsExt.jl")
    include("../ext/FlexiChainsExt.jl")
end

end
