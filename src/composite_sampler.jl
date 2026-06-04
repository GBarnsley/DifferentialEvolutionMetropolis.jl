"""
Create a composite sampler scheme from multiple differential evolution update steps.

The update method used in each iteration for each chain is randomly selected from the
provided update steps according to their weights. This allows combining different
sampling strategies (e.g., DE updates with snooker updates) in a single sampler.

# Arguments
- `updates...`: One or more differential evolution sampler objects created by functions
  like [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).

# Keyword Arguments
- `w`: Vector of weights for each update step. If not provided, all updates are chosen
  with equal probability. Weights must be non-negative and will be automatically normalized.

# Returns
- A `DifferentialEvolutionCompositeSampler` that can be used with `AbstractMCMC.sample`.

# Examples
```@example sampler_scheme
using DifferentialEvolutionMetropolis

# Only snooker updates
sampler1 = setup_sampler_scheme(setup_snooker_update())

# DE and Snooker with equal probability
sampler2 = setup_sampler_scheme(setup_de_update(), setup_snooker_update())

# Snooker 10% of the time, DE 90% of the time
sampler3 = setup_sampler_scheme(setup_de_update(), setup_snooker_update(); w = [0.9, 0.1])
```

See also [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).
"""
function setup_sampler_scheme(
        updates::AbstractDifferentialEvolutionSampler...;
        w::Vector{Float64} = ones(length(updates))
    )
    return DifferentialEvolutionCompositeSampler(collect(updates), w)
end

struct DifferentialEvolutionCompositeSampler{
        T <: Real, S <: Tuple{Vararg{AbstractDifferentialEvolutionSampler}},
    } <: AbstractDifferentialEvolutionSampler
    updates::S
    update_weights::Vector{T}
    function DifferentialEvolutionCompositeSampler(
            updates::Vector{A}, update_weights::Vector{T}
        ) where {T <: Real, A <: AbstractDifferentialEvolutionSampler}
        if length(update_weights) != length(updates)
            error("Number of update weights must be equal to the number of updates")
        end
        if any(update_weights .< 0)
            error("Update weights must be non-negative")
        end
        tuple_updates = Tuple(updates)
        return new{T, typeof(tuple_updates)}(tuple_updates, update_weights)
    end
end

# if there are no actually adaptive states, we can just use the static one
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionCompositeSampler,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        kwargs...
    ) where {T <: Real}
    sampler_id = wsample(rng, 1:length(sampler.updates), sampler.update_weights)

    return step(rng, model_wrapper, sampler.updates[sampler_id], state; kwargs...)
end

#if there are adaptive states, we need to keep track of them
struct DifferentialEvolutionAdaptiveComposite{
        T <: Real,
        A <: Tuple{ #compat for <1.12
            AbstractDifferentialEvolutionAdaptiveState{T},
            Vararg{AbstractDifferentialEvolutionAdaptiveState{T}},
        },
    } <: AbstractDifferentialEvolutionAdaptiveState{T}

    adaptive_states::A
end

# no updates to adaptive step
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionCompositeSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveComposite{T},
        };
        kwargs...
    ) where {T <: Real}
    sampler_id = wsample(rng, 1:length(sampler.updates), sampler.update_weights)

    fixed_sampler = fix_sampler(sampler.updates[sampler_id], state.adaptive_state.adaptive_states[sampler_id])

    return step(rng, model_wrapper, fixed_sampler, state; kwargs...)
end

# also need to update the adaptive state
"""
    step_warmup(rng, model_wrapper, sampler, state; kwargs...)

Perform a single MCMC step during warm-up for composite samplers.

For composite samplers, this function randomly selects one of the component update
methods, performs a warm-up step with that method, and updates the corresponding
adaptive state while preserving other component states.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Composite differential evolution sampler
- `state`: Current state with composite adaptive parameters

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers).
  Defaults to `true`. Useful if memory has grown too large.
- `kwargs...`: Additional keyword arguments passed to component update functions

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state with adapted parameters for the selected component

See also [`step_warmup`](@ref), [`setup_sampler_scheme`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionCompositeSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveComposite{T},
        };
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    sampler_id = wsample(rng, 1:length(sampler.updates), sampler.update_weights)

    fixed_sampler = fix_sampler(sampler.updates[sampler_id], state.adaptive_state.adaptive_states[sampler_id])

    fixed_state = update_state(
        state; adaptive_state = state.adaptive_state.adaptive_states[sampler_id],
        temperature_ladder = state.temperature_ladder
    )

    sample,
        substate = step_warmup(rng, model_wrapper, fixed_sampler, fixed_state; kwargs...)

    return sample,
        update_state(
            state; new_state = substate,
            update_memory = update_memory, temperature_ladder = substate.temperature_ladder
        )
end

function initialize_adaptive_state(
        sampler::DifferentialEvolutionCompositeSampler,
        model_wrapper::LogDensityModel, n_chains::Int
    )
    adaptive_states = [
        initialize_adaptive_state(s, model_wrapper, n_chains)
            for s in sampler.updates
    ]
    T = Float64

    if all(s -> s isa DifferentialEvolutionAdaptiveStatic, adaptive_states)
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        tuple_states = Tuple(adaptive_states)
        return DifferentialEvolutionAdaptiveComposite{T, typeof(tuple_states)}(Tuple(adaptive_states))
    end
end
