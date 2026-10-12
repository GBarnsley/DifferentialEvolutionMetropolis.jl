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
        w::AbstractVector{<:Real} = ones(length(updates))
    )
    return DifferentialEvolutionCompositeSampler(collect(updates), collect(w))
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

    return apply_update(sampler_id, sampler.updates) do update
        step(rng, model_wrapper, update, state; kwargs...)
    end
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

const DifferentialEvolutionSchemeSampler = Union{
    DifferentialEvolutionCompositeSampler, DifferentialEvolutionFixedOrderSampler,
}

const DifferentialEvolutionAdaptiveScheme{T} = Union{
    DifferentialEvolutionAdaptiveComposite{T}, DifferentialEvolutionAdaptiveFixedOrder{T},
}

function select_update!(
        rng::AbstractRNG, sampler::DifferentialEvolutionCompositeSampler,
        ::DifferentialEvolutionAdaptiveComposite
    )
    return wsample(rng, 1:length(sampler.updates), sampler.update_weights)
end

# Calls `f` on the `i`th elements of the tuples, unrolled so each call has a concrete type
apply_update(f, i::Int, updates::Tuple{}, adaptive_states::Tuple{}...) = error("update index out of range")
function apply_update(f, i::Int, updates::Tuple, adaptive_states::Tuple...)
    i == 1 && return f(first(updates), map(first, adaptive_states)...)
    return apply_update(f, i - 1, Base.tail(updates), map(Base.tail, adaptive_states)...)
end

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionSchemeSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveScheme{T},
        };
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    sampler_id = select_update!(rng, sampler, state.adaptive_state)

    return apply_update(sampler_id, sampler.updates, state.adaptive_state.adaptive_states) do update, adaptive_state
        substate = update_state(
            state; adaptive_state = adaptive_state,
            temperature_ladder = state.temperature_ladder
        )
        sample, new_substate = step(
            rng, model_wrapper, update, substate;
            update_memory = update_memory, kwargs...
        )
        return sample,
            update_state(
                state; new_state = new_substate,
                update_memory = false,
                memory = new_substate.memory,
                temperature_ladder = new_substate.temperature_ladder
            )
    end
end

"""
    step_warmup(rng, model_wrapper, sampler, state; kwargs...)

Perform a single MCMC step during warm-up for composite and fixed order samplers.

The scheme selects one of its updates (randomly for [`setup_sampler_scheme`](@ref), by its
position in the cycle for [`setup_fixed_order_scheme`](@ref)), performs a warm-up step with
that update, and adapts only that update's adaptive state.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Composite or fixed order differential evolution sampler
- `state`: Current state with the scheme's adaptive state

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions. Defaults to `true`.
  This is forwarded to the inner step, and the outer state's memory is updated without duplicating writes.
- `kwargs...`: Additional keyword arguments passed to component update functions

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state with adapted parameters for the selected component

See also [`step_warmup`](@ref), [`setup_sampler_scheme`](@ref), [`setup_fixed_order_scheme`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionSchemeSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveScheme{T},
        };
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    sampler_id = select_update!(rng, sampler, state.adaptive_state)

    return apply_update(sampler_id, sampler.updates, state.adaptive_state.adaptive_states) do update, adaptive_state
        substate = update_state(
            state; adaptive_state = adaptive_state,
            temperature_ladder = state.temperature_ladder
        )
        sample, new_substate = step_warmup(
            rng, model_wrapper, update, substate;
            update_memory = update_memory, kwargs...
        )
        return sample,
            update_state(
                state; new_state = new_substate,
                update_memory = false,
                memory = new_substate.memory,
                temperature_ladder = new_substate.temperature_ladder
            )
    end
end

function fix_sampler(
        sampler::DifferentialEvolutionCompositeSampler,
        adaptive_state::DifferentialEvolutionAdaptiveComposite
    )
    return DifferentialEvolutionCompositeSampler(
        [fix_sampler(s, a) for (s, a) in zip(sampler.updates, adaptive_state.adaptive_states)],
        sampler.update_weights
    )
end

function initialize_adaptive_state(
        sampler::DifferentialEvolutionCompositeSampler,
        model_wrapper::LogDensityModel, n_chains::Int, ::Type{T}
    ) where {T <: Real}
    adaptive_states = [
        initialize_adaptive_state(s, model_wrapper, n_chains, T)
            for s in sampler.updates
    ]
    return composite_adaptive_state(adaptive_states, T)
end

function initialize_nonadaptive_state(
        sampler::DifferentialEvolutionCompositeSampler, ::Type{T}
    ) where {T <: Real}
    return composite_adaptive_state([initialize_nonadaptive_state(s, T) for s in sampler.updates], T)
end

function composite_adaptive_state(adaptive_states::Vector, ::Type{T}) where {T <: Real}
    if all(s -> s isa DifferentialEvolutionAdaptiveStatic, adaptive_states)
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        tuple_states = Tuple(adaptive_states)
        return DifferentialEvolutionAdaptiveComposite{T, typeof(tuple_states)}(tuple_states)
    end
end

chains_required(s::DifferentialEvolutionCompositeSampler) = maximum(chains_required, s.updates)
