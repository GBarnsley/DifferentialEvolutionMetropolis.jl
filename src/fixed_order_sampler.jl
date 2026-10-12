"""
Create a sampler scheme that alternates between two updates in a fixed order.

Each cycle runs the `main` update for `n_main` iterations and then the `secondary` update
for one iteration. Either update can itself be a scheme, so fixed order and random
([`setup_sampler_scheme`](@ref)) schemes can be nested to build more complex mixes.

The position in the cycle is stored in the sampler state, so it continues from warm-up into
sampling and is unaffected by `adapt = false`.

# Arguments
- `main`: The update run for `n_main` iterations of each cycle.
- `secondary`: The update run once at the end of each cycle.

# Keyword Arguments
- `n_main`: Number of `main` iterations for each `secondary` iteration. Must be at least 1.
  Defaults to 1, i.e. the updates alternate.

# Returns
- A `DifferentialEvolutionFixedOrderSampler` that can be used with `AbstractMCMC.sample`.

# Examples
```@example fixed_order_scheme
using DifferentialEvolutionMetropolis

# Snooker update once for every 10 DE updates
sampler1 = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n_main = 10)

# Every 5th iteration, randomly pick between a snooker update and a subspace update
sampler2 = setup_fixed_order_scheme(
    setup_de_update(),
    setup_sampler_scheme(setup_snooker_update(), setup_subspace_sampling());
    n_main = 4
)
```

See also [`setup_sampler_scheme`](@ref), [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).
"""
function setup_fixed_order_scheme(
        main::AbstractDifferentialEvolutionSampler,
        secondary::AbstractDifferentialEvolutionSampler;
        n_main::Integer = 1
    )
    return DifferentialEvolutionFixedOrderSampler((main, secondary), Int(n_main))
end

struct DifferentialEvolutionFixedOrderSampler{
        S <: Tuple{AbstractDifferentialEvolutionSampler, AbstractDifferentialEvolutionSampler},
    } <: AbstractDifferentialEvolutionSampler
    updates::S
    n_main::Int
    function DifferentialEvolutionFixedOrderSampler(updates::S, n_main::Int) where {S}
        if n_main < 1
            error("n_main must be at least 1")
        end
        return new{S}(updates, n_main)
    end
end

mutable struct DifferentialEvolutionAdaptiveFixedOrder{
        T <: Real,
        A <: Tuple{
            AbstractDifferentialEvolutionAdaptiveState{T},
            AbstractDifferentialEvolutionAdaptiveState{T},
        },
    } <: AbstractDifferentialEvolutionAdaptiveState{T}
    const adaptive_states::A
    "number of completed iterations"
    iteration::Int
end

function DifferentialEvolutionAdaptiveFixedOrder{T}(adaptive_states::A, iteration::Int) where {T <: Real, A}
    return DifferentialEvolutionAdaptiveFixedOrder{T, A}(adaptive_states, iteration)
end

function next_update!(adaptive_state::DifferentialEvolutionAdaptiveFixedOrder, n_main::Int)
    update_id = rem(adaptive_state.iteration, n_main + 1) < n_main ? 1 : 2
    adaptive_state.iteration += 1
    return update_id
end

# Runs `step_function` with the next update in the cycle on a state holding that update's own adaptive state
function fixed_order_step(
        step_function, rng::AbstractRNG, model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionFixedOrderSampler, state::DifferentialEvolutionState;
        update_memory::Bool = true, kwargs...
    )
    update_id = next_update!(state.adaptive_state, sampler.n_main)

    substate = update_state(
        state; adaptive_state = state.adaptive_state.adaptive_states[update_id],
        temperature_ladder = state.temperature_ladder
    )

    sample, new_substate = step_function(
        rng, model_wrapper, sampler.updates[update_id], substate;
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

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionFixedOrderSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveFixedOrder{T},
        };
        kwargs...
    ) where {T <: Real}
    return fixed_order_step(step, rng, model_wrapper, sampler, state; kwargs...)
end

"""
    step_warmup(rng, model_wrapper, sampler::DifferentialEvolutionFixedOrderSampler, state; kwargs...)

Perform a single warm-up step with the next update in a fixed order scheme, adapting only that update.

See also [`setup_fixed_order_scheme`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionFixedOrderSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveFixedOrder{T},
        };
        kwargs...
    ) where {T <: Real}
    return fixed_order_step(step_warmup, rng, model_wrapper, sampler, state; kwargs...)
end

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionFixedOrderSampler,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        kwargs...
    ) where {T <: Real}
    return error("A fixed order scheme needs a state that tracks its position in the cycle, initialise it with `step(rng, model_wrapper, sampler)` or use `fix_sampler_state` on the fixed order scheme itself")
end

function fix_sampler(
        sampler::DifferentialEvolutionFixedOrderSampler,
        adaptive_state::DifferentialEvolutionAdaptiveFixedOrder
    )
    return DifferentialEvolutionFixedOrderSampler(
        map(fix_sampler, sampler.updates, adaptive_state.adaptive_states), sampler.n_main
    )
end

function fix_sampler_state(
        sampler::DifferentialEvolutionFixedOrderSampler,
        state::DifferentialEvolutionState{T, <:DifferentialEvolutionAdaptiveFixedOrder{T}}
    ) where {T <: Real}
    fixed_sampler = fix_sampler(sampler, state.adaptive_state)
    adaptive_state = DifferentialEvolutionAdaptiveFixedOrder{T}(
        map(s -> initialize_nonadaptive_state(s, T), fixed_sampler.updates),
        state.adaptive_state.iteration
    )
    return fixed_sampler,
        update_state(
            state; update_memory = false,
            adaptive_state = adaptive_state,
            temperature_ladder = state.temperature_ladder
        )
end

function initialize_adaptive_state(
        sampler::DifferentialEvolutionFixedOrderSampler,
        model_wrapper::LogDensityModel, n_chains::Int, ::Type{T}
    ) where {T <: Real}
    return DifferentialEvolutionAdaptiveFixedOrder{T}(
        map(s -> initialize_adaptive_state(s, model_wrapper, n_chains, T), sampler.updates), 0
    )
end

function initialize_nonadaptive_state(
        sampler::DifferentialEvolutionFixedOrderSampler, ::Type{T}
    ) where {T <: Real}
    return DifferentialEvolutionAdaptiveFixedOrder{T}(
        map(s -> initialize_nonadaptive_state(s, T), sampler.updates), 0
    )
end

chains_required(s::DifferentialEvolutionFixedOrderSampler) = maximum(chains_required, s.updates)
