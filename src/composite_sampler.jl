"""
Create a composite sampler scheme from multiple differential evolution update steps.

The type of scheme depends on `w`:
A vector of weights gives a random order scheme, see [`setup_random_order_scheme`](@ref).
An integer `n` with two updates gives a fixed order scheme that runs the first update `n` times
  and then the second update once, see [`setup_fixed_order_scheme`](@ref).

Schemes can be passed as updates to other schemes, see [`setup_fixed_order_scheme`](@ref) for
how nested schemes select their updates.

# Arguments
- `updates...`: One or more differential evolution sampler objects created by functions
  like [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).

# Keyword Arguments
- `w`: Vector of weights for each update step, or an integer representing the number of first updates for each update using the second update.
  Defaults to equal weights for random updates.

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

# Snooker once after every 9 DE updates
sampler4 = setup_sampler_scheme(setup_de_update(), setup_snooker_update(); w = 9)
```

See also [`setup_random_order_scheme`](@ref), [`setup_fixed_order_scheme`](@ref).
"""
function setup_sampler_scheme(
        updates::AbstractDifferentialEvolutionSampler...;
        w::Union{Vector{<:Real}, Integer} = ones(length(updates))
    )
    return setup_sampler_scheme(w, updates...)
end

function setup_sampler_scheme(w::Vector{<:Real}, updates::AbstractDifferentialEvolutionSampler...)
    return setup_random_order_scheme(updates...; w = w)
end

function setup_sampler_scheme(
        n::Integer, main::AbstractDifferentialEvolutionSampler, secondary::AbstractDifferentialEvolutionSampler
    )
    return setup_fixed_order_scheme(main, secondary; n = n)
end

function setup_sampler_scheme(::Integer, updates::AbstractDifferentialEvolutionSampler...)
    return error("An integer `w` requires exactly two updates, got $(length(updates)). Use a vector of weights for a random order scheme.")
end

"""
Create a composite sampler scheme that picks updates at random.

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
- A `DifferentialEvolutionCompositeSamplerRandom` that can be used with `AbstractMCMC.sample`.

# Examples
```@example random_order_scheme
using DifferentialEvolutionMetropolis

# Snooker 10% of the time, DE 90% of the time
sampler = setup_random_order_scheme(setup_de_update(), setup_snooker_update(); w = [0.9, 0.1])
```

See also [`setup_sampler_scheme`](@ref), [`setup_fixed_order_scheme`](@ref).
"""
function setup_random_order_scheme(
        updates::AbstractDifferentialEvolutionSampler...;
        w::Vector{<:Real} = ones(length(updates))
    )
    return DifferentialEvolutionCompositeSamplerRandom(updates, w)
end

"""
Create a composite sampler scheme that alternates between two updates in a fixed order.

Each cycle runs the `main` update for `n` iterations and then the `secondary` update
for one iteration. Either update can itself be a composite scheme, so fixed and random order
schemes can be nested to build more complex mixes.

# Nesting
Each iteration, the outermost scheme picks one of its updates. If that update is a scheme, it
then picks one of its own updates, and so on. A nested fixed order scheme moves to the next
position in its cycle only on the iterations where it is picked, so its position counts how
often it has been picked, not the total number of iterations. For updates `A`, `B` and `C`:

- `setup_fixed_order_scheme(setup_fixed_order_scheme(A, B), C)` runs `A, C, B, C, A, C, B, C, ...`
- `setup_fixed_order_scheme(setup_fixed_order_scheme(A, B), C; n = 2)` runs `A, B, C, A, B, C, ...`
- `setup_fixed_order_scheme(setup_random_order_scheme(A, B), C)` runs `A or B, C, A or B, C, ...`
- `setup_random_order_scheme(setup_fixed_order_scheme(A, B), C)` alternates `A` and `B` on the iterations
  where the fixed order branch is picked, e.g. `A, C, C, B, A, C, B, ...`

The position in the cycle is stored in the sampler state, so it carries over from warm-up into
sampling and through [`fix_sampler_state`](@ref). Each call to `sample` starts a new cycle, unless
it resumes from an `initial_state`.

# Arguments
- `main`: The update run for `n` iterations of each cycle.
- `secondary`: The update run once at the end of each cycle.

# Keyword Arguments
- `n`: Number of `main` iterations for each `secondary` iteration. Must be at least 1.
  Defaults to 1, alternating updates.

# Returns
- A `DifferentialEvolutionCompositeSamplerFixedOrder` that can be used with `AbstractMCMC.sample`.

# Examples
```@example fixed_order_scheme
using DifferentialEvolutionMetropolis

# Snooker update once for every 10 DE updates
sampler1 = setup_fixed_order_scheme(setup_de_update(), setup_snooker_update(); n = 10)

# Every 5th iteration, randomly pick between a snooker update and a subspace update
sampler2 = setup_fixed_order_scheme(
    setup_de_update(),
    setup_random_order_scheme(setup_snooker_update(), setup_subspace_sampling());
    n = 4
)
```

See also [`setup_sampler_scheme`](@ref), [`setup_random_order_scheme`](@ref).
"""
function setup_fixed_order_scheme(
        main::AbstractDifferentialEvolutionSampler,
        secondary::AbstractDifferentialEvolutionSampler;
        n::Integer = 1
    )
    return DifferentialEvolutionCompositeSamplerFixedOrder((main, secondary), Int(n))
end

struct DifferentialEvolutionCompositeSamplerRandom{
        T <: Real, S <: Tuple{Vararg{AbstractDifferentialEvolutionSampler}},
    } <: AbstractDifferentialEvolutionSampler
    updates::S
    update_weights::Vector{T}
    function DifferentialEvolutionCompositeSamplerRandom(
            updates::S, update_weights::Vector{T}
        ) where {T <: Real, S <: Tuple{Vararg{AbstractDifferentialEvolutionSampler}}}
        if length(update_weights) != length(updates)
            error("Number of update weights must be equal to the number of updates")
        end
        if any(update_weights .< 0)
            error("Update weights must be non-negative")
        end
        return new{T, S}(updates, update_weights)
    end
end

struct DifferentialEvolutionCompositeSamplerFixedOrder{
        S <: Tuple{AbstractDifferentialEvolutionSampler, AbstractDifferentialEvolutionSampler},
    } <: AbstractDifferentialEvolutionSampler
    updates::S
    n::Int
    function DifferentialEvolutionCompositeSamplerFixedOrder(updates::S, n::Int) where {S}
        if n < 1
            error("n must be at least 1")
        end
        return new{S}(updates, n)
    end
end

const DifferentialEvolutionCompositeSampler = Union{
    DifferentialEvolutionCompositeSamplerRandom, DifferentialEvolutionCompositeSamplerFixedOrder,
}

mutable struct DifferentialEvolutionOrderState{S <: Tuple}
    "number of times a fixed order scheme has been stepped, unused by random order schemes"
    iteration::Int
    "order states of the scheme's updates"
    order_states::S
end

# `nothing` unless the sampler contains a fixed order scheme
initialize_order_state(::AbstractDifferentialEvolutionSampler) = nothing
initialize_order_state(sampler::DifferentialEvolutionCompositeSamplerFixedOrder) =
    DifferentialEvolutionOrderState(0, map(initialize_order_state, sampler.updates))

function initialize_order_state(sampler::DifferentialEvolutionCompositeSamplerRandom)
    order_states = map(initialize_order_state, sampler.updates)
    return all(isnothing, order_states) ? nothing : DifferentialEvolutionOrderState(0, order_states)
end

inner_order_states(::Nothing, sampler::DifferentialEvolutionCompositeSampler) = map(_ -> nothing, sampler.updates)
inner_order_states(order_state::DifferentialEvolutionOrderState, ::DifferentialEvolutionCompositeSampler) =
    order_state.order_states

# the same state with its order state replaced, used to pass a nested scheme its own position
function with_order_state(state::DifferentialEvolutionState, order_state)
    return DifferentialEvolutionState(
        state.x, state.ld, state.rngs, state.adaptive_state, state.temperature_ladder, state.memory,
        state.chain_models, state.multi_try, state.xₚ, state.ldₚ, state.x_smpl_view, state.ld_smpl_view,
        state.xₚ_smpl_view, state.ldₚ_smpl_view, order_state
    )
end

select_update!(rng::AbstractRNG, sampler::DifferentialEvolutionCompositeSamplerRandom, order_state) =
    wsample(rng, sampler.update_weights)

function select_update!(
        ::AbstractRNG, sampler::DifferentialEvolutionCompositeSamplerFixedOrder, order_state::DifferentialEvolutionOrderState
    )
    update_id = rem(order_state.iteration, sampler.n + 1) < sampler.n ? 1 : 2
    order_state.iteration += 1
    return update_id
end

supports_multi_try(s::DifferentialEvolutionCompositeSampler) = any(supports_multi_try, s.updates)

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
    order_state = state.order_state
    sampler_id = select_update!(rng, sampler, order_state)

    sample, new_substate = step_selected(
        sampler_id, sampler.updates, inner_order_states(order_state, sampler), rng, model_wrapper, state; kwargs...
    )
    return sample, with_order_state(new_substate, order_state)
end

# walk the tuples instead of indexing them, so each call sees a concrete update type
function step_selected(
        i::Int, updates::Tuple, order_states::Tuple, rng::AbstractRNG, model_wrapper::LogDensityModel,
        state::DifferentialEvolutionState; kwargs...
    )
    if i == 1
        return step(rng, model_wrapper, first(updates), with_order_state(state, first(order_states)); kwargs...)
    end
    return step_selected(
        i - 1, Base.tail(updates), Base.tail(order_states), rng, model_wrapper, state; kwargs...
    )
end

function step_warmup_selected(
        i::Int, updates::Tuple, adaptive_states::Tuple, order_states::Tuple, rng::AbstractRNG,
        model_wrapper::LogDensityModel, state::DifferentialEvolutionState; kwargs...
    )
    if i == 1
        substate = update_state(
            state; adaptive_state = first(adaptive_states), order_state = first(order_states),
            temperature_ladder = state.temperature_ladder
        )
        return step_warmup(rng, model_wrapper, first(updates), substate; kwargs...)
    end
    return step_warmup_selected(
        i - 1, Base.tail(updates), Base.tail(adaptive_states), Base.tail(order_states),
        rng, model_wrapper, state; kwargs...
    )
end

#if there are adaptive states, we need to keep track of them
struct DifferentialEvolutionAdaptiveComposite{
        T <: Real,
        # at least one state, otherwise `T` is unbound on Julia < 1.12 (Aqua unbound_args)
        A <: Tuple{
            AbstractDifferentialEvolutionAdaptiveState{T},
            Vararg{AbstractDifferentialEvolutionAdaptiveState{T}},
        },
    } <: AbstractDifferentialEvolutionAdaptiveState{T}

    adaptive_states::A
end

# also need to update the adaptive state
"""
    step_warmup(rng, model_wrapper, sampler, state; kwargs...)

Perform a single MCMC step during warm-up for composite samplers.

The scheme selects one of its updates (at random for [`setup_random_order_scheme`](@ref), by its
position in the cycle for [`setup_fixed_order_scheme`](@ref)), performs a warm-up step with
that update, and adapts only that update's adaptive state.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Composite differential evolution sampler
- `state`: Current state with the composite adaptive parameters

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions. Defaults to `true`.
  This is forwarded to the inner step, and the outer state's memory is updated without duplicating writes.
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
    sampler_id = select_update!(rng, sampler, state.order_state)

    sample, new_substate = step_warmup_selected(
        sampler_id, sampler.updates, state.adaptive_state.adaptive_states, inner_order_states(state.order_state, sampler),
        rng, model_wrapper, state; update_memory = update_memory, kwargs...
    )

    return sample,
        update_state(
            state; new_state = new_substate,
            update_memory = false,
            memory = new_substate.memory,
            temperature_ladder = new_substate.temperature_ladder
        )
end

function fix_sampler(
        sampler::DifferentialEvolutionCompositeSamplerRandom,
        adaptive_state::DifferentialEvolutionAdaptiveComposite
    )
    return DifferentialEvolutionCompositeSamplerRandom(
        map(fix_sampler, sampler.updates, adaptive_state.adaptive_states), sampler.update_weights
    )
end

function fix_sampler(
        sampler::DifferentialEvolutionCompositeSamplerFixedOrder,
        adaptive_state::DifferentialEvolutionAdaptiveComposite
    )
    return DifferentialEvolutionCompositeSamplerFixedOrder(
        map(fix_sampler, sampler.updates, adaptive_state.adaptive_states), sampler.n
    )
end

function initialize_adaptive_state(
        sampler::DifferentialEvolutionCompositeSampler,
        model_wrapper::LogDensityModel, n_chains::Int, ::Type{T}
    ) where {T <: Real}
    adaptive_states = map(s -> initialize_adaptive_state(s, model_wrapper, n_chains, T), sampler.updates)

    if all(s -> s isa DifferentialEvolutionAdaptiveStatic, adaptive_states)
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        return DifferentialEvolutionAdaptiveComposite{T, typeof(adaptive_states)}(adaptive_states)
    end
end

chains_required(s::DifferentialEvolutionCompositeSampler) = maximum(chains_required, s.updates)
