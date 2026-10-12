"""
Create a composite sampler scheme from multiple differential evolution update steps.

The type of scheme depends on `w`:
- A vector of weights gives a random order scheme, see [`setup_random_order_composite`](@ref).
- An integer `n` with two updates gives a fixed order scheme that runs the first update `n` times
  and then the second update once, see [`setup_fixed_order_composite`](@ref).

Schemes can be passed as updates to other schemes, see [`setup_fixed_order_composite`](@ref) for
how nested schemes select their updates.

# Arguments
- `updates...`: One or more differential evolution sampler objects created by functions
  like [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref).

# Keyword Arguments
- `w`: Vector of weights for each update step, or an integer number of first update iterations for
  each second update iteration. Defaults to equal weights.

# Returns
- A `DifferentialEvolutionCompositeSampler` that can be used with `AbstractMCMC.sample`.

# Examples
```@example composite_scheme
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

See also [`setup_random_order_composite`](@ref), [`setup_fixed_order_composite`](@ref).
"""
function setup_sampler_scheme(
        updates::AbstractDifferentialEvolutionSampler...;
        w::Union{Vector{<:Real}, Integer} = ones(length(updates))
    )
    return setup_sampler_scheme(w, updates...)
end

function setup_sampler_scheme(w::Vector{<:Real}, updates::AbstractDifferentialEvolutionSampler...)
    return setup_random_order_composite(updates...; w = w)
end

function setup_sampler_scheme(
        n::Integer, main::AbstractDifferentialEvolutionSampler, secondary::AbstractDifferentialEvolutionSampler
    )
    return setup_fixed_order_composite(main, secondary; n = n)
end

"""
Create a composite sampler scheme that picks its update at random.

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
```@example random_order_composite
using DifferentialEvolutionMetropolis

# Snooker 10% of the time, DE 90% of the time
sampler = setup_random_order_composite(setup_de_update(), setup_snooker_update(); w = [0.9, 0.1])
```

See also [`setup_sampler_scheme`](@ref), [`setup_fixed_order_composite`](@ref).
"""
function setup_random_order_composite(
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

The position in the cycle is stored in the sampler state, so it continues from warm-up into
sampling and is unaffected by `adapt = false`. [`fix_sampler_state`](@ref) restarts the cycle.

# Nesting
Each iteration, the outermost scheme picks one of its updates. If that update is a scheme, it
then picks one of its own updates, and so on. A nested fixed order scheme moves to the next
position in its cycle only on the iterations where it is picked, so its position counts how
often it has been picked, not the total number of iterations. For updates `A`, `B` and `C`:

| Scheme | Updates run |
|:--|:--|
| `setup_fixed_order_composite(setup_fixed_order_composite(A, B), C)` | `A, C, B, C, A, C, B, C, ...` |
| `setup_fixed_order_composite(setup_fixed_order_composite(A, B), C; n = 2)` | `A, B, C, A, B, C, ...` |
| `setup_fixed_order_composite(setup_random_order_composite(A, B), C)` | `A or B, C, A or B, C, ...` |
| `setup_random_order_composite(setup_fixed_order_composite(A, B), C)` | `A` and `B` alternate on the iterations where the fixed order branch is picked, e.g. `A, C, C, B, A, C, B, ...` |

# Arguments
- `main`: The update run for `n` iterations of each cycle.
- `secondary`: The update run once at the end of each cycle.

# Keyword Arguments
- `n`: Number of `main` iterations for each `secondary` iteration. Must be at least 1.
  Defaults to 1, i.e. the updates alternate.

# Returns
- A `DifferentialEvolutionCompositeSamplerFixedOrder` that can be used with `AbstractMCMC.sample`.

# Examples
```@example fixed_order_composite
using DifferentialEvolutionMetropolis

# Snooker update once for every 10 DE updates
sampler1 = setup_fixed_order_composite(setup_de_update(), setup_snooker_update(); n = 10)

# Every 5th iteration, randomly pick between a snooker update and a subspace update
sampler2 = setup_fixed_order_composite(
    setup_de_update(),
    setup_random_order_composite(setup_snooker_update(), setup_subspace_sampling());
    n = 4
)
```

See also [`setup_sampler_scheme`](@ref), [`setup_random_order_composite`](@ref).
"""
function setup_fixed_order_composite(
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

# if there are no actually adaptive states, we can just use the static one
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionCompositeSamplerRandom,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        kwargs...
    ) where {T <: Real}
    sampler_id = wsample(rng, sampler.update_weights)

    return step(rng, model_wrapper, sampler.updates[sampler_id], state; kwargs...)
end

#if there are adaptive states (or a fixed order), we need to keep track of them
mutable struct DifferentialEvolutionAdaptiveComposite{
        T <: Real,
        A <: Tuple{ #compat for <1.12
            AbstractDifferentialEvolutionAdaptiveState{T},
            Vararg{AbstractDifferentialEvolutionAdaptiveState{T}},
        },
        I <: Union{Int, Nothing},
    } <: AbstractDifferentialEvolutionAdaptiveState{T}
    const adaptive_states::A
    "number of completed iterations, used by fixed order schemes to find their position in the cycle, `nothing` for random order schemes"
    iteration::I
end

function DifferentialEvolutionAdaptiveComposite{T}(adaptive_states::A, iteration::I) where {T <: Real, A, I}
    return DifferentialEvolutionAdaptiveComposite{T, A, I}(adaptive_states, iteration)
end

function select_update!(
        rng::AbstractRNG, sampler::DifferentialEvolutionCompositeSamplerRandom,
        ::DifferentialEvolutionAdaptiveComposite
    )
    return wsample(rng, sampler.update_weights)
end

function select_update!(
        ::AbstractRNG, sampler::DifferentialEvolutionCompositeSamplerFixedOrder,
        adaptive_state::DifferentialEvolutionAdaptiveComposite
    )
    update_id = rem(adaptive_state.iteration, sampler.n + 1) < sampler.n ? 1 : 2
    adaptive_state.iteration += 1
    return update_id
end

# Steps the `i`th update with its own adaptive state, recursing over the tuples so the update has a concrete type
function apply_update(
        step_function, i::Int, updates::Tuple, adaptive_states::Tuple,
        rng::AbstractRNG, model_wrapper::LogDensityModel, state::DifferentialEvolutionState;
        update_memory::Bool, kwargs...
    )
    if i > 1
        return apply_update(
            step_function, i - 1, Base.tail(updates), Base.tail(adaptive_states),
            rng, model_wrapper, state; update_memory = update_memory, kwargs...
        )
    end

    substate = update_state(state; adaptive_state = first(adaptive_states), temperature_ladder = state.temperature_ladder)

    sample,
        new_substate = step_function(
        rng, model_wrapper, first(updates), substate;
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

# the selected update is given its own adaptive state, so nested schemes and adaptive updates keep their progress
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionCompositeSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveComposite{T},
        };
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    sampler_id = select_update!(rng, sampler, state.adaptive_state)

    return apply_update(
        step, sampler_id, sampler.updates, state.adaptive_state.adaptive_states,
        rng, model_wrapper, state; update_memory = update_memory, kwargs...
    )
end

"""
    step_warmup(rng, model_wrapper, sampler, state; kwargs...)

Perform a single MCMC step during warm-up for composite samplers.

The scheme selects one of its updates (at random for [`setup_random_order_composite`](@ref), by its
position in the cycle for [`setup_fixed_order_composite`](@ref)), performs a warm-up step with
that update, and adapts only that update's adaptive state.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Composite differential evolution sampler
- `state`: Current state with the composite adaptive state

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
    sampler_id = select_update!(rng, sampler, state.adaptive_state)

    return apply_update(
        step_warmup, sampler_id, sampler.updates, state.adaptive_state.adaptive_states,
        rng, model_wrapper, state; update_memory = update_memory, kwargs...
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
    return composite_adaptive_state(sampler, adaptive_states, T)
end

function initialize_nonadaptive_state(
        sampler::DifferentialEvolutionCompositeSampler, ::Type{T}
    ) where {T <: Real}
    return composite_adaptive_state(sampler, map(s -> initialize_nonadaptive_state(s, T), sampler.updates), T)
end

# A random order scheme only needs to track its updates' states if any of them are not static
function composite_adaptive_state(
        ::DifferentialEvolutionCompositeSamplerRandom, adaptive_states::Tuple, ::Type{T}
    ) where {T <: Real}
    if all(s -> s isa DifferentialEvolutionAdaptiveStatic, adaptive_states)
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        return DifferentialEvolutionAdaptiveComposite{T}(adaptive_states, nothing)
    end
end

function composite_adaptive_state(
        ::DifferentialEvolutionCompositeSamplerFixedOrder, adaptive_states::Tuple, ::Type{T}
    ) where {T <: Real}
    return DifferentialEvolutionAdaptiveComposite{T}(adaptive_states, 0)
end

chains_required(s::DifferentialEvolutionCompositeSampler) = maximum(chains_required, s.updates)
