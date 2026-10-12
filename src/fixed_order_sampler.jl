"""
Create a sampler scheme that alternates between two updates in a fixed order.

Each cycle runs the `main` update for `n_main` iterations and then the `secondary` update
for one iteration. Either update can itself be a scheme, so fixed order and random
([`setup_sampler_scheme`](@ref)) schemes can be nested to build more complex mixes.

The position in the cycle is stored in the sampler state, so it continues from warm-up into
sampling and is unaffected by `adapt = false`. [`fix_sampler_state`](@ref) restarts the cycle.

# Nesting
Each scheme keeps its own position, and a nested scheme only moves on when the scheme above
it selects it. The outer scheme decides which of its two branches runs, then the nested scheme
decides which of its own updates runs. For example,
`setup_fixed_order_scheme(setup_fixed_order_scheme(A, B), C; n_main = 2)` runs
`A, B, C, A, B, C, ...`, and `setup_sampler_scheme(setup_fixed_order_scheme(A, B), C)` runs
`A` and `B` alternately on the iterations where the random scheme picks the fixed order branch.

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

function select_update!(
        ::AbstractRNG, sampler::DifferentialEvolutionFixedOrderSampler,
        adaptive_state::DifferentialEvolutionAdaptiveFixedOrder
    )
    update_id = rem(adaptive_state.iteration, sampler.n_main + 1) < sampler.n_main ? 1 : 2
    adaptive_state.iteration += 1
    return update_id
end

function fix_sampler(
        sampler::DifferentialEvolutionFixedOrderSampler,
        adaptive_state::DifferentialEvolutionAdaptiveFixedOrder
    )
    return DifferentialEvolutionFixedOrderSampler(
        map(fix_sampler, sampler.updates, adaptive_state.adaptive_states), sampler.n_main
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
