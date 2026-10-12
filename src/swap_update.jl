struct DifferentialEvolutionSwapSampler <: AbstractDifferentialEvolutionSampler end

"""
Set up a parallel tempering swap update.

Each step proposes swapping the positions of chains at adjacent temperatures, accepting
with the tempered Metropolis-Hastings ratio. Every pair of adjacent temperatures is
proposed once per step, in two passes over non-overlapping pairs (the first pass is chosen
at random). When several chains share a temperature (e.g. the cold chains), one of them is
picked at random for each pass. Temperatures, random states and adaptive states stay with
their chains; only positions and their log densities move, so no log density evaluations
are needed.

During annealing, adjacency is based on the current temperatures. Swaps have no effect
when all chains share a temperature, so this update is only useful with hot chains
(`n_hot_chains > 0`).

Intended for use within [`setup_sampler_scheme`](@ref), alongside updates that move the
chains, e.g. with a low weight.

# Returns
- A `DifferentialEvolutionSwapSampler` that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref).

# Example
```@example swap_update
using DifferentialEvolutionMetropolis

# DE updates 90% of the time, swaps 10% of the time
sampler = setup_sampler_scheme(setup_de_update(), setup_swap_update(); w = [0.9, 0.1])
```

See also [`setup_sampler_scheme`](@ref), [`setup_de_update`](@ref).
"""
setup_swap_update() = DifferentialEvolutionSwapSampler()

chains_required(::DifferentialEvolutionSwapSampler) = 2  # COV_EXCL_LINE

has_swap_update(::AbstractDifferentialEvolutionSampler) = false  # COV_EXCL_LINE
has_swap_update(::DifferentialEvolutionSwapSampler) = true  # COV_EXCL_LINE
has_swap_update(s::DifferentialEvolutionCompositeSampler) = any(has_swap_update, s.updates)

has_hot_chains(::DifferentialEvolutionNullTemperatureLadder) = false  # COV_EXCL_LINE
has_hot_chains(ladder::AbstractDifferentialEvolutionTemperatureLadder) =
    length(ladder.cold_chains) < length(ladder.temperature)

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionSwapSampler,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    for i in eachindex(state.x)
        copyto!(state.xₚ[i], state.x[i])
    end
    copyto!(state.ldₚ, state.ld)

    swap_adjacent_temperatures!(rng, state, state.temperature_ladder)

    return create_sample(state),
        update_state(
            state;
            swap_positions = Val(true),
            update_memory = update_memory
        )
end

swap_adjacent_temperatures!(rng, state, ::DifferentialEvolutionNullTemperatureLadder) = nothing

function swap_adjacent_temperatures!(rng, state, ladder::AbstractDifferentialEvolutionTemperatureLadder)
    has_distinct_temperatures(ladder) || return nothing
    first_pass = rand(rng, 1:2)
    for pass in (first_pass, 3 - first_pass)
        swap_pass!(rng, state, ladder.temperature, pass)
    end
    return nothing
end

has_distinct_temperatures(ladder) = !all(==(first(ladder.temperature)), ladder.temperature)

# Final temperatures are sorted at setup, only custom annealing schedules can be out of order.
function swap_pass!(rng, state, temperature, pass)
    if issorted(temperature)
        return swap_pass_sorted!(rng, state, temperature, pass)
    else
        return swap_pass_unsorted!(rng, state, temperature, pass)
    end
end

# Pairs levels (1, 2), (3, 4), ... for pass 1 and (2, 3), (4, 5), ... for pass 2.
pair_with_previous(level, pass) = level > pass && iseven(level - 1 - pass)

function swap_pass_sorted!(rng, state, temperature, pass)
    level = 0
    previous = 0
    first = firstindex(temperature)
    while first <= lastindex(temperature)
        last = first
        while last < lastindex(temperature) && temperature[last + 1] == temperature[first]
            last += 1
        end
        level += 1
        chain = first == last ? first : rand(rng, first:last)
        pair_with_previous(level, pass) && propose_swap!(rng, state, previous, chain)
        previous = chain
        first = last + 1
    end
    return nothing
end

function swap_pass_unsorted!(rng, state, temperature, pass)
    level = 1
    current = minimum(temperature)
    previous = random_chain_at(rng, temperature, current)
    next = next_temperature(temperature, current)
    while next != current
        level += 1
        current = next
        chain = random_chain_at(rng, temperature, current)
        pair_with_previous(level, pass) && propose_swap!(rng, state, previous, chain)
        previous = chain
        next = next_temperature(temperature, current)
    end
    return nothing
end

function next_temperature(temperature, current)
    next = current
    for t in temperature
        if t > current && (next == current || t < next)
            next = t
        end
    end
    return next
end

function random_chain_at(rng, temperature, t)
    n = count(==(t), temperature)
    k = n == 1 ? 1 : rand(rng, 1:n)
    i = firstindex(temperature) - 1
    while k > 0
        i += 1
        temperature[i] == t && (k -= 1)
    end
    return i
end

function propose_swap!(rng, state, i, j)
    tᵢ = get_temperature(state.temperature_ladder, i)
    tⱼ = get_temperature(state.temperature_ladder, j)
    log_ratio = (inv(tᵢ) - inv(tⱼ)) * (state.ldₚ[j] - state.ldₚ[i])
    if log(rand(rng)) < min(zero(log_ratio), log_ratio)
        state.xₚ[i], state.xₚ[j] = state.xₚ[j], state.xₚ[i]
        state.ldₚ[i], state.ldₚ[j] = state.ldₚ[j], state.ldₚ[i]
        return true
    end
    return false
end

# Appends a swap update to a template's scheme, warning about and disabling nonsensical settings.
function add_swap_update(
        sampler_scheme::DifferentialEvolutionCompositeSampler, n_hot_chains::Int,
        use_parallel_tempering_swaps::Bool, parallel_tempering_swap_probability::Real
    )
    use_parallel_tempering_swaps || return sampler_scheme
    if !(0 <= parallel_tempering_swap_probability < 1)
        error("parallel_tempering_swap_probability must be in [0, 1), got $parallel_tempering_swap_probability")
    elseif n_hot_chains == 0
        @warn "use_parallel_tempering_swaps = true has no effect without hot chains (n_hot_chains = 0), disabling swaps."
        return sampler_scheme
    elseif parallel_tempering_swap_probability == 0
        @warn "use_parallel_tempering_swaps = true but parallel_tempering_swap_probability = 0, disabling swaps."
        return sampler_scheme
    end
    p = Float64(parallel_tempering_swap_probability)
    w = sampler_scheme.update_weights
    return setup_sampler_scheme(
        sampler_scheme.updates..., setup_swap_update();
        w = [Float64.(w) .* ((1 - p) / sum(w)); p]
    )
end

function check_swap_setup(sampler::AbstractDifferentialEvolutionSampler, ladder)
    return if has_hot_chains(ladder) && !has_swap_update(sampler)
        @warn "Parallel tempering without a swap update, the chains may mix better with swaps. Add `setup_swap_update()` to the sampler scheme (or set `use_parallel_tempering_swaps = true` in the templates)."
    elseif !has_hot_chains(ladder) && has_swap_update(sampler)
        @warn "The sampler includes a swap update but there are no hot chains, swaps will have no effect."
    end
end
