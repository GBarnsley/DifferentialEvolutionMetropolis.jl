function create_temperature_ladder(
        n_cold_chains::Int, n_hot_chains::Int, α::T, max_temp_pt::T,
        max_temp_sa::T, annealing_steps::Int
    ) where {T <: Real}
    cold_chains = ones(T, n_cold_chains)
    if n_hot_chains == 0
        final_temperature = cold_chains
    else
        final_temperature = [
            cold_chains...,
            ((range(zero(T), one(T); length = n_hot_chains + 1) .^ α) .* (max_temp_pt - 1) .+ 1)[2:end]...,
        ]
    end

    if annealing_steps > 0
        step_size = (max_temp_sa .- final_temperature) ./ annealing_steps
        ladder = [max_temp_sa .- step_size .* step for step in 0:annealing_steps]
        ladder[end] = final_temperature
        return ladder
    else
        return [final_temperature]
    end
end

function setup_temperature_struct(ladder::Vector{Vector{T}}) where {T <: Real}
    #ensure that final temperatures are in increasing order (for the sampler)
    increasing_indices = sortperm(ladder[end])
    ladder = [step[increasing_indices] for step in ladder]

    cold_chains = findall(x -> x == one(T), ladder[end])

    n_steps = length(ladder)
    if n_steps == 1
        if length(cold_chains) == length(ladder[1])
            return DifferentialEvolutionNullTemperatureLadder{T}()
        else
            return DifferentialEvolutionStaticTemperatureLadder{T}(ladder[1], cold_chains)
        end
    else
        return DifferentialEvolutionAnnealingTemperatureLadder{T}(
            ladder[1],
            view(ladder, 1:n_steps),
            cold_chains,
            collect(eachindex(ladder[1]))
        )
    end
end

function setup_view(x::VV, ld::V, ladder::AbstractDifferentialEvolutionTemperatureLadder{T}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return view(x, ladder.cold_chains), view(ld, ladder.cold_chains)
end


function get_temperature(ladder::AbstractDifferentialEvolutionTemperatureLadder, current_chain::Int)
    return ladder.temperature[current_chain]
end

function update_ladder!!(ladder::AbstractDifferentialEvolutionTemperatureLadder)
    return ladder
end

struct DifferentialEvolutionNullTemperatureLadder{T <: Real} <:
    AbstractDifferentialEvolutionTemperatureLadder{T}
end

function setup_view(x::VV, ld::V, ladder::DifferentialEvolutionNullTemperatureLadder{T}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return view(x, :), view(ld, :)
end

function get_temperature(
        ladder::DifferentialEvolutionNullTemperatureLadder{T},
        current_chain::Int
    ) where {T <: Real}
    return one(T)
end

#for parallel tempering
struct DifferentialEvolutionStaticTemperatureLadder{T <: Real} <:
    AbstractDifferentialEvolutionTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "indicator for cold chains"
    cold_chains::Vector{Int}
end

#for annealing
struct DifferentialEvolutionAnnealingTemperatureLadder{T <: Real} <:
    AbstractDifferentialEvolutionTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "view of the temperature ladder"
    temperature_ladder::SubArray{Vector{T}, 1, Vector{Vector{T}}}
    "indicator for cold chains"
    cold_chains::Vector{Int}
    "chain indices in current temperature order, reused between steps"
    order::Vector{Int}
end

function update_ladder!!(ladder::DifferentialEvolutionAnnealingTemperatureLadder{T}) where {
        T <:
        Real,
    }
    n_steps = length(ladder.temperature_ladder)
    if n_steps == 1
        if length(ladder.cold_chains) == length(ladder.temperature)
            return DifferentialEvolutionNullTemperatureLadder{T}()
        else
            return DifferentialEvolutionStaticTemperatureLadder{T}(ladder.temperature, ladder.cold_chains)
        end
    else
        return DifferentialEvolutionAnnealingTemperatureLadder{T}(
            ladder.temperature_ladder[2],
            view(ladder.temperature_ladder, 2:n_steps),
            ladder.cold_chains,
            ladder.order
        )
    end
end

# Exchange the accepted proposal buffers before sampling, memory writes and ladder advance.
function replica_exchange!(rng, state; enabled::Bool = true)
    enabled || return nothing
    return replica_exchange!(rng, state, state.temperature_ladder)
end

replica_exchange!(rng, state, ::DifferentialEvolutionNullTemperatureLadder) = nothing

# Static temperatures are sorted at setup, so chain index order is rung order.
function replica_exchange!(rng, state, ladder::DifferentialEvolutionStaticTemperatureLadder)
    return exchange_adjacent_rungs!(rng, state, ladder.temperature, eachindex(ladder.temperature))
end

# Custom annealing schedules can reorder chains, so re-sort from the previous step's order.
function replica_exchange!(rng, state, ladder::DifferentialEvolutionAnnealingTemperatureLadder)
    length(ladder.cold_chains) == length(ladder.temperature) && return nothing
    if issorted(ladder.temperature)
        order = eachindex(ladder.temperature)
    else
        order = sortperm!(ladder.order, ladder.temperature; alg = InsertionSort, initialized = true)
    end
    return exchange_adjacent_rungs!(rng, state, ladder.temperature, order)
end

# Random odd/even matching of neighbouring distinct temperatures; a random chain represents each tied rung.
function exchange_adjacent_rungs!(rng, state, temperature, order)
    offset = rand(rng, 1:2)
    rung = 0
    previous = 0
    first = firstindex(order)
    while first <= lastindex(order)
        last = first
        while last < lastindex(order) && temperature[order[last + 1]] == temperature[order[first]]
            last += 1
        end
        rung += 1
        representative = first == last ? order[first] : order[rand(rng, first:last)]
        if rung > offset && iseven(rung - 1 - offset)
            exchange_pair!(rng, state, previous, representative)
        end
        previous = representative
        first = last + 1
    end
    return nothing
end

function exchange_pair!(rng, state, i, j)
    ti = get_temperature(state.temperature_ladder, i)
    tj = get_temperature(state.temperature_ladder, j)
    logratio = (inv(ti) - inv(tj)) * (state.ldₚ[j] - state.ldₚ[i])
    if log(rand(rng)) < min(zero(logratio), logratio)
        state.xₚ[i], state.xₚ[j] = state.xₚ[j], state.xₚ[i]
        state.ldₚ[i], state.ldₚ[j] = state.ldₚ[j], state.ldₚ[i]
        return true
    end
    return false
end
