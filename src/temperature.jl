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
            cold_chains
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
            ladder.cold_chains
        )
    end
end

# Exchange the accepted proposal buffers before sampling, memory writes and ladder advance.
# Pair neighbouring *current* temperatures, not neighbouring indices: custom annealing
# schedules can change their ordering. Choose a random representative of each tied rung.
function replica_exchange!(rng, state; enabled::Bool = true)
    ladder = state.temperature_ladder
    if !enabled || ladder isa DifferentialEvolutionNullTemperatureLadder ||
            length(ladder.cold_chains) == length(state.x)
        return nothing
    end
    order = sortperm(ladder.temperature)
    representatives = Int[]
    first = 1
    while first <= length(order)
        last = first
        while last < length(order) && ladder.temperature[order[last + 1]] == ladder.temperature[order[first]]
            last += 1
        end
        push!(representatives, order[rand(rng, first:last)])
        first = last + 1
    end
    # Random odd/even matching is position-independent and each pair is disjoint.
    for k in rand(rng, 1:2):2:(length(representatives) - 1)
        i, j = representatives[k], representatives[k + 1]
        exchange_pair!(rng, state, i, j)
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
