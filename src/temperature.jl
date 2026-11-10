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
            return DEMNullTemperatureLadder{T}()
        else
            return DEMStaticTemperatureLadder{T}(ladder[1], cold_chains)
        end
    else
        return DEMAnnealingTemperatureLadder{T}(
            ladder[1],
            view(ladder, 1:n_steps),
            cold_chains
        )
    end
end

function setup_view(x::VV, ld::V, ladder::AbstractDEMTemperatureLadder{T}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return view(x, ladder.cold_chains), view(ld, ladder.cold_chains)
end


function get_temperature(ladder::AbstractDEMTemperatureLadder, current_chain::Int)
    return ladder.temperature[current_chain]
end

function update_ladder!!(ladder::AbstractDEMTemperatureLadder)
    return ladder
end

struct DEMNullTemperatureLadder{T <: Real} <:
    AbstractDEMTemperatureLadder{T}
end

function setup_view(x::VV, ld::V, ladder::DEMNullTemperatureLadder{T}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return view(x, :), view(ld, :)
end

function get_temperature(
        ladder::DEMNullTemperatureLadder{T},
        current_chain::Int
    ) where {T <: Real}
    return one(T)
end

#for parallel tempering
struct DEMStaticTemperatureLadder{T <: Real} <:
    AbstractDEMTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "indicator for cold chains"
    cold_chains::Vector{Int}
end

#for annealing
struct DEMAnnealingTemperatureLadder{T <: Real} <:
    AbstractDEMTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "view of the temperature ladder"
    temperature_ladder::SubArray{Vector{T}, 1, Vector{Vector{T}}}
    "indicator for cold chains"
    cold_chains::Vector{Int}
end

function update_ladder!!(ladder::DEMAnnealingTemperatureLadder{T}) where {
        T <:
        Real,
    }
    n_steps = length(ladder.temperature_ladder)
    if n_steps == 1
        if length(ladder.cold_chains) == length(ladder.temperature)
            return DEMNullTemperatureLadder{T}()
        else
            return DEMStaticTemperatureLadder{T}(ladder.temperature, ladder.cold_chains)
        end
    else
        return DEMAnnealingTemperatureLadder{T}(
            ladder.temperature_ladder[2],
            view(ladder.temperature_ladder, 2:n_steps),
            ladder.cold_chains
        )
    end
end
