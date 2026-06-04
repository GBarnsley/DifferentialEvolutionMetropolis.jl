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
            ((collect(0:(1 / (n_hot_chains)):1) .^ α) .* (max_temp_pt - 1) .+ 1)[2:end]...,
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

function setup_temperature_struct(xₚ::VV, ldₚ::V, ladder::Vector{Vector{T}}) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    #ensure that final temperatures are in increasing order (for the sampler)
    increasing_indices = sortperm(ladder[end])
    ladder = [step[increasing_indices] for step in ladder]

    cold_chains = findall(x -> x == one(T), ladder[end])

    xₚ_cc_view = view(xₚ, cold_chains)
    ldₚ_cc_view = view(ldₚ, cold_chains)

    n_steps = length(ladder)
    if n_steps == 1
        if length(cold_chains) == length(ladder[1])
            return DifferentialEvolutionNullTemperatureLadder{T}()
        else
            return DifferentialEvolutionStaticTemperatureLadder{T, typeof(xₚ_cc_view), typeof(ldₚ_cc_view)}(ladder[1], cold_chains, xₚ_cc_view, ldₚ_cc_view)
        end
    else
        return DifferentialEvolutionAnnealingTemperatureLadder{T, typeof(xₚ_cc_view), typeof(ldₚ_cc_view)}(
            view(ladder, 1)[1],
            view(ladder, 1:n_steps),
            cold_chains,
            xₚ_cc_view, ldₚ_cc_view
        )
    end
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

function get_temperature(
        ladder::DifferentialEvolutionNullTemperatureLadder{T},
        current_chain::Int
    ) where {T <: Real}
    return one(T)
end

#for parallel tempering
struct DifferentialEvolutionStaticTemperatureLadder{T <: Real, V1 <: SubArray, V2 <: SubArray} <:
    AbstractDifferentialEvolutionTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "indicator for cold chains"
    cold_chains::Vector{Int}
    "view of memory/x"
    xₚ_cc_view::V1
    ldₚ_cc_view::V2
end

#for annealing
struct DifferentialEvolutionAnnealingTemperatureLadder{T <: Real, V1 <: SubArray, V2 <: SubArray} <:
    AbstractDifferentialEvolutionTemperatureLadder{T}
    "temperature for each chain"
    temperature::Vector{T}
    "view of the temperature ladder"
    temperature_ladder::SubArray{Vector{T}, 1, Vector{Vector{T}}}
    "indicator for cold chains"
    cold_chains::Vector{Int}
    "view of memory/x"
    xₚ_cc_view::V1
    ldₚ_cc_view::V2
end

function update_ladder!!(ladder::DifferentialEvolutionAnnealingTemperatureLadder{T, V1, V2}) where {
        T <:
        Real,
        V1 <: SubArray, V2 <: SubArray,
    }
    n_steps = length(ladder.temperature_ladder)
    if n_steps == 1
        if length(ladder.cold_chains) == length(ladder.temperature)
            return DifferentialEvolutionNullTemperatureLadder{T}()
        else
            return DifferentialEvolutionStaticTemperatureLadder{T, V1, V2}(ladder.temperature, ladder.cold_chains, ladder.xₚ_cc_view, ladder.ldₚ_cc_view)
        end
    else
        return DifferentialEvolutionAnnealingTemperatureLadder{T, V1, V2}(
            view(ladder.temperature_ladder, 2)[1],
            view(ladder.temperature_ladder, 2:n_steps),
            ladder.cold_chains,
            ladder.xₚ_cc_view,
            ladder.ldₚ_cc_view
        )
    end
end
