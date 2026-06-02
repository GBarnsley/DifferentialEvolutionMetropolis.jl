#uses preallocated for anything less than 4 chains (which the most default methods ask for)
function fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int},
        current_chain::Int
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    if n_chains > length(indices)
        @warn "Picking $n_chains chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains`." maxlog = 1
        resize!(indices, n_chains)
        resize!(ordered_indices, n_chains)
    end
    _fast_sample_chains!(
        rng,
        x,
        max_length,
        n_chains,
        indices,
        ordered_indices,
        current_chain
    )
    return nothing
end
function _fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int},
        current_chain::Int
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    ordered_indices[1] = current_chain

    for i in 1:(n_chains - 1)
        idx = rand(rng, 1:(max_length - i))
        insert_pos = i + 1

        for j in 1:i
            if ordered_indices[j] ≤ idx
                idx += 1
            else
                insert_pos = j
                break
            end
        end

        indices[i] = idx

        # Shift elements rightward to make room for the new index
        for k in (i + 1):-1:(insert_pos + 1)
            ordered_indices[k] = ordered_indices[k - 1]
        end
        ordered_indices[insert_pos] = idx
    end

    # Process the final index without adding it to ordered_indices
    idx = rand(rng, 1:(max_length - n_chains))
    for i in 1:n_chains
        if ordered_indices[i] ≤ idx
            idx += 1
        else
            break
        end
    end
    indices[n_chains] = idx

    return nothing
end

#uses preallocated for anything less than 4 chains (which the most default methods ask for)
function fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int}
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}
    if n_chains > length(indices)
        @warn "Picking $n_chains chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains`." maxlog = 1
        resize!(indices, n_chains)
        resize!(ordered_indices, n_chains - 1)
    end
    _fast_sample_chains!(
        rng,
        x,
        max_length,
        n_chains,
        indices,
        ordered_indices
    )
    return nothing
end

function _fast_sample_chains!(
        rng::AbstractRNG,
        x::VV,
        max_length::Int,
        n_chains::Int,
        indices::Vector{Int},
        ordered_indices::Vector{Int}
    ) where {VV <: AbstractVector{<:AbstractVector{<:Real}}}

    indices[1] = rand(rng, 1:max_length)
    ordered_indices[1] = indices[1]

    for i in 2:(n_chains - 1)
        idx = rand(rng, 1:(max_length - i + 1))
        insert_pos = i

        for j in 1:(i - 1)
            if ordered_indices[j] ≤ idx
                idx += 1
            else
                insert_pos = j
                break
            end
        end

        indices[i] = idx

        for k in i:-1:(insert_pos + 1)
            ordered_indices[k] = ordered_indices[k - 1]
        end
        ordered_indices[insert_pos] = idx
    end

    idx = rand(rng, 1:(max_length - n_chains + 1))
    for i in 1:(n_chains - 1)
        if ordered_indices[i] ≤ idx
            idx += 1
        else
            break
        end
    end
    indices[n_chains] = idx

    return nothing
end
