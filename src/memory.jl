abstract type AbstractDEMMemoryless{T} <: AbstractDEMMemory{T} end

update_memory!(memory::AbstractDEMMemoryless, x) = nothing

#no memory
struct DEMMemoryless{T} <: AbstractDEMMemoryless{T}
    chain_ensembles::Tuple{UnitRange{Int}}
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

#no memory, odd and even chains form two ensembles that each propose from the other
struct DEMDoubleEnsemble{T} <: AbstractDEMMemoryless{T}
    chain_ensembles::NTuple{2, StepRange{Int, Int}}
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

# Ensembles of chains that are updated in turn, chains within an ensemble can be updated in parallel
chain_ensembles(memory::AbstractDEMMemory) = memory.chain_ensembles

n_ensembles(::AbstractDEMMemory) = 1
n_ensembles(memory::DEMDoubleEnsemble) = 2

#abstract type for methods of filling memory
abstract type AbstractDEMMemoryFillMethod end

#history archive, filled up to its size then grown by `memory_size` or, when `refill`, overwritten from the start
struct DEMMemoryFill{T, VV <: AbstractVector{<:AbstractVector{T}}, F <: AbstractDEMMemoryFillMethod} <:
    AbstractDEMMemory{T}
    chain_ensembles::Tuple{UnitRange{Int}}
    mem_x::VV
    fill::F
    refill::Bool
    memory_size::Int
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!(
        memory::DEMMemoryFill{T, VV},
        x::VV
    ) where {T <: Real, V_elem <: AbstractVector{T}, VV <: AbstractVector{V_elem}}
    if update_position!(memory.fill)
        for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
        memory.fill.stored = max(memory.fill.stored, memory.fill.position)
    end

    if memory.fill.position == length(memory.mem_x)
        if memory.refill
            memory.fill.position = 0
        else
            #increase memory size
            resize!(memory.mem_x, length(memory.mem_x) + memory.memory_size)
            dims = size(memory.mem_x[1])
            for i in (length(memory.mem_x) - memory.memory_size + 1):length(memory.mem_x)
                memory.mem_x[i] = V_elem(undef, dims)
            end
        end
    end
    return nothing
end

mutable struct DEMMemoryFillEvery <:
    AbstractDEMMemoryFillMethod
    position::Int
    "number of stored positions, the whole archive once it has been refilled"
    stored::Int
    const n_chains::Int
    DEMMemoryFillEvery(position::Int, n_chains::Int) = new(position, position, n_chains)
end

function update_position!(method::DEMMemoryFillEvery)
    method.position += method.n_chains
    return true
end

mutable struct DEMMemoryFillThin <:
    AbstractDEMMemoryFillMethod
    position::Int
    "number of stored positions, the whole archive once it has been refilled"
    stored::Int
    const n_chains::Int
    count::Int
    const max_count::Int
    DEMMemoryFillThin(position::Int, n_chains::Int, interval::Int) = new(position, position, n_chains, interval, interval)
end

function update_position!(method::DEMMemoryFillThin)
    method.count -= 1
    if method.count == 0
        method.position += method.n_chains
        method.count = method.max_count
        return true
    else
        return false
    end
end

function target_memory_size(sampler::AbstractDEMMetaSampler, n_chains::Int; p::Real = 0.01)
    k = memory_picks(sampler)
    return ceil(Int, n_chains / (1 - (1 - p)^(1 / k)) - n_chains)
end

function check_memory_size!(
        log::Vector{String}, sampler::AbstractDEMMetaSampler, n_chains::Int, N₀::Int,
        num_warmup::Int, thin_interval::Int, refill::Bool, total_memory_size::Int
    )
    A = target_memory_size(sampler, n_chains)
    refill_too_small = refill && total_memory_size < A
    if refill_too_small
        @warn "With `memory_refill = true` the memory stores at most $total_memory_size positions, fewer than the suggested minimum of $A before sampling. Consider `memory_size` of at least $(cld(A, n_chains))."
    end
    if num_warmup == 0
        push!(log, "   Suggested minimum memory size before sampling is $A positions (N₀ = $N₀).")
    elseif !refill_too_small
        thin = max(thin_interval, 1)
        B = N₀ + n_chains * div(num_warmup, thin)
        if B < A
            D = thin * cld(A - N₀, n_chains)
            @warn "After num_warmup = $num_warmup warm-up steps the memory will store $B positions, fewer than the suggested minimum of $A before sampling. Consider using at least $D warm-up steps and ensure convergence."
        end
    end
    return log
end
