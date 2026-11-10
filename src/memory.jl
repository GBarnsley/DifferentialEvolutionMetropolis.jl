#no memory
struct DEMMemoryless{T} <: AbstractDEMMemory{T}
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

#abstract memory-full type
abstract type AbstractDEMMemoryFormat{
    T, VV <: AbstractVector{<:AbstractVector{T}},
} <:
AbstractDEMMemory{T} end

#abstract type for methods of filling memory
abstract type AbstractDEMMemoryFillMethod end

#full memory, refreshing from the start
struct DEMMemoryRefill{T, VV <: AbstractVector{<:AbstractVector{T}}, F <: AbstractDEMMemoryFillMethod} <:
    AbstractDEMMemoryFormat{T, VV}
    mem_x::VV
    fill::F
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DEMMemoryRefill{T, VV},
        x::VV
    ) where {T <: Real, VV <: AbstractVector{<:AbstractVector{T}}}
    if update_position!(memory.fill)
        for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
    end
    if memory.fill.position == length(memory.mem_x)
        memory.fill.position = 0
    end
    return memory
end

#non-full memory, filling up to a max size then extending or refilling
struct DEMMemoryFill{T, VV <: AbstractVector{<:AbstractVector{T}}, F <: AbstractDEMMemoryFillMethod} <:
    AbstractDEMMemoryFormat{T, VV}
    mem_x::VV
    fill::F
    refill::Bool
    memory_size::Int
    #for preallocation only for internal use
    indices_INTERNAL::Vector{Vector{Int}}
    ordered_indices_INTERNAL::Vector{Vector{Int}}
end

function update_memory!!(
        memory::DEMMemoryFill{T, VV},
        x::VV
    ) where {T <: Real, V_elem <: AbstractVector{T}, VV <: AbstractVector{V_elem}}
    if update_position!(memory.fill)
        for i in 1:memory.fill.n_chains
            memory.mem_x[memory.fill.position - i + 1] .= x[i]
        end
    end

    if memory.fill.position == length(memory.mem_x)
        if memory.refill
            memory = DEMMemoryRefill(memory.mem_x, memory.fill, memory.indices_INTERNAL, memory.ordered_indices_INTERNAL)
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
    return memory
end

mutable struct DEMMemoryFillEvery <:
    AbstractDEMMemoryFillMethod
    position::Int
    n_chains::Int
end

function update_position!(method::DEMMemoryFillEvery)
    method.position += method.n_chains
    return true
end

mutable struct DEMMemoryFillThin <:
    AbstractDEMMemoryFillMethod
    position::Int
    n_chains::Int
    count::Int
    max_count::Int
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
