"""
    DEMProposalScratch{N}()

Per-chain proposal buffers held by an update: `N` vectors per chain, with the element type and length of the positions.

The buffers are sized by [`DifferentialEvolutionMetropolis.prepare_scratch!`](@ref) and only reallocated when the element type, number of chains or dimensions changes.
Access them with [`DifferentialEvolutionMetropolis.chain_buffers`](@ref).
"""
mutable struct DEMProposalScratch{N}
    "`Vector{NTuple{N, Vector{T}}}` for the element type `T` of the positions"
    buffers::Any
    "(n_chains, n_dims) the buffers were last sized for"
    size::Tuple{Int, Int}
end

DEMProposalScratch{N}() where {N} = DEMProposalScratch{N}(NTuple{N, Vector{Float64}}[], (0, 0))

function ensure_size!(scratch::DEMProposalScratch{N}, ::Type{T}, n_chains::Int, n_dims::Int) where {N, T}
    if !(scratch.buffers isa Vector{NTuple{N, Vector{T}}})
        scratch.buffers = NTuple{N, Vector{T}}[]
        scratch.size = (0, 0)
    end
    scratch.size == (n_chains, n_dims) && return nothing
    buffers = scratch.buffers::Vector{NTuple{N, Vector{T}}}
    while length(buffers) < n_chains
        push!(buffers, ntuple(_ -> Vector{T}(undef, n_dims), Val(N)))
    end
    for per_chain in buffers, buffer in per_chain
        length(buffer) == n_dims || resize!(buffer, n_dims)
    end
    scratch.size = (n_chains, n_dims)
    return nothing
end

"""
    chain_buffers(scratch::DEMProposalScratch{N}, ::Type{T}, i::Int)

Return the `N` buffers of chain `i` as a tuple of `Vector{T}`.

`T` must be the element type the scratch was last prepared for, i.e. `eltype(x)` inside [`DifferentialEvolutionMetropolis.proposal!`](@ref).
"""
function chain_buffers(scratch::DEMProposalScratch{N}, ::Type{T}, i::Int) where {N, T}
    return (scratch.buffers::Vector{NTuple{N, Vector{T}}})[i]
end

"""
    prepare_scratch!(sampler::AbstractDEMMetaSampler, state)
    prepare_scratch!(scratch::DEMProposalScratch, state)

Size the scratch buffers of `sampler` for `state`, called before each step's chain updates since proposals may run in parallel and cannot resize safely.

The fallback does nothing.
An update holding a [`DifferentialEvolutionMetropolis.DEMProposalScratch`](@ref) should define a method that calls the `DEMProposalScratch` method on it.
"""
prepare_scratch!(sampler::AbstractDEMMetaSampler, state) = nothing

function prepare_scratch!(scratch::DEMProposalScratch, state)
    x = first(state.x)
    return ensure_size!(scratch, eltype(x), length(state.x), length(x))
end

# Scratch buffers hold no sampler state, so updates compare and hash by their other fields
Base.:(==)(::DEMProposalScratch, ::DEMProposalScratch) = true
Base.hash(::DEMProposalScratch, h::UInt) = h
function Base.:(==)(a::S, b::S) where {S <: AbstractDEMMetaSampler}
    return all(name -> getfield(a, name) == getfield(b, name), fieldnames(S))
end
function Base.hash(s::S, h::UInt) where {S <: AbstractDEMMetaSampler}
    return foldl((h, name) -> hash(getfield(s, name), h), fieldnames(S); init = hash(S, h))
end
