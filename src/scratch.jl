# Per-chain proposal buffers held by an update, only reallocated when the element type, number of chains or dimensions changes
mutable struct ProposalScratch{N}
    "`Vector{NTuple{N, Vector{T}}}` for the element type `T` of the positions"
    buffers::Any
    "(n_chains, n_dims) the buffers were last sized for"
    size::Tuple{Int, Int}
end

ProposalScratch{N}() where {N} = ProposalScratch{N}(NTuple{N, Vector{Float64}}[], (0, 0))

function ensure_size!(scratch::ProposalScratch{N}, ::Type{T}, n_chains::Int, n_dims::Int) where {N, T}
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

# Buffers for chain `i`, `T` must be the element type the scratch was last prepared for
function chain_buffers(scratch::ProposalScratch{N}, ::Type{T}, i::Int) where {N, T}
    return (scratch.buffers::Vector{NTuple{N, Vector{T}}})[i]
end

# Must be called before chains are updated, as proposals may run in parallel and cannot resize safely
prepare_scratch!(sampler::AbstractDifferentialEvolutionSampler, state) = nothing
function prepare_scratch!(scratch::ProposalScratch, state)
    x = first(state.x)
    return ensure_size!(scratch, eltype(x), length(state.x), length(x))
end

# Scratch buffers hold no sampler state, so updates compare and hash by their other fields
Base.:(==)(::ProposalScratch, ::ProposalScratch) = true
Base.hash(::ProposalScratch, h::UInt) = h
function Base.:(==)(a::S, b::S) where {S <: AbstractDifferentialEvolutionSampler}
    return all(name -> getfield(a, name) == getfield(b, name), fieldnames(S))
end
function Base.hash(s::S, h::UInt) where {S <: AbstractDifferentialEvolutionSampler}
    return foldl((h, name) -> hash(getfield(s, name), h), fieldnames(S); init = hash(S, h))
end
