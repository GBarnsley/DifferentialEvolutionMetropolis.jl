# Per-chain proposal buffers held by an update, only reallocated when the number of chains or dimensions changes
struct ProposalScratch{N}
    buffers::Vector{NTuple{N, Vector{Float64}}}
    "(n_chains, n_dims) the buffers were last sized for"
    size::Base.RefValue{Tuple{Int, Int}}
end

ProposalScratch{N}() where {N} = ProposalScratch{N}(NTuple{N, Vector{Float64}}[], Ref((0, 0)))

function ensure_size!(scratch::ProposalScratch{N}, n_chains::Int, n_dims::Int) where {N}
    scratch.size[] == (n_chains, n_dims) && return nothing
    while length(scratch.buffers) < n_chains
        push!(scratch.buffers, ntuple(_ -> Vector{Float64}(undef, n_dims), Val(N)))
    end
    for chain_buffers in scratch.buffers, buffer in chain_buffers
        length(buffer) == n_dims || resize!(buffer, n_dims)
    end
    scratch.size[] = (n_chains, n_dims)
    return nothing
end

# Must be called before chains are updated, as proposals may run in parallel and cannot resize safely
prepare_scratch!(sampler::AbstractDifferentialEvolutionSampler, state) = nothing
function prepare_scratch!(scratch::ProposalScratch, state)
    return ensure_size!(scratch, length(state.x), length(first(state.x)))
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
