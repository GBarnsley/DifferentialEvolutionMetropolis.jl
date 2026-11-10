# Mutable so the state holds a pointer to the buffers rather than storing their fields inline
mutable struct DEMMultiTryBuffers{
        T <: Real, V <: AbstractVector{T}, W <: SubArray{T, 1}, WV <: SubArray{V, 1},
    }
    const forward::Array{V, 2}
    const forward_ld::Array{T, 2}
    "crossover probability of each forward trial, for adaptive subspace warm-up"
    const forward_cr::Array{T, 2}
    "cumulative selection weights of the forward trials"
    const forward_weight::Array{T, 2}
    const forward_log_weight::Vector{T}
    const reference::Array{V, 2}
    const reference_ld::Array{T, 2}
    "selected forward trial, 0 if every forward trial has zero weight"
    const selected::Vector{Int}
    "per-chain columns of the arrays above"
    const forward_view::Vector{WV}
    const forward_ld_view::Vector{W}
    const forward_weight_view::Vector{W}
    const reference_view::Vector{WV}
    const reference_ld_view::Vector{W}
end

function DEMMultiTryBuffers(
        x::AbstractVector{V}, n_tries::Int
    ) where {T <: Real, V <: AbstractVector{T}}
    n_chains = length(x)
    forward = [similar(first(x)) for _ in 1:n_tries, _ in 1:n_chains]
    forward_ld = Array{T, 2}(undef, n_tries, n_chains)
    forward_weight = Array{T, 2}(undef, n_tries, n_chains)
    reference = [similar(first(x)) for _ in 1:(n_tries - 1), _ in 1:n_chains]
    reference_ld = Array{T, 2}(undef, n_tries - 1, n_chains)
    return DEMMultiTryBuffers(
        forward,
        forward_ld,
        Array{T, 2}(undef, n_tries, n_chains),
        forward_weight,
        Vector{T}(undef, n_chains),
        reference,
        reference_ld,
        zeros(Int, n_chains),
        collect(eachcol(forward)),
        collect(eachcol(forward_ld)),
        collect(eachcol(forward_weight)),
        collect(eachcol(reference)),
        collect(eachcol(reference_ld))
    )
end

multi_try_buffers(x, n_tries::Int) = n_tries > 1 ? DEMMultiTryBuffers(x, n_tries) : nothing

n_tries(buffers::DEMMultiTryBuffers) = size(buffers.forward, 1)
n_tries(::Nothing) = 1


function check_n_tries(sampler::AbstractDEMMetaSampler, n_tries::Int)
    if n_tries < 1
        error("n_tries should be ≥ 1")
    elseif n_tries > 1 && !sampler_supports_multitry(sampler)
        @warn "n_tries > 1 but the sampler has no update that supports multiple tries, using n_tries = 1."
        return 1
    end
    return n_tries
end

sampler_supports_multitry(sampler::AbstractDEMSampler{<:Any, AbstractDEMSamplerSingleTry}) = false
sampler_supports_multitry(sampler::AbstractDEMSampler{<:Any, AbstractDEMSamplerMultipleTry}) = true
sampler_supports_multitry(sampler::AbstractDEMCompositeSampler) = any(update -> sampler_supports_multitry(update), sampler.updates)

trial_cr(proposal) = NaN
trial_cr(proposal::NamedTuple{(:offset, :cr)}) = proposal.cr

# Trials with zero weight cannot be selected as the cumulative weight does not increase at them
function select_trial!(weights::AbstractVector{T}, rng::AbstractRNG, lds::AbstractVector{T}, t::Real, log_weight::Real) where {T <: Real}
    weights .= exp.(lds ./ t .- log_weight)
    cumsum!(weights, weights)
    return findfirst(>(rand(rng) * last(weights)), weights)
end

function foreach_chain(f, n_chains::Int, parallel::Bool)
    if parallel
        Threads.@threads for i in 1:n_chains
            f(i)
        end
    else
        for i in 1:n_chains
            f(i)
        end
    end
    return nothing
end

trial_logdensity(model, x, ld) = ld > -Inf ? checked_logdensity(model, x) : ld

# Trials with an offset of -Inf are rejected by the proposal and are not evaluated
function evaluate_trials!(lds::AbstractArray{T}, xs, chain_models, parallel::Bool) where {T <: Real}
    if parallel
        Threads.@threads for m in eachindex(xs)
            lds[m] = trial_logdensity(chain_models[m], xs[m], lds[m])
        end
    else
        lds .= trial_logdensity.(Ref(chain_models[1]), xs, lds)
    end
    return nothing
end

function propose_forward!(buffers::DEMMultiTryBuffers, state, sampler::AbstractDEMSampler{<:Any, AbstractDEMSamplerMultipleTry}, i::Int)
    forward_ld = buffers.forward_ld_view[i]
    for (j, xₜ) in enumerate(buffers.forward_view[i])
        proposal = proposal!(xₜ, state.x[i], state, sampler, i)
        forward_ld[j], = proposal
        buffers.forward_cr[j, i] = trial_cr(proposal)
    end
    return nothing
end

function propose_reference!(buffers::DEMMultiTryBuffers, state, sampler::AbstractDEMSampler{<:Any, AbstractDEMSamplerMultipleTry}, i::Int)
    t = get_temperature(state.temperature_ladder, i)
    forward_ld = buffers.forward_ld_view[i]
    log_weight = logsumexp(ld / t for ld in forward_ld)
    buffers.forward_log_weight[i] = log_weight
    if log_weight == -Inf
        buffers.selected[i] = 0
        fill!(buffers.reference_ld_view[i], -Inf)
        return nothing
    end
    j = select_trial!(buffers.forward_weight_view[i], state.rngs[i], forward_ld, t, log_weight)
    buffers.selected[i] = j
    xⱼ = buffers.forward[j, i]
    map!(xₜ -> first(proposal!(xₜ, xⱼ, state, sampler, i)), buffers.reference_ld_view[i], buffers.reference_view[i])
    return nothing
end

function accept_trial!(buffers::DEMMultiTryBuffers, state, i::Int)
    j = buffers.selected[i]
    if j > 0
        t = get_temperature(state.temperature_ladder, i)
        reference_log_weight = logaddexp(logsumexp(ld / t for ld in buffers.reference_ld_view[i]), state.ld[i] / t)
        if log(rand(state.rngs[i])) < buffers.forward_log_weight[i] - reference_log_weight
            copyto!(state.xₚ[i], buffers.forward[j, i])
            state.ldₚ[i] = buffers.forward_ld[j, i]
            return true
        end
    end
    copyto!(state.xₚ[i], state.x[i])
    state.ldₚ[i] = state.ld[i]
    return false
end

# Forward and reference trials are each evaluated in one loop over every (trial, chain)
function update_chains!(
        state::DEMState{<:Any, <:Any, <:Any, <:Any, <:Any, <:DEMMultiTryBuffers},
        sampler::AbstractDEMSampler{<:Any, AbstractDEMSamplerMultipleTry}, parallel::Bool
    )
    buffers = state.multi_try
    n_chains = length(state.x)
    foreach_chain(i -> propose_forward!(buffers, state, sampler, i), n_chains, parallel)
    evaluate_trials!(buffers.forward_ld, buffers.forward, state.chain_models, parallel)
    foreach_chain(i -> propose_reference!(buffers, state, sampler, i), n_chains, parallel)
    evaluate_trials!(buffers.reference_ld, buffers.reference, state.chain_models, parallel)
    foreach_chain(i -> accept_trial!(buffers, state, i), n_chains, parallel)
    return nothing
end

# Memoryless chains are updated in turn, so only each chain's trials are evaluated in parallel
function update_chains!(
        state::DEMState{<:Any, <:Any, <:Any, <:DEMMemoryless, <:Any, <:DEMMultiTryBuffers},
        sampler::AbstractDEMSampler{AbstractDEMSamplerSerial, AbstractDEMSamplerMultipleTry}, parallel::Bool
    )
    buffers = state.multi_try
    for i in eachindex(state.x)
        propose_forward!(buffers, state, sampler, i)
        evaluate_trials!(buffers.forward_ld_view[i], buffers.forward_view[i], state.chain_models, parallel)
        propose_reference!(buffers, state, sampler, i)
        evaluate_trials!(buffers.reference_ld_view[i], buffers.reference_view[i], state.chain_models, parallel)
        accept_trial!(buffers, state, i)
    end
    return nothing
end
