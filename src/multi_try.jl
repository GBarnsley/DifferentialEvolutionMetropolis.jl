struct DifferentialEvolutionMetropolisMultiTryBuffers{T <: Real, V <: AbstractVector{T}}
    forward::Array{V, 2}
    forward_ld::Array{T, 2}
    "crossover probability of each forward trial, for adaptive subspace warm-up"
    forward_cr::Array{T, 2}
    forward_log_weight::Vector{T}
    reference::Array{V, 2}
    reference_ld::Array{T, 2}
    "selected forward trial, 0 if every forward trial has zero weight"
    selected::Vector{Int}
end

function DifferentialEvolutionMetropolisMultiTryBuffers(
        x::AbstractVector{V}, n_tries::Int
    ) where {T <: Real, V <: AbstractVector{T}}
    n_chains = length(x)
    return DifferentialEvolutionMetropolisMultiTryBuffers(
        [similar(first(x)) for _ in 1:n_tries, _ in 1:n_chains],
        Array{T, 2}(undef, n_tries, n_chains),
        Array{T, 2}(undef, n_tries, n_chains),
        Vector{T}(undef, n_chains),
        [similar(first(x)) for _ in 1:(n_tries - 1), _ in 1:n_chains],
        Array{T, 2}(undef, n_tries - 1, n_chains),
        zeros(Int, n_chains)
    )
end

n_tries(buffers::DifferentialEvolutionMetropolisMultiTryBuffers) = size(buffers.forward, 1)

# Multiple-try weights are the target density, which is only valid for symmetric proposals
supports_multi_try(::AbstractDifferentialEvolutionSampler) = false
supports_multi_try(::DifferentialEvolutionSampler) = true
supports_multi_try(::AbstractDifferentialEvolutionSubspaceSampler) = true
supports_multi_try(s::DifferentialEvolutionCompositeSampler) = any(supports_multi_try, s.updates)

uses_multi_try(sampler, state) = supports_multi_try(sampler) && n_tries(state.multi_try) > 1

function check_n_tries(sampler::AbstractDifferentialEvolutionSampler, n_tries::Int)
    if n_tries < 1
        error("n_tries should be ≥ 1")
    elseif n_tries > 1 && !supports_multi_try(sampler)
        @warn "n_tries > 1 but the sampler has no update that supports multiple tries (setup_de_update or setup_subspace_sampling), using n_tries = 1."
        return 1
    end
    return n_tries
end

trial_cr(proposal) = NaN
trial_cr(proposal::NamedTuple{(:offset, :cr)}) = proposal.cr

# log Σⱼ exp(lds[j, i] / t)
function tempered_log_sum(lds::Array{T, 2}, i::Int, t::Real) where {T <: Real}
    m = lds[1, i]
    for j in 2:size(lds, 1)
        m = max(m, lds[j, i])
    end
    isfinite(m) || return m
    s = zero(T)
    for j in axes(lds, 1)
        s += exp((lds[j, i] - m) / t)
    end
    return m / t + log(s)
end

function log_add_exp(a::Real, b::Real)
    m = max(a, b)
    isfinite(m) || return m
    return m + log(exp(a - m) + exp(b - m))
end

function select_trial(rng::AbstractRNG, lds::Array{T, 2}, i::Int, t::Real, log_weight::Real) where {T <: Real}
    u = rand(rng)
    for j in axes(lds, 1)
        u -= exp(lds[j, i] / t - log_weight)
        u < 0 && return j
    end
    j = size(lds, 1)
    while lds[j, i] == -Inf
        j -= 1
    end
    return j
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

# Trials with an offset of -Inf are rejected by the proposal and are not evaluated
function evaluate_trials!(lds::Array{T, 2}, xs, model, chain_models, parallel::Bool) where {T <: Real}
    if parallel
        Threads.@threads for m in eachindex(xs)
            if lds[m] > -Inf
                lds[m] = logdensity(chain_models[m], xs[m])
            end
        end
    else
        for m in eachindex(xs)
            if lds[m] > -Inf
                lds[m] = logdensity(model, xs[m])
            end
        end
    end
    return nothing
end

function propose_forward!(buffers::DifferentialEvolutionMetropolisMultiTryBuffers, state, sampler, i::Int)
    for j in axes(buffers.forward, 1)
        proposal = proposal!(buffers.forward[j, i], state.x[i], state, sampler, i)
        offset, = proposal
        buffers.forward_ld[j, i] = offset
        buffers.forward_cr[j, i] = trial_cr(proposal)
    end
    return nothing
end

function propose_reference!(buffers::DifferentialEvolutionMetropolisMultiTryBuffers, state, sampler, i::Int)
    t = get_temperature(state.temperature_ladder, i)
    log_weight = tempered_log_sum(buffers.forward_ld, i, t)
    buffers.forward_log_weight[i] = log_weight
    if log_weight == -Inf
        buffers.selected[i] = 0
        for r in axes(buffers.reference, 1)
            buffers.reference_ld[r, i] = -Inf
        end
        return nothing
    end
    j = select_trial(state.rngs[i], buffers.forward_ld, i, t, log_weight)
    buffers.selected[i] = j
    for r in axes(buffers.reference, 1)
        offset, = proposal!(buffers.reference[r, i], buffers.forward[j, i], state, sampler, i)
        buffers.reference_ld[r, i] = offset
    end
    return nothing
end

function accept_trial!(buffers::DifferentialEvolutionMetropolisMultiTryBuffers, state, i::Int)
    j = buffers.selected[i]
    if j > 0
        t = get_temperature(state.temperature_ladder, i)
        reference_log_weight = log_add_exp(tempered_log_sum(buffers.reference_ld, i, t), state.ld[i] / t)
        if !(log(rand(state.rngs[i])) > buffers.forward_log_weight[i] - reference_log_weight)
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
function multi_try_update!(model, state, sampler, parallel::Bool)
    buffers = state.multi_try
    n_chains = length(state.x)
    foreach_chain(i -> propose_forward!(buffers, state, sampler, i), n_chains, parallel)
    evaluate_trials!(buffers.forward_ld, buffers.forward, model, state.chain_models, parallel)
    foreach_chain(i -> propose_reference!(buffers, state, sampler, i), n_chains, parallel)
    evaluate_trials!(buffers.reference_ld, buffers.reference, model, state.chain_models, parallel)
    foreach_chain(i -> accept_trial!(buffers, state, i), n_chains, parallel)
    return nothing
end
