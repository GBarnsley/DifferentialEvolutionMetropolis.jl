# Mutable so the state holds a pointer to the buffers rather than storing their fields inline
mutable struct DEMMultiTryBuffers{T <: Real, V <: AbstractVector{T}, W <: SubArray{T, 1}, E <: Tuple}
    const forward::Array{V, 2}
    const forward_ld::Array{T, 2}
    "log Hastings correction of each forward trial"
    const forward_offset::Array{T, 2}
    "crossover probability of each forward trial, for adaptive subspace warm-up"
    const forward_cr::Array{T, 2}
    "cumulative selection weights of the forward trials"
    const forward_weight::Array{T, 2}
    const forward_log_weight::Vector{T}
    const reference::Array{V, 2}
    const reference_ld::Array{T, 2}
    const reference_offset::Array{T, 2}
    "selected forward trial, 0 if every forward trial has zero weight"
    const selected::Vector{Int}
    "`current_state` passed to `proposal!` for each trial, the first trial of a chain is passed the chain's index"
    const current_states::Array{Int, 2}
    "per-chain columns of the arrays above"
    const forward_ld_view::Vector{W}
    const forward_offset_view::Vector{W}
    const forward_weight_view::Vector{W}
    const reference_ld_view::Vector{W}
    const reference_offset_view::Vector{W}
    "(trial, chain) indices of the forward and reference trials of each ensemble"
    const ensemble_trials::E
end

function DEMMultiTryBuffers(x::AbstractVector{V}, n_tries::Int, ensembles::Tuple) where {T <: Real, V <: AbstractVector{T}}
    n_chains = length(x)
    forward_ld = Array{T, 2}(undef, n_tries, n_chains)
    forward_offset = Array{T, 2}(undef, n_tries, n_chains)
    forward_weight = Array{T, 2}(undef, n_tries, n_chains)
    reference_ld = Array{T, 2}(undef, n_tries - 1, n_chains)
    reference_offset = Array{T, 2}(undef, n_tries - 1, n_chains)
    ensemble_trials = map(ensembles) do chains
        (; forward = CartesianIndices((1:n_tries, chains)), reference = CartesianIndices((1:(n_tries - 1), chains)))
    end
    return DEMMultiTryBuffers(
        [similar(first(x)) for _ in 1:n_tries, _ in 1:n_chains],
        forward_ld,
        forward_offset,
        Array{T, 2}(undef, n_tries, n_chains),
        forward_weight,
        Vector{T}(undef, n_chains),
        [similar(first(x)) for _ in 1:(n_tries - 1), _ in 1:n_chains],
        reference_ld,
        reference_offset,
        zeros(Int, n_chains),
        [(j - 1) * n_chains + i for j in 1:n_tries, i in 1:n_chains],
        collect(eachcol(forward_ld)),
        collect(eachcol(forward_offset)),
        collect(eachcol(forward_weight)),
        collect(eachcol(reference_ld)),
        collect(eachcol(reference_offset)),
        ensemble_trials
    )
end

multi_try_buffers(x, n_tries::Int, ensembles) =
    n_tries > 1 ? DEMMultiTryBuffers(x, n_tries, ensembles) : nothing

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

sampler_supports_multitry(sampler::AbstractDEMSampler{<:Any, DEMSamplerSingleTry}) = false
sampler_supports_multitry(sampler::AbstractDEMSampler{<:Any, DEMSamplerMultipleTry}) = true
sampler_supports_multitry(sampler::AbstractDEMCompositeSampler) = any(update -> sampler_supports_multitry(update), sampler.updates)

trial_cr(proposal) = NaN
trial_cr(proposal::NamedTuple{(:offset, :cr)}) = proposal.cr

# Log weight of a trial y proposed from x with offset log(q(x | y) / q(y | x))
trial_log_weight(ld::Real, offset::Real, t::Real) = ld / t + offset / 2
trial_log_weight(lds::AbstractVector, offsets::AbstractVector, t::Real) =
    logsumexp(trial_log_weight(ld, offset, t) for (ld, offset) in zip(lds, offsets))

# Trials with zero weight cannot be selected as the cumulative weight does not increase at them
function select_trial!(
        weights::AbstractVector{T}, rng::AbstractRNG, lds::AbstractVector{T}, offsets::AbstractVector{T}, t::Real, log_weight::Real
    ) where {T <: Real}
    weights .= exp.(trial_log_weight.(lds, offsets, t) .- log_weight)
    cumsum!(weights, weights)
    return findfirst(>(rand(rng) * last(weights)), weights)
end

# Trials with an offset of -Inf are rejected by the proposal and are not evaluated
trial_logdensity(model, x, offset) = offset > -Inf ? checked_logdensity(model, x) : offset

# Each trial has its own rng, scratch and model copy, so every trial of an ensemble can be proposed and evaluated in parallel
function forward_trial!(buffers::DEMMultiTryBuffers, state, sampler::AbstractDEMSampler{<:Any, DEMSamplerMultipleTry}, j::Int, i::Int)
    current_state = buffers.current_states[j, i]
    xₜ = buffers.forward[j, i]
    proposal = proposal!(xₜ, state.x[i], state, sampler, current_state)
    buffers.forward_offset[j, i] = proposal.offset
    buffers.forward_cr[j, i] = trial_cr(proposal)
    buffers.forward_ld[j, i] = trial_logdensity(state.chain_models[current_state], xₜ, proposal.offset)
    return nothing
end

function select_trial!(buffers::DEMMultiTryBuffers, state, i::Int)
    t = get_temperature(state.temperature_ladder, i)
    forward_ld = buffers.forward_ld_view[i]
    forward_offset = buffers.forward_offset_view[i]
    log_weight = trial_log_weight(forward_ld, forward_offset, t)
    buffers.forward_log_weight[i] = log_weight
    buffers.selected[i] = log_weight == -Inf ? 0 :
        select_trial!(buffers.forward_weight_view[i], state.rngs[i], forward_ld, forward_offset, t, log_weight)
    return nothing
end

# No reference trials are proposed when every forward trial has zero weight (`selected[i] == 0`), as `accept_trial!` then keeps the current position
function reference_trial!(buffers::DEMMultiTryBuffers, state, sampler::AbstractDEMSampler{<:Any, DEMSamplerMultipleTry}, k::Int, i::Int)
    j = buffers.selected[i]
    j == 0 && return nothing
    current_state = buffers.current_states[k, i]
    xₜ = buffers.reference[k, i]
    offset = proposal!(xₜ, buffers.forward[j, i], state, sampler, current_state).offset
    buffers.reference_offset[k, i] = offset
    buffers.reference_ld[k, i] = trial_logdensity(state.chain_models[current_state], xₜ, offset)
    return nothing
end

# The current position is the final reference trial, its offset is the reverse of the selected trial's
function accept_trial!(buffers::DEMMultiTryBuffers, state, i::Int)
    j = buffers.selected[i]
    if j > 0
        t = get_temperature(state.temperature_ladder, i)
        reference_log_weight = logaddexp(
            trial_log_weight(buffers.reference_ld_view[i], buffers.reference_offset_view[i], t),
            trial_log_weight(state.ld[i], -buffers.forward_offset[j, i], t)
        )
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

# For each ensemble, the forward then the reference trials are each proposed and evaluated in one loop over every (trial, chain)
function update_chains!(
        state::DEMState{<:Any, <:Any, <:Any, <:Any, <:Any, <:DEMMultiTryBuffers},
        sampler::AbstractDEMSampler{<:Any, DEMSamplerMultipleTry}, parallel::Bool
    )
    buffers = state.multi_try
    for (chains, trials) in zip(chain_ensembles(state.memory), buffers.ensemble_trials)
        foreach_threaded(t -> forward_trial!(buffers, state, sampler, t[1], t[2]), trials.forward, parallel)
        foreach(i -> select_trial!(buffers, state, i), chains)
        foreach_threaded(t -> reference_trial!(buffers, state, sampler, t[1], t[2]), trials.reference, parallel)
        foreach(i -> accept_trial!(buffers, state, i), chains)
    end
    return nothing
end

# Memoryless chains are updated in turn, so only each chain's trials run in parallel
function update_chains!(
        state::DEMState{<:Any, <:Any, <:Any, <:DEMMemoryless, <:Any, <:DEMMultiTryBuffers},
        sampler::AbstractDEMSampler{DEMSamplerSerial, DEMSamplerMultipleTry}, parallel::Bool
    )
    buffers = state.multi_try
    for i in eachindex(state.x)
        foreach_threaded(j -> forward_trial!(buffers, state, sampler, j, i), axes(buffers.forward, 1), parallel)
        select_trial!(buffers, state, i)
        foreach_threaded(k -> reference_trial!(buffers, state, sampler, k, i), axes(buffers.reference, 1), parallel)
        accept_trial!(buffers, state, i)
    end
    return nothing
end
