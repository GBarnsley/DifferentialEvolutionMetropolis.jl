struct MultiTryBuffers{T <: Real}
    "forward trials, (n_tries, n_chains)"
    forward::Matrix{Vector{T}}
    "log densities of the forward trials, -Inf for trials rejected by the proposal"
    forward_ld::Matrix{T}
    "crossover probability of each forward trial, NaN if the update has none"
    forward_cr::Matrix{T}
    "tempered log sum of the forward weights for each chain"
    forward_lse::Vector{T}
    "reference trials drawn from the selected trial, (n_tries - 1, n_chains)"
    reference::Matrix{Vector{T}}
    "log densities of the reference trials"
    reference_ld::Matrix{T}
    "index of the selected forward trial for each chain, 0 if no trial could be selected"
    selected::Vector{Int}
end

function MultiTryBuffers{T}(n_tries::Int, n_chains::Int, n_dims::Int) where {T <: Real}
    return MultiTryBuffers{T}(
        [Vector{T}(undef, n_dims) for _ in 1:n_tries, _ in 1:n_chains],
        Matrix{T}(undef, n_tries, n_chains),
        Matrix{T}(undef, n_tries, n_chains),
        Vector{T}(undef, n_chains),
        [Vector{T}(undef, n_dims) for _ in 1:(n_tries - 1), _ in 1:n_chains],
        Matrix{T}(undef, n_tries - 1, n_chains),
        zeros(Int, n_chains)
    )
end

# Trial buffers and model copies held by a multi-try update, only rebuilt when the sizes, element type or model change
mutable struct MultiTryScratch
    "`MultiTryBuffers{T}` for the element type `T` of the positions"
    buffers::Any
    "`Vector` of model copies for threaded evaluation, one per forward trial"
    models::Any
    "model the copies were made from"
    model_source::Any
end

MultiTryScratch() = MultiTryScratch(nothing, nothing, nothing)

Base.:(==)(::MultiTryScratch, ::MultiTryScratch) = true
Base.hash(::MultiTryScratch, h::UInt) = h

struct DifferentialEvolutionMultiTrySampler{S <: AbstractDifferentialEvolutionSampler} <:
    AbstractDifferentialEvolutionSampler
    update::S
    n_tries::Int
    scratch::MultiTryScratch
end

"""
    supports_multi_try(update)

Whether `update` can be wrapped by [`setup_multi_try_update`](@ref).

Supported updates must have symmetric proposals (zero Hastings offset, or `-Inf` to reject the
proposal outright) and implement `proposal!(xₚ, x, state, update, i)`, which writes a proposal
from position `x` into `xₚ` for chain `i`.
"""
supports_multi_try(::AbstractDifferentialEvolutionSampler) = false
supports_multi_try(::DifferentialEvolutionSampler) = true
supports_multi_try(::AbstractDifferentialEvolutionSubspaceSampler) = true

"""
Wrap an update in a multiple-try Metropolis step.

Each chain draws `n_tries` candidate proposals from the wrapped update, selects one with
probability proportional to its (tempered) target density, then draws `n_tries - 1` reference
proposals from the selected candidate. Together with the current position these give the
multiple-try acceptance probability of Liu, Liang and Wong (2000), as used in MT-DREAM(ZS).

See doi.org/10.1029/2011WR010608 for more information.

Each step evaluates up to `n_chains * (2 * n_tries - 1)` log densities, in two batches. With
`parallel = true` each batch is evaluated in one threaded loop over every (chain, trial) pair,
using `n_chains * n_tries` copies of the model, so multiple-try is most useful for expensive
log densities with spare threads. `n_tries = 1` is identical to the wrapped update.

Only symmetric proposals are supported: the differential evolution and subspace updates (see
[`DifferentialEvolutionMetropolis.supports_multi_try`](@ref)). During warm-up, adaptive subspace
updates adapt the crossover probabilities using the crossover probability of the selected trial.

# Arguments
- `update`: The update to wrap, from [`setup_de_update`](@ref) or [`setup_subspace_sampling`](@ref).

# Keyword Arguments
- `n_tries`: Number of candidate proposals per chain each step. Defaults to 5.

# Returns
- A `DifferentialEvolutionMultiTrySampler` that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref) or [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).

# Example
```@example multi_try_update
using DifferentialEvolutionMetropolis

# Five DE proposals per chain each step
multi_try_de = setup_multi_try_update(setup_de_update(); n_tries = 5)

# Multiple-try subspace updates alongside snooker updates
sampler = setup_sampler_scheme(
    setup_multi_try_update(setup_subspace_sampling()), setup_snooker_update(); w = [0.9, 0.1]
)
```

See also [`setup_de_update`](@ref), [`setup_subspace_sampling`](@ref), [`setup_sampler_scheme`](@ref).
"""
function setup_multi_try_update(update::AbstractDifferentialEvolutionSampler; n_tries::Int = 5)
    if n_tries < 1
        error("n_tries should be ≥ 1")
    elseif !supports_multi_try(update)
        error("Multiple-try is only supported for symmetric updates (setup_de_update or setup_subspace_sampling), got $(typeof(update))")
    end
    return DifferentialEvolutionMultiTrySampler(update, n_tries, MultiTryScratch())
end

chains_required(s::DifferentialEvolutionMultiTrySampler) = chains_required(s.update)

function prepare_scratch!(sampler::DifferentialEvolutionMultiTrySampler, state)
    prepare_scratch!(sampler.update, state)
    x = first(state.x)
    T = eltype(x)
    n_chains = length(state.x)
    buffers = sampler.scratch.buffers
    if !(buffers isa MultiTryBuffers{T}) || size(buffers.forward) != (sampler.n_tries, n_chains) ||
            length(first(buffers.forward)) != length(x)
        sampler.scratch.buffers = MultiTryBuffers{T}(sampler.n_tries, n_chains, length(x))
    end
    return nothing
end

trial_buffers(scratch::MultiTryScratch, ::Type{T}) where {T} = scratch.buffers::MultiTryBuffers{T}

function trial_models!(scratch::MultiTryScratch, model::Model, n::Int) where {Model}
    if !(scratch.models isa Vector{Model}) || scratch.model_source !== model || length(scratch.models) != n
        scratch.models = [deepcopy(model) for _ in 1:n]
        scratch.model_source = model
    end
    return scratch.models::Vector{Model}
end

function fix_sampler(
        sampler::DifferentialEvolutionMultiTrySampler,
        adaptive_state::AbstractDifferentialEvolutionAdaptiveState
    )
    return DifferentialEvolutionMultiTrySampler(
        fix_sampler(sampler.update, adaptive_state), sampler.n_tries, sampler.scratch
    )
end

function initialize_adaptive_state(
        sampler::DifferentialEvolutionMultiTrySampler,
        model_wrapper::LogDensityModel, n_chains::Int, ::Type{T}
    ) where {T <: Real}
    return initialize_adaptive_state(sampler.update, model_wrapper, n_chains, T)
end

trial_offset(prop::Real) = prop
trial_offset(prop::NamedTuple) = prop.offset

trial_cr(prop) = NaN
trial_cr(prop::NamedTuple{(:offset, :cr)}) = prop.cr

# log Σ exp(ld / t) over `lds` and `init`
function tempered_logsumexp(lds, t, init)
    m = maximum(lds; init = init) / t
    isfinite(m) || return m
    return m + log(sum(ld -> exp(ld / t - m), lds; init = exp(init / t - m)))
end

function select_trial(rng::AbstractRNG, lds, t, lse)
    length(lds) == 1 && return 1
    u = rand(rng)
    c = zero(u)
    for j in eachindex(lds)
        c += exp(lds[j] / t - lse)
        u < c && return j
    end
    return findlast(>(-Inf), lds)
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

# Trials marked -Inf by their proposal are not evaluated
function evaluate_trials!(lds, xs, model, models, parallel::Bool)
    if parallel
        Threads.@threads for m in eachindex(xs)
            if isfinite(lds[m])
                lds[m] = logdensity(models[m], xs[m])
            end
        end
    else
        for m in eachindex(xs)
            if isfinite(lds[m])
                lds[m] = logdensity(model, xs[m])
            end
        end
    end
    return nothing
end

function propose_forward!(buffers::MultiTryBuffers, state, update, i::Int)
    x = state.x[i]
    for j in axes(buffers.forward, 1)
        prop = proposal!(buffers.forward[j, i], x, state, update, i)
        buffers.forward_ld[j, i] = trial_offset(prop)
        buffers.forward_cr[j, i] = trial_cr(prop)
    end
    return nothing
end

function select_and_propose_reference!(buffers::MultiTryBuffers{T}, state, update, i::Int) where {T}
    t = get_temperature(state.temperature_ladder, i)
    lds = view(buffers.forward_ld, :, i)
    lse = tempered_logsumexp(lds, t, T(-Inf))
    buffers.forward_lse[i] = lse
    if !(lse > -Inf)
        buffers.selected[i] = 0
        buffers.reference_ld[:, i] .= -Inf
        return nothing
    end
    j = select_trial(state.rngs[i], lds, t, lse)
    buffers.selected[i] = j
    y = buffers.forward[j, i]
    for r in axes(buffers.reference, 1)
        prop = proposal!(buffers.reference[r, i], y, state, update, i)
        buffers.reference_ld[r, i] = trial_offset(prop)
    end
    return nothing
end

function accept_trial!(buffers::MultiTryBuffers, state, i::Int)
    j = buffers.selected[i]
    if j > 0
        t = get_temperature(state.temperature_ladder, i)
        reference_lse = tempered_logsumexp(view(buffers.reference_ld, :, i), t, state.ld[i])
        if !(log(rand(state.rngs[i])) > buffers.forward_lse[i] - reference_lse)
            copyto!(state.xₚ[i], buffers.forward[j, i])
            state.ldₚ[i] = buffers.forward_ld[j, i]
            return true
        end
    end
    copyto!(state.xₚ[i], state.x[i])
    state.ldₚ[i] = state.ld[i]
    return false
end

# Writes the multiple-try result for every chain into `state.xₚ` and `state.ldₚ`, the scratch must be prepared first
function multi_try_update!(model, sampler::DifferentialEvolutionMultiTrySampler, state, parallel::Bool)
    buffers = trial_buffers(sampler.scratch, eltype(first(state.x)))
    models = parallel ? trial_models!(sampler.scratch, model, length(buffers.forward)) : nothing
    n_chains = length(state.x)
    update = sampler.update

    foreach_chain(i -> propose_forward!(buffers, state, update, i), n_chains, parallel)
    evaluate_trials!(buffers.forward_ld, buffers.forward, model, models, parallel)
    foreach_chain(i -> select_and_propose_reference!(buffers, state, update, i), n_chains, parallel)
    evaluate_trials!(buffers.reference_ld, buffers.reference, model, models, parallel)
    foreach_chain(i -> accept_trial!(buffers, state, i), n_chains, parallel)
    return buffers
end

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionMultiTrySampler,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        parallel::Bool = false,
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    for i in eachindex(state.rngs)
        reseed!(state.rngs[i], rng)
    end
    prepare_scratch!(sampler, state)
    multi_try_update!(model_wrapper.logdensity, sampler, state, parallel)

    return create_sample(state),
        update_state(
            state;
            swap_positions = Val(true),
            update_memory = update_memory
        )
end

function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::DifferentialEvolutionMultiTrySampler{<:AbstractDifferentialEvolutionSubspaceSampler},
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveSubspace{T},
        };
        update_memory::Bool = true,
        parallel::Bool = false,
        kwargs...
    ) where {T <: Real}
    for i in eachindex(state.rngs)
        reseed!(state.rngs[i], rng)
    end
    adaptive_state = state.adaptive_state
    variance_ready = calculate_current_variance!(adaptive_state, sampler.update.min_variance_count)
    cold_chains = parentindices(state.x_smpl_view)[1]

    fixed_sampler = fix_sampler(sampler, adaptive_state)
    prepare_scratch!(fixed_sampler, state)
    buffers = multi_try_update!(model_wrapper.logdensity, fixed_sampler, state, parallel)

    if variance_ready
        for i in cold_chains
            j = buffers.selected[i]
            j == 0 && continue
            cr_index = findfirst(==(buffers.forward_cr[j, i]), adaptive_state.cr_spl.support)
            adaptive_state.L[cr_index] += 1
            adaptive_state.Δ[cr_index] += sum(
                (state.x[i] .- state.xₚ[i]) .* (state.x[i] .- state.xₚ[i]) ./
                    adaptive_state.variance
            )
        end
    end

    calculate_running_variance!(adaptive_state, state.xₚ_smpl_view)
    if all(adaptive_state.L .> 0) && any(adaptive_state.Δ .> 0)
        adapted_cr_probabilities!(adaptive_state, sampler.update.cr_uniform_weight)
        set_weights!(adaptive_state.cr_spl, adaptive_state.p)
    end

    return create_sample(state),
        update_state(
            state;
            update_memory = update_memory,
            swap_positions = Val(true)
        )
end
