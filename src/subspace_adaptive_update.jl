mutable struct DEMAdaptiveSubspace{T <: Real} <:
    AbstractDEMAdaptiveState{T}
    "adapted crossover probability holder"
    const p::Vector{T}
    "attempts for each crossover probability"
    const L::Vector{Int}
    "squared normalised jumping distance for each crossover probability for each crossover probability"
    const Δ::Vector{T}
    "sampler for crossover probabilities, reweighted in place"
    const cr_spl::DEMCrossoverSampler{T}
    "running count for variance calculation"
    var_count::Int
    "running mean for each dimension"
    const var_mean::Vector{T}
    "running M2 for variance calculation (Welford's algorithm)"
    const var_m2::Vector{T}
    "preallocated delta for variance calculation"
    const delta::Vector{T}
    "preallocated variance vector"
    const variance::Vector{T}
    "crossover probability index of the last proposal for each `current_state`"
    const chain_cr::Vector{Int}
end

# Squared jump from `x` to `xₚ` normalised by the running variance, without allocating
function normalised_squared_jump(x, xₚ, variance)
    return sum(abs2(xᵢ - xₚᵢ) / vᵢ for (xᵢ, xₚᵢ, vᵢ) in zip(x, xₚ, variance))
end

# Helper function to update running variance using Welford's algorithm
function calculate_running_variance!(
        adaptive_state::DEMAdaptiveSubspace{T},
        new_values::VV
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    for new_value in new_values
        adaptive_state.var_count += 1
        adaptive_state.delta .= new_value .- adaptive_state.var_mean
        adaptive_state.var_mean .+= adaptive_state.delta ./ adaptive_state.var_count
        adaptive_state.var_m2 .+= adaptive_state.delta .*
            (new_value .- adaptive_state.var_mean)
    end
    return nothing
end

# Helper function to get current sample variance, returns whether it is available yet
function calculate_current_variance!(
        adaptive_state::DEMAdaptiveSubspace, min_variance_count::Int
    )
    ready = adaptive_state.var_count ≥ min_variance_count
    if ready
        adaptive_state.variance .= adaptive_state.var_m2 ./ (adaptive_state.var_count - 1)
    end
    return ready
end

# Probabilities proportional to mean normalised squared jump, mixed with uniform (weight `w`) so none reach zero
function adapted_cr_probabilities!(adaptive_state::DEMAdaptiveSubspace, w::Real)
    adaptive_state.p .= adaptive_state.Δ ./ adaptive_state.L
    adaptive_state.p ./= sum(adaptive_state.p)
    n_cr = length(adaptive_state.p)
    adaptive_state.p .= (1 - w) .* adaptive_state.p .+ w / n_cr
    return nothing
end

function proposal!(
        xₚ::V, x::V, state::DEMState,
        sampler::AbstractDEMSubspaceSampler, adaptive_state::DEMAdaptiveSubspace, current_state::Int
    ) where {V <: AbstractVector}
    return subspace_proposal!(xₚ, x, state, sampler, adaptive_state.cr_spl, current_state)
end

# The crossover probability index of each proposal, indexed by `current_state`
function record_proposal!(adaptive_state::DEMAdaptiveSubspace, proposal::NamedTuple, current_state::Int)
    adaptive_state.chain_cr[current_state] = findfirst(==(proposal.cr), adaptive_state.cr_spl.support)
    return nothing
end

# Records each cold chain's normalised squared jump against the crossover probability it used, then reweights the crossover probabilities
function adapt!(adaptive_state::DEMAdaptiveSubspace, sampler::AbstractDEMSubspaceSampler, state::DEMState)
    if calculate_current_variance!(adaptive_state, sampler.min_variance_count)
        for i in parentindices(state.x_smpl_view)[1]
            current_state = selected_state(state.multi_try, i)
            current_state == 0 && continue
            cr_index = adaptive_state.chain_cr[current_state]
            adaptive_state.L[cr_index] += 1
            adaptive_state.Δ[cr_index] += normalised_squared_jump(state.x[i], state.xₚ[i], adaptive_state.variance)
        end
    end
    calculate_running_variance!(adaptive_state, state.x_smpl_view)
    if all(>(0), adaptive_state.L) && any(>(0), adaptive_state.Δ)
        adapted_cr_probabilities!(adaptive_state, sampler.cr_uniform_weight)
        set_weights!(adaptive_state.cr_spl, adaptive_state.p)
    end
    return nothing
end

function initialize_adaptive_state(
        sampler::AbstractDEMSubspaceSampler,
        model_wrapper::LogDensityModel, n_chains::Int, n_tries::Int, ::Type{T}
    ) where {T <: Real}
    n_cr = sampler.n_cr
    d = dimension(model_wrapper.logdensity)
    if n_cr == 0
        @warn "sampler already has a fixed crossover probability, cannot adapt."
        return DEMAdaptiveStatic{T}()
    elseif n_cr == 1
        @warn "Only one crossover probability, cannot adapt."
        return DEMAdaptiveStatic{T}()
    else
        p = zeros(T, n_cr)
        L = zeros(Int, n_cr)
        Δ = zeros(T, n_cr)
        cr_dist = sampler.cr_spl isa DiscreteNonParametric ? sampler.cr_spl : create_cr_dist(n_cr)
        if !all(Distributions.support(cr_dist) .≈ Distributions.support(create_cr_dist(n_cr)))
            @warn "Adapting provided crossover probabilities."
        end
        cr_spl = DEMCrossoverSampler(T.(Distributions.support(cr_dist)), Distributions.probs(cr_dist))
        # Initialize running variance tracking
        var_count = 0
        var_mean = zeros(T, d)
        var_m2 = zeros(T, d)
        delta = zeros(T, d)
        variance = ones(T, d)
        return DEMAdaptiveSubspace{T}(
            p, L, Δ, cr_spl, var_count, var_mean, var_m2, delta, variance, zeros(Int, n_chains * n_tries)
        )
    end
end
