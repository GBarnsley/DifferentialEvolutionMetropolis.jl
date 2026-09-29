mutable struct DifferentialEvolutionAdaptiveSubspace{T <: Real} <:
    AbstractDifferentialEvolutionAdaptiveState{T}
    "adapted crossover probability holder"
    p::Vector{T}
    "attempts for each crossover probability"
    L::Vector{Int}
    "squared normalised jumping distance for each crossover probability for each crossover probability"
    Δ::Vector{T}
    "sampler for crossover probabilities, reweighted in place"
    cr_spl::CrossoverSampler{T}
    "running count for variance calculation"
    var_count::Int
    "running mean for each dimension"
    var_mean::Vector{T}
    "running M2 for variance calculation (Welford's algorithm)"
    var_m2::Vector{T}
    "preallocated delta for variance calculation"
    delta::Vector{T}
    "preallocated variance vector"
    variance::Vector{T}
end

# Helper function to update running variance using Welford's algorithm
function calculate_running_variance!(
        adaptive_state::DifferentialEvolutionAdaptiveSubspace{T},
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
        adaptive_state::DifferentialEvolutionAdaptiveSubspace, min_variance_count::Int
    )
    ready = adaptive_state.var_count ≥ min_variance_count
    if ready
        adaptive_state.variance .= adaptive_state.var_m2 ./ (adaptive_state.var_count - 1)
    end
    return ready
end

# Probabilities proportional to mean normalised squared jump, mixed with uniform (weight `w`) so none reach zero
function adapted_cr_probabilities!(adaptive_state::DifferentialEvolutionAdaptiveSubspace, w::Real)
    adaptive_state.p .= adaptive_state.Δ ./ adaptive_state.L
    adaptive_state.p ./= sum(adaptive_state.p)
    n_cr = length(adaptive_state.p)
    adaptive_state.p .= (1 - w) .* adaptive_state.p .+ w / n_cr
    return nothing
end

#update the sampler with the adapted cr

function fix_sampler(
        sampler::DifferentialEvolutionSubspaceSampler,
        adaptive_state::DifferentialEvolutionAdaptiveSubspace
    )
    return DifferentialEvolutionSubspaceSampler(
        adaptive_state.cr_spl,
        sampler.n_cr,
        sampler.δ_spl,
        sampler.ϵ_spl,
        sampler.e_spl,
        sampler.cr_uniform_weight,
        sampler.min_variance_count,
        sampler.scratch
    )
end

function fix_sampler(
        sampler::DifferentialEvolutionSubspaceSamplerFixedGamma,
        adaptive_state::DifferentialEvolutionAdaptiveSubspace
    )
    return DifferentialEvolutionSubspaceSamplerFixedGamma(
        adaptive_state.cr_spl,
        sampler.n_cr,
        sampler.δ_spl,
        sampler.ϵ_spl,
        sampler.e_spl,
        sampler.cr_uniform_weight,
        sampler.min_variance_count,
        sampler.scratch,
        sampler.γ
    )
end

"""
    step_warmup(rng, model_wrapper, sampler, state; parallel=false, kwargs...)

Perform a single MCMC step during the warm-up (adaptive) phase.

During warm-up, this function performs the same sampling as [`step`](@ref) but also
updates adaptive parameters. For subspace samplers, it adapts crossover probabilities
based on the effectiveness of different parameter subsets. Only cold chains contribute
to the adaptation, and jumps are recorded once the running variance is available.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Adaptive differential evolution sampler
- `state`: Current state including adaptive parameters

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers).
  Defaults to `true`. Useful if memory has grown too large.
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to update functions

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state with adapted parameters for the next iteration

# Example
```@example step_warmup
using DifferentialEvolutionMetropolis, Random, Distributions

# Setup for warmup step example
rng = Random.default_rng()
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)
sampler = DREAMz()

# Initialize state (this would typically be done by AbstractMCMC.sample)
# sample, new_state = step_warmup(rng, model_wrapper, sampler, state; parallel=false)
```

See also [`step`](@ref), [`fix_sampler`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSubspaceSampler,
        state::DifferentialEvolutionState{
            T, <:DifferentialEvolutionAdaptiveSubspace{T},
        };
        update_memory::Bool = true,
        parallel::Bool = false,
        kwargs...
    ) where {T <: Real}
    # Derive per-chain RNGs deterministically from the provided rng for this step.
    for i in eachindex(state.rngs)
        reseed!(state.rngs[i], rng)
    end
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current state
    x = state.x
    adaptive_state = state.adaptive_state

    variance_ready = calculate_current_variance!(adaptive_state, sampler.min_variance_count)
    cold_chains = parentindices(state.x_smpl_view)[1]

    # loop through chains running the update
    fixed_sampler = fix_sampler(sampler, adaptive_state)
    prepare_scratch!(fixed_sampler, state)

    if parallel
        # thread safe updating
        Δ_update = zeros(T, length(x))
        cr_update = Vector{Int}(undef, length(x))

        Threads.@threads for i in eachindex(x)
            prop = proposal!(state, fixed_sampler, i)
            accepted = update_chain!(state.chain_models[i], state, prop.offset, i)
            cr_update[i] = findfirst(==(prop.cr), adaptive_state.cr_spl.support)
            if accepted
                Δ_update[i] += sum(
                    (state.x[i] .- state.xₚ[i]) .* (state.x[i] .- state.xₚ[i]) ./
                        adaptive_state.variance
                )
            end
        end
        if variance_ready
            for i in cold_chains
                adaptive_state.L[cr_update[i]] += 1
                adaptive_state.Δ[cr_update[i]] += Δ_update[i]
            end
        end
    else
        for i in eachindex(x)
            prop = proposal!(state, fixed_sampler, i)
            accepted = update_chain!(model, state, prop.offset, i)

            if variance_ready && i in cold_chains
                cr_update = findfirst(==(prop.cr), adaptive_state.cr_spl.support)
                adaptive_state.L[cr_update] += 1
                if accepted
                    adaptive_state.Δ[cr_update] += sum(
                        (state.x[i] .- state.xₚ[i]) .* (state.x[i] .- state.xₚ[i]) ./
                            adaptive_state.variance
                    )
                end
            end
        end
    end

    #update variance
    calculate_running_variance!(adaptive_state, state.xₚ_smpl_view)
    if all(adaptive_state.L .> 0) && any(adaptive_state.Δ .> 0)
        adapted_cr_probabilities!(adaptive_state, sampler.cr_uniform_weight)
        set_weights!(adaptive_state.cr_spl, adaptive_state.p)
    end

    return create_sample(state),
        update_state(
            state;
            update_memory = update_memory,
            swap_positions = Val(true)
        )
end

function initialize_adaptive_state(
        sampler::AbstractDifferentialEvolutionSubspaceSampler,
        model_wrapper::LogDensityModel, n_chains::Int
    )
    n_cr = sampler.n_cr
    T = Float64
    d = dimension(model_wrapper.logdensity)
    if n_cr == 0
        @warn "sampler already has a fixed crossover probability, cannot adapt."
        return DifferentialEvolutionAdaptiveStatic{T}()
    elseif n_cr == 1
        @warn "Only one crossover probability, cannot adapt."
        return DifferentialEvolutionAdaptiveStatic{T}()
    else
        p = zeros(T, n_cr)
        L = zeros(Int, n_cr)
        Δ = zeros(T, n_cr)
        cr_dist = sampler.cr_spl isa DiscreteNonParametric ? sampler.cr_spl : create_cr_dist(n_cr)
        if !all(Distributions.support(cr_dist) .≈ Distributions.support(create_cr_dist(n_cr)))
            @warn "Adapting provided crossover probabilities."
        end
        cr_spl = CrossoverSampler(T.(Distributions.support(cr_dist)), Distributions.probs(cr_dist))
        # Initialize running variance tracking
        var_count = 0
        var_mean = zeros(T, d)
        var_m2 = zeros(T, d)
        delta = zeros(T, d)
        variance = ones(T, d)
        return DifferentialEvolutionAdaptiveSubspace{T}(
            p, L, Δ, cr_spl, var_count, var_mean, var_m2, delta, variance
        )
    end
end
