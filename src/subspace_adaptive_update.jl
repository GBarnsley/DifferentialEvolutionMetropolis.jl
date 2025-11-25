mutable struct DifferentialEvolutionAdaptiveSubspace{T <: Real} <:
    AbstractDifferentialEvolutionAdaptiveState{T}
    "attempts for each crossover probability"
    L::Vector{Int}
    "squared normalised jumping distance for each crossover probability for each crossover probability"
    Δ::Vector{T}
    "distribution for crossover probabilities"
    cr_spl::DiscreteNonParametricSampler
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
        new_values::Vector{V}
    ) where {T <: Real, V <: Vector{T}}
    for new_value in new_values
        adaptive_state.var_count += 1
        adaptive_state.delta .= new_value .- adaptive_state.var_mean
        adaptive_state.var_mean .+= adaptive_state.delta ./ adaptive_state.var_count
        adaptive_state.var_m2 .+= adaptive_state.delta .*
            (new_value .- adaptive_state.var_mean)
    end
    return nothing
end

# Helper function to get current variance
function calculate_current_variance!(adaptive_state::DifferentialEvolutionAdaptiveSubspace)
    if adaptive_state.var_count ≥ 10
        adaptive_state.variance .= adaptive_state.var_m2 ./ adaptive_state.var_count
    end
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
        sampler.e_spl
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
        sampler.γ
    )
end

"""
    step_warmup(rng, model_wrapper, sampler, state; parallel=false, kwargs...)

Perform a single MCMC step during the warm-up (adaptive) phase.

During warm-up, this function performs the same sampling as [`step`](@ref) but also
updates adaptive parameters. For subspace samplers, it adapts crossover probabilities
based on the effectiveness of different parameter subsets.

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
            T, DifferentialEvolutionAdaptiveSubspace{T},
        };
        update_memory::Bool = true,
        parallel::Bool = false,
        kwargs...
    ) where {T <: Real}
    # Derive per-chain RNGs deterministically from the provided rng for this step.
    for i in eachindex(state.rngs)
        state.rngs[i] = Random.seed!(copy(rng), rand(rng, UInt))
    end
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current state
    x = state.x
    adaptive_state = state.adaptive_state

    calculate_current_variance!(adaptive_state)

    # loop through chains running the update
    fixed_sampler = fix_sampler(sampler, adaptive_state)

    if parallel
        # thread safe updating
        Δ_update = zeros(T, length(x))
        cr_update = Vector{Int}(undef, length(x))

        Threads.@threads for i in eachindex(x)
            prop = proposal!(state, fixed_sampler, i)
            accepted = update_chain!(model, state, prop.offset, i)
            cr_update[i] = findfirst(prop.cr .== adaptive_state.cr_spl.support)
            if accepted
                Δ_update[i] += sum(
                    (state.x[i] .- state.xₚ[i]) .* (state.x[i] .- state.xₚ[i]) ./
                        adaptive_state.variance
                )
            end
        end
        for i in eachindex(x)
            adaptive_state.L[cr_update[i]] += 1
            adaptive_state.Δ[cr_update[i]] += Δ_update[i]
        end
    else
        for i in eachindex(x)
            prop = proposal!(state, fixed_sampler, i)
            accepted = update_chain!(model, state, prop.offset, i)

            cr_update = findfirst(prop.cr .== adaptive_state.cr_spl.support)
            adaptive_state.L[cr_update] += 1
            if accepted
                adaptive_state.Δ[cr_update] += sum(
                    (state.x[i] .- state.xₚ[i]) .* (state.x[i] .- state.xₚ[i]) ./
                        adaptive_state.variance
                )
            end
        end
    end

    #update variance
    calculate_running_variance!(adaptive_state, state.xₚ)
    if all(adaptive_state.L .> 0) && all(adaptive_state.Δ .> 0)
        adaptive_state.cr_spl = Distributions.sampler(
            DiscreteNonParametric(
                adaptive_state.cr_spl.support,
                sum_to_one!(
                    sum(adaptive_state.L) .* (adaptive_state.Δ ./ adaptive_state.L) ./
                        sum(adaptive_state.Δ)
                )
            )
        )
    end

    return create_sample(state),
        update_state(
            state;
            update_memory = update_memory,
            x = state.xₚ, ld = state.ldₚ, xₚ = state.x, ldₚ = state.ld
        )
end

function sum_to_one!(v::Vector{T}) where {T <: Real}
    v ./= sum(v)
    return v
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
        L = zeros(Int, n_cr)
        Δ = zeros(T, n_cr)
        if sampler.cr_spl isa DiscreteNonParametricSampler
            if any(.!(sampler.cr_spl.support .≈ create_cr_dist(n_cr).support))
                @warn "Adapting provided crossover probabilities."
            end
            cr_spl = sampler.cr_spl
        else
            cr_spl = Distributions.sampler(create_cr_dist(n_cr))
        end
        # Initialize running variance tracking
        var_count = 0
        var_mean = zeros(T, d)
        var_m2 = zeros(T, d)
        delta = zeros(T, d)
        variance = ones(T, d)
        return DifferentialEvolutionAdaptiveSubspace{T}(
            L, Δ, cr_spl, var_count, var_mean, var_m2, delta, variance
        )
    end
end
