struct DifferentialEvolutionSampler <: AbstractDifferentialEvolutionSampler
    γ_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    β_spl::Sampleable{Univariate, Continuous}
end

"""
Set up a Differential Evolution (DE) update step for MCMC sampling.

Creates a sampler that proposes new states by adding scaled difference vectors between
randomly selected chains plus small noise. This is the core update mechanism from the
original DE-MC algorithm by ter Braak (2006).

See doi.org/10.1007/s11222-006-8769-1 for more information.

# Keyword Arguments
- `γ`: Scaling factor for the difference vector. Can be a `Real` (fixed value), a
  `UnivariateDistribution` (random scaling), or `nothing` (automatic based on `n_dims`).
  Defaults to `nothing`.
- `β`: Distribution for small noise added to proposals. Must be a univariate continuous
  distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `n_dims`: Problem dimension used for automatic `γ` selection. If > 0 and `γ` is `nothing`,
  sets `γ` to the theoretically optimal `2.38 / sqrt(2 * n_dims)`. If ≤ 0, uses
  `Uniform(0.8, 1.2)`. Defaults to 0.
- `check_args`: Whether to validate input distributions. Defaults to `true`.

# Returns
- A `DifferentialEvolutionSampler` that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref) or [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).

# Example
```@example de_update
using DifferentialEvolutionMetropolis, Distributions

# Setup differential evolution update with custom parameters
de_update = setup_de_update(γ = 1.0, β = Normal(0.0, 0.01))
```

See also [`setup_snooker_update`](@ref), [`setup_subspace_sampling`](@ref), [`setup_sampler_scheme`](@ref).
"""
function setup_de_update(;
        γ::Union{Nothing, UnivariateDistribution, Real} = nothing,
        β::ContinuousUnivariateDistribution = Uniform(-1.0e-4, 1.0e-4),
        n_dims::Int = 0,
        check_args::Bool = true
    )
    if isnothing(γ)
        if n_dims > 0
            γ = Dirac(2.38 / sqrt(2 * n_dims))
        else
            γ = Uniform(0.8, 1.2)
        end
    elseif isa(γ, Real)
        γ = Dirac(γ)
    end

    if check_args
        if Distributions.minimum(γ) < 0
            error("Distribution of γ should be bounded above 0")
        elseif Distributions.maximum(γ) ≤ 0
            error("Distribution of γ should be able to return values above 0")
        end
        noise_checks(β, "β")
    end

    return DifferentialEvolutionSampler(sampler(γ), sampler(β))
end

function noise_checks(dist, name)
    return if !(Distributions.mode(dist) ≈ 0.0)
        error("Distribution of $name should be centred around 0")
    elseif Distributions.median(dist) != Distributions.mode(dist)
        error("Distribution of $name should be uni-model and symmetric")
    end
end

function proposal!(
        state::DifferentialEvolutionState,
        sampler::DifferentialEvolutionSampler, current_state::Int
    )
    # Propose a new position.
    x₁, x₂ = pick_chains(state, current_state, 2)
    if x₁ == x₂
        state.xₚ[current_state] .= x₁
        return (offset = -Inf)
    else
        state.xₚ[current_state] .= state.x[current_state] .+
            (
            rand(state.rngs[current_state], sampler.γ_spl) .*
                (x₁ - x₂)
        ) .+
            rand(
            state.rngs[current_state], sampler.β_spl, length(state.x[current_state])
        )
        return (offset = zero(eltype(x₁)))
    end
end
