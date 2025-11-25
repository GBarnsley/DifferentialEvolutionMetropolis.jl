struct DifferentialEvolutionSnookerSampler <: AbstractDifferentialEvolutionSampler
    γ_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
end

"""
Set up a Snooker update step for MCMC sampling.

Creates a sampler that proposes moves along the line connecting the current position
to a projection point, scaled by the difference between two other randomly selected
chains. This update can help with sampling from distributions with complex geometries
by making larger moves in effective directions.

See doi.org/10.1007/s11222-008-9104-9 for more information.

# Keyword Arguments
- `γ`: Scaling factor for the projection. Can be a `Real` (fixed value), a
  `UnivariateDistribution` (random scaling), or `nothing` (automatic based on `deterministic_γ`).
  Defaults to `nothing`.
- `deterministic_γ`: When `γ` is `nothing`, determines the automatic value. If `true`,
  uses the theoretically optimal `2.38 / sqrt(2)`. If `false`, uses `Uniform(0.8, 1.2)`.
  Defaults to `true`.
- `check_args`: Whether to validate input distributions. Defaults to `true`.

# Returns
- A `DifferentialEvolutionSnookerSampler` that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref) or [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).

# Example
```@example snooker_update
using DifferentialEvolutionMetropolis, Distributions

# Setup snooker update with custom gamma distribution
snooker_update = setup_snooker_update(γ = Uniform(0.1, 2.0))
```

See also [`setup_de_update`](@ref), [`setup_subspace_sampling`](@ref), [`setup_sampler_scheme`](@ref).
"""
function setup_snooker_update(;
        γ::Union{Nothing, UnivariateDistribution, Real} = nothing,
        deterministic_γ::Bool = true,
        check_args::Bool = true
    )
    if isnothing(γ)
        if deterministic_γ
            γ = Dirac(2.38 / sqrt(2))
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
    end

    return DifferentialEvolutionSnookerSampler(sampler(γ))
end

function proposal!(
        state::DifferentialEvolutionState,
        sampler::DifferentialEvolutionSnookerSampler, current_state::Int
    )
    # Propose a new position.
    x₁, x₂, xₐ = pick_chains(state, current_state, 3)

    if xₐ == state.x[current_state] || x₁ == x₂
        state.xₚ[current_state] .= x₁
        return (offset = -Inf)
    else
        e = normalize(xₐ .- state.x[current_state])
        state.xₚ[current_state] .= state.x[current_state] .+
            rand(state.rngs[current_state], sampler.γ_spl) .*
            dot(x₁ .- x₂, e) .* e

        return (
            offset = (length(state.x[current_state]) - 1) *
                (
                log(norm(xₐ .- state.xₚ[current_state])) -
                    log(norm(xₐ .- state.x[current_state]))
            )
        )
    end
end
