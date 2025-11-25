abstract type AbstractDifferentialEvolutionSubspaceSampler <:
AbstractDifferentialEvolutionSampler end

struct DifferentialEvolutionSubspaceSampler <: AbstractDifferentialEvolutionSubspaceSampler
    cr_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    n_cr::Int
    δ_spl::Sampleable{Univariate, Discrete}
    ϵ_spl::Sampleable{Univariate, Continuous}
    e_spl::Sampleable{Univariate, Continuous}
end

struct DifferentialEvolutionSubspaceSamplerFixedGamma{T <: Real} <:
    AbstractDifferentialEvolutionSubspaceSampler
    cr_spl::Sampleable{Univariate, <:Union{Continuous, Discrete}}
    n_cr::Int
    δ_spl::Sampleable{Univariate, Discrete}
    ϵ_spl::Sampleable{Univariate, Continuous}
    e_spl::Sampleable{Univariate, Continuous}
    γ::T
end

"""
Set up a Subspace Sampling (DREAM-like) update step for MCMC sampling.

Creates a sampler that updates only a random subset of parameters in each iteration,
using multiple scaled difference vectors. The crossover probability determines which
parameters to update and can be adapted during warm-up for improved efficiency.

See doi.org/10.1515/IJNSNS.2009.10.3.273 for more information.

# Keyword Arguments
- `γ`: Scaling factor for the difference vector sum. If `nothing` (default), uses the
  adaptive formula `2.38 / sqrt(2 * δ * d)` where `d` is the number of updated dimensions.
  If a `Real` is provided, uses that fixed value throughout sampling.
- `cr`: Crossover probability for parameter selection. Can be a `Real` (fixed probability),
  `nothing` (adaptive using `n_cr` values), or a `UnivariateDistribution`. If cr is a `DiscreteNonParametric` then it those values can also be adapted. Defaults to `nothing`.
- `n_cr`: Number of crossover probabilities to adapt between when `cr` is `nothing`.
  Higher values allow more fine-tuned adaptation. Defaults to 3.
- `δ`: Number of difference vectors to sum. Can be an `Integer` (fixed) or a
  `DiscreteUnivariateDistribution` (random). Defaults to `DiscreteUniform(1, 3)`.
- `ϵ`: Distribution for small additive noise in the selected subspace. Defaults to
  `Uniform(-1e-4, 1e-4)`.
- `e`: Distribution for multiplicative noise `(1 + e)` applied to the difference vector sum.
  Defaults to `Normal(0.0, 1e-2)`.
- `check_args`: Whether to validate input distributions. Defaults to `true`.

# Returns
- A subspace sampler that can be used with [`setup_sampler_scheme`](@ref) or [`step`](@ref) or [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).

# Example
```@example subspace_sampling
using DifferentialEvolutionMetropolis, Distributions

# Setup subspace sampling with custom crossover rate and delta
subspace_config = setup_subspace_sampling(cr = Beta(1, 2), δ = 2)
```

See also [`setup_de_update`](@ref), [`setup_snooker_update`](@ref), [`setup_sampler_scheme`](@ref).
"""
function setup_subspace_sampling(;
        γ::Union{Nothing, Real} = nothing,
        cr::Union{Real, UnivariateDistribution, Nothing} = nothing,
        n_cr::Int = 3,
        δ::Union{Integer, DiscreteUnivariateDistribution} = DiscreteUniform(
            1, 3
        ),
        ϵ::ContinuousUnivariateDistribution = Uniform(-1.0e-4, 1.0e-4),
        e::ContinuousUnivariateDistribution = Normal(0.0, 1.0e-2),
        check_args::Bool = true
    )
    if isa(δ, Integer)
        δ = Dirac(δ)
    end

    if isa(cr, Real)
        cr = Dirac(cr)
        n_cr = 0
    elseif isnothing(cr)
        cr = create_cr_dist(n_cr)
    elseif isa(cr, DiscreteNonParametric)
        n_cr = length(Distributions.support(cr))
    else
        n_cr = 0
    end

    if check_args
        if Distributions.minimum(cr) ≤ 0
            error("Distribution of crossover probabilities (cr) should be bounded above 0")
        elseif Distributions.maximum(cr) > 1
            error("Distribution of crossover probabilities (cr) should be ≤ 1")
        elseif Distributions.minimum(δ) ≤ 0
            error("Distribution of δ should be bounded above 0")
        end
        noise_checks(ϵ, "ϵ")
        noise_checks(e, "e")
    end

    return if isnothing(γ)
        DifferentialEvolutionSubspaceSampler(
            sampler(cr),
            n_cr,
            sampler(δ),
            sampler(ϵ),
            sampler(e)
        )
    else
        if check_args
            if γ ≤ 0
                error("γ should be ≥ 0")
            end
        end
        DifferentialEvolutionSubspaceSamplerFixedGamma{eltype(γ)}(
            sampler(cr),
            n_cr,
            sampler(δ),
            sampler(ϵ),
            sampler(e),
            γ
        )
    end
end

function create_cr_dist(n_cr::Int)
    return DiscreteNonParametric(collect(1:n_cr) ./ n_cr, repeat([1 / n_cr], n_cr))
end

function proposal!(
        state::DifferentialEvolutionState,
        sampler::AbstractDifferentialEvolutionSubspaceSampler, current_state::Int
    )
    rng = state.rngs[current_state]
    x = state.x[current_state]
    xₚ = state.xₚ[current_state]

    copyto!(xₚ, x) #try the range methods?

    #determine how many dimensions to update
    cr = rand(rng, sampler.cr_spl)
    to_update = rand(rng, length(x)) .< cr
    d = sum(to_update)

    if d == 0
        #just pick one
        to_update[rand(rng, eachindex(to_update))] = true
        d = 1
    end

    δ = rand(rng, sampler.δ_spl)

    #set modified to 0
    xₚ[to_update] .= zero(eltype(x))

    #generate candidate
    for _ in 1:δ
        #pick to random chains find the difference and add to the candidate
        x₁, x₂ = pick_chains(state, current_state, 2)
        xₚ[to_update] .+= x₁[to_update] .- x₂[to_update]
    end

    #add the other parts of the equation
    xₚ[to_update] .= x[to_update] .+ (
        (1 .+ rand(rng, sampler.e_spl, d)) .* get_γ(rng, sampler, δ, d) .* xₚ[to_update]
    ) .+ rand(rng, sampler.ϵ_spl, d)

    return (offset = zero(eltype(xₚ)), cr = cr)
end

function get_γ(rng::AbstractRNG, sampler::DifferentialEvolutionSubspaceSampler, δ::Int, d::Int)
    return 2.38 / sqrt(2 * δ * d)
end

function get_γ(
        rng::AbstractRNG, sampler::DifferentialEvolutionSubspaceSamplerFixedGamma{T},
        δ::Int, d::Int
    ) where {T <: Real}
    return sampler.γ
end
