"""
    deMC(model_wrapper, n_its; kwargs...)

$(deMC_description)

The algorithm runs for a fixed number of iterations with optional burn-in.

# Arguments
- `model_wrapper`: A `LogDensityModel` or a target implementing the LogDensityProblems interface
- `n_its`: Number of sampling iterations per chain

# Keyword Arguments
- `rng`: Random number generator. Defaults to `default_rng()`.
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`.
$(deMC_kwargs)
$(template_chains_kwargs)
$(generic_de_kwargs_no_mem)
$(abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example deMC
using DifferentialEvolutionMetropolis, Random, Distributions, LinearAlgebra, LogDensityProblems

# Define an untransformed target on two real-valued parameters
struct StandardNormalTarget end
LogDensityProblems.dimension(::StandardNormalTarget) = 2
LogDensityProblems.capabilities(::Type{StandardNormalTarget}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::StandardNormalTarget, θ) = -sum(abs2, θ) / 2
model_wrapper = StandardNormalTarget()

# Run differential evolution MCMC
result = deMC(model_wrapper, 1000; n_chains = 10, parallel = false)
```

# Notes
$(generic_notes)

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`setup_de_update`](@ref).
"""
function deMC(
        model_wrapper, n_its::Int; n_burnin::Int = n_its * 5, save_burnt::Bool = false, kwargs...
    )

    model_wrapper = as_logdensity_model(model_wrapper)
    n_its, num_warmup = set_iterations(
        save_burnt, n_its, n_burnin
    )

    return _deMC(
        model_wrapper,
        n_its,
        num_warmup,
        save_burnt;
        kwargs...
    )
end


function _deMC(
        model_wrapper::LogDensityModel,
        N_or_is_done,
        num_warmup::Int,
        save_burnt::Bool;
        rng::AbstractRNG = default_rng(),
        γ₁::Union{Nothing, T} = nothing,
        γ₂::T = 1.0,
        p_γ₂::T = 0.1,
        β::ContinuousUnivariateDistribution = Normal(0.0, 1.0e-12),
        chain_type = DifferentialEvolutionOutput,
        memory = false,
        kwargs...
    ) where {T <: Real}

    #build sampler scheme
    if γ₁ != γ₂
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ₂, β = β, n_dims = dimension(model_wrapper.logdensity)),
            setup_de_update(γ = γ₁, β = β, n_dims = dimension(model_wrapper.logdensity));
            w = [p_γ₂, 1 - p_γ₂]
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ₁, β = β, n_dims = dimension(model_wrapper.logdensity))
        )
    end

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        N_or_is_done;
        num_warmup = num_warmup,
        discard_initial = save_burnt ? 0 : num_warmup,
        chain_type = chain_type,
        memory = memory,
        kwargs...
    )
end

"""
    deMCzs(model_wrapper, n_its; kwargs...)

$(deMCzs_description)

The algorithm runs for a fixed number of iterations with optional burn-in.

# Arguments
- `model_wrapper`: A `LogDensityModel` or a target implementing the LogDensityProblems interface
- `n_its`: Number of sampling iterations per chain

# Keyword Arguments
- `rng`: Random number generator. Defaults to `default_rng()`.
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`.
$(deMCzs_kwargs)
$(template_chains_kwargs)
$(generic_de_kwargs)
$(abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example deMCzs
using DifferentialEvolutionMetropolis, Random, Distributions, LinearAlgebra, LogDensityProblems

# Define an untransformed target on two real-valued parameters
struct StandardNormalTarget end
LogDensityProblems.dimension(::StandardNormalTarget) = 2
LogDensityProblems.capabilities(::Type{StandardNormalTarget}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::StandardNormalTarget, θ) = -sum(abs2, θ) / 2
model_wrapper = StandardNormalTarget()

# Run differential evolution MCMC
result = deMCzs(model_wrapper, 1000; n_chains = 3)
```

# Notes
$(generic_notes)

See also [`deMC`](@ref), [`DREAMz`](@ref).
"""
function deMCzs(
        model_wrapper, n_its::Int; n_burnin::Int = n_its * 5, save_burnt::Bool = false, kwargs...
    )

    model_wrapper = as_logdensity_model(model_wrapper)
    n_its, num_warmup = set_iterations(
        save_burnt, n_its, n_burnin
    )

    return _deMCzs(
        model_wrapper,
        n_its,
        num_warmup,
        save_burnt;
        kwargs...
    )
end

function _deMCzs(
        model_wrapper::LogDensityModel,
        N_or_is_done,
        num_warmup::Int,
        save_burnt::Bool;
        rng::AbstractRNG = default_rng(),
        γ::Union{Nothing, Real} = nothing,
        γₛ::Union{Nothing, Real} = nothing,
        p_snooker::Union{Nothing, Real} = 0.1,
        β::ContinuousUnivariateDistribution = Normal(0.0, 1.0e-12),
        chain_type = DifferentialEvolutionOutput,
        kwargs...
    )

    #build sampler scheme
    if p_snooker == 0
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ, β = β, n_dims = dimension(model_wrapper.logdensity))
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_de_update(γ = γ, β = β, n_dims = dimension(model_wrapper.logdensity)),
            setup_snooker_update(γ = γₛ),
            w = [1 - p_snooker, p_snooker]
        )
    end

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        N_or_is_done;
        num_warmup = num_warmup,
        discard_initial = save_burnt ? 0 : num_warmup,
        chain_type = chain_type,
        kwargs...
    )
end

"""
    DREAMz(model_wrapper, n_its; kwargs...)

$(DREAMz_description)

The algorithm runs for a fixed number of iterations with optional burn-in.

# Arguments
- `model_wrapper`: A `LogDensityModel` or a target implementing the LogDensityProblems interface
- `n_its`: Number of sampling iterations per chain

# Keyword Arguments
- `rng`: Random number generator. Defaults to `default_rng()`.
- `n_burnin`: Number of burn-in iterations. Defaults to `n_its * 5`.
$(DREAMz_kwargs)
$(template_chains_kwargs)
$(generic_de_kwargs)
$(abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example DREAMz
using DifferentialEvolutionMetropolis, Random, Distributions, LinearAlgebra, LogDensityProblems

# Define an untransformed target on two real-valued parameters
struct StandardNormalTarget end
LogDensityProblems.dimension(::StandardNormalTarget) = 2
LogDensityProblems.capabilities(::Type{StandardNormalTarget}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::StandardNormalTarget, θ) = -sum(abs2, θ) / 2
model_wrapper = StandardNormalTarget()

# Run DREAM with subspace sampling
result = DREAMz(model_wrapper, 1000; n_chains = 10, memory = false)
```
# Notes
$(generic_notes)

See also [`deMC`](@ref), [`deMCzs`](@ref), [`setup_subspace_sampling`](@ref).
"""
function DREAMz(
        model_wrapper, n_its::Int; n_burnin::Int = n_its * 5, save_burnt::Bool = false, kwargs...
    )

    model_wrapper = as_logdensity_model(model_wrapper)

    n_its, num_warmup = set_iterations(
        save_burnt, n_its, n_burnin
    )

    return _DREAMz(
        model_wrapper,
        n_its,
        num_warmup,
        save_burnt;
        kwargs...
    )
end

function _DREAMz(
        model_wrapper::LogDensityModel,
        N_or_is_done,
        num_warmup::Int,
        save_burnt::Bool;
        rng::AbstractRNG = default_rng(),
        γ₁::Union{Nothing, Real} = nothing,
        γ₂::Union{Nothing, Real} = 1.0,
        p_γ₂::Real = 0.2,
        n_cr::Int = 3,
        cr₁::Union{Nothing, Real, UnivariateDistribution} = nothing,
        cr₂::Union{Nothing, Real, UnivariateDistribution} = nothing,
        ϵ::ContinuousUnivariateDistribution = Distributions.Normal(0.0, 1.0e-12),
        e::ContinuousUnivariateDistribution = Distributions.Uniform(-0.1, 0.1),
        δ::Union{Integer, DiscreteUnivariateDistribution} = Distributions.DiscreteUniform(1, 3),
        cr_uniform_weight::Real = 0.05,
        min_variance_count::Int = 10,
        chain_type = DifferentialEvolutionOutput,
        kwargs...
    )

    #build sampler scheme
    if p_γ₂ == 0
        sampler_scheme = setup_sampler_scheme(
            setup_subspace_sampling(
                γ = γ₁, n_cr = n_cr, cr = cr₁, δ = δ, ϵ = ϵ, e = e,
                cr_uniform_weight = cr_uniform_weight, min_variance_count = min_variance_count
            )
        )
    else
        sampler_scheme = setup_sampler_scheme(
            setup_subspace_sampling(
                γ = γ₁, n_cr = n_cr, cr = cr₁, δ = δ, ϵ = ϵ, e = e,
                cr_uniform_weight = cr_uniform_weight, min_variance_count = min_variance_count
            ),
            setup_subspace_sampling(
                γ = γ₂, n_cr = n_cr, cr = cr₂, δ = δ, ϵ = ϵ, e = e,
                cr_uniform_weight = cr_uniform_weight, min_variance_count = min_variance_count
            ),
            w = Float64[1 - p_γ₂, p_γ₂]
        )
    end

    return sample(
        rng,
        model_wrapper,
        sampler_scheme,
        N_or_is_done;
        num_warmup = num_warmup,
        discard_initial = save_burnt ? 0 : num_warmup,
        chain_type = chain_type,
        kwargs...
    )
end

function set_iterations(
        save_burnt::Bool, n_its::Int, n_burnin::Int
    )
    if save_burnt
        n_its = n_its + n_burnin
    else
        n_its = n_its
    end
    num_warmup = n_burnin

    return n_its, num_warmup
end
