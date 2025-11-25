module MCMCDiagnosticToolsExt
# stopping criteria that uses rhat from MCMCDiagnosticTools

import DifferentialEvolutionMetropolis
import MCMCDiagnosticTools: rhat
import AbstractMCMC: LogDensityModel, sample, AbstractModel
import Random: AbstractRNG, default_rng

export deMC, deMCzs, DREAMz

"""
    r̂_stopping_criteria(rng, model, sampler, samples, state, iteration; kwargs...)

Stopping criterion based on the Gelman-Rubin diagnostic (R̂).

Sampling continues until the R̂ value for all parameters falls below `maximum_R̂`,
indicating convergence across chains. This function is designed to be used as the
`N_or_isdone` argument in `AbstractMCMC.sample` for adaptive stopping.

The diagnostic is computed on the last half of the collected samples to focus on
the stationary portion of the chains.

# Arguments
- `rng`: Random number generator (unused but required by AbstractMCMC interface)
- `model`: The model being sampled (unused but required by interface)
- `sampler`: The differential evolution sampler (unused but required by interface)
- `samples`: Vector of collected samples from all chains
- `state`: Current sampler state (unused but required by interface)
- `iteration`: Current iteration number

# Keyword Arguments
- `check_every`: Frequency (in iterations) for checking R̂ values. Defaults to 1000.
- `maximum_R̂`: Maximum acceptable R̂ value for convergence. Defaults to 1.2.
- `maximum_iterations`: Maximum number of iterations before forced stopping. Defaults to 100000.
- `minimum_iterations`: Minimum iterations before convergence checking begins. Defaults to 0.

# Returns
- `true` if sampling should stop (converged or maximum iterations reached)
- `false` if sampling should continue

# Example
```@example convergence
using DifferentialEvolutionMetropolis, AbstractMCMC, Random, Distributions

# Create a simple model
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Setup sampler
sampler = deMCzs()

# Use with adaptive stopping criterion
rng = Random.default_rng()
chains = sample(rng, model_wrapper, sampler, r̂_stopping_criteria;
               n_chains=4, check_every=500, maximum_R̂=1.1)
```

See also [`MCMCDiagnosticTools.rhat`](@extref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function DifferentialEvolutionMetropolis.r̂_stopping_criteria(
        rng::AbstractRNG,
        model::AbstractModel,
        sampler::DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionSampler,
        samples::Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionSample{V, VV}},
        state::DifferentialEvolutionMetropolis.DifferentialEvolutionState{T, A, L, M, V, VV},
        iteration::Int;
        check_every::Int = 1000,
        maximum_R̂::T = 1.2,
        maximum_iterations::Int = 100000,
        minimum_iterations::Int = 0,
        kwargs...
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A <: DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionAdaptiveState{T},
        M <: DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionMemory{T},
        L <: DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionTemperatureLadder{T},
    }
    if iteration % check_every != 0 || iteration < minimum_iterations
        return false
    elseif iteration >= maximum_iterations
        println("Maximum iterations reached: ", maximum_iterations)
        return true
    else
        #check the last half of the sampling iterations
        rhat_ = rhat(DifferentialEvolutionMetropolis.samples_to_array(samples[(iteration ÷ 2 + 1):end]))
        println("Rhat: ", round.(rhat_, sigdigits = 3))
        if all(rhat_ .< maximum_R̂)
            return true
        else
            return false
        end
    end
end

#extend templates to use this function too
"""
    deMC(model_wrapper, epoch_size, epoch_limit; kwargs...)

$(DifferentialEvolutionMetropolis.deMC_description)

The algorithm runs until the R̂ diagnostic indicates convergence or the maximum number of iterations is reached.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `epoch_size`: Number of iterations per chain per convergence check
- `epoch_limit`: Maximum number of total epochs

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs before convergence checking. Defaults to 5.
- `maximum_R̂`: Convergence threshold for Gelman-Rubin diagnostic. Defaults to 1.2
$(DifferentialEvolutionMetropolis.deMC_kwargs)
$(DifferentialEvolutionMetropolis.template_chains_kwargs)
$(DifferentialEvolutionMetropolis.generic_de_kwargs_no_mem)
$(DifferentialEvolutionMetropolis.abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example deMC_r
using DifferentialEvolutionMetropolis, Random, Distributions, MCMCDiagnosticTools

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run differential evolution MCMC with max 20 epochs
result = deMC(model_wrapper, 1000, 20; warmup_epochs = 5, n_chains = 10, parallel = false)
```

# Notes
$(DifferentialEvolutionMetropolis.generic_notes)

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`setup_de_update`](@ref), [`r̂_stopping_criteria`](@ref).
"""
function DifferentialEvolutionMetropolis.deMC(
        model_wrapper::LogDensityModel, epoch_size::Int, epoch_limit::Int;
        warmup_epochs::Int = 5, save_burnt::Bool = false, kwargs...
    )

    maximum_iterations, minimum_iterations, num_warmup = set_iterations(
        save_burnt, epoch_size, warmup_epochs, epoch_limit
    )

    return DifferentialEvolutionMetropolis._deMC(
        model_wrapper,
        DifferentialEvolutionMetropolis.r̂_stopping_criteria,
        num_warmup,
        save_burnt;
        minimum_iterations = minimum_iterations,
        maximum_iterations = maximum_iterations,
        check_every = epoch_size,
        kwargs...
    )
end

"""
    deMCzs(model_wrapper, epoch_size, epoch_limit; kwargs...)

$(DifferentialEvolutionMetropolis.deMCzs_description)

The algorithm runs until the R̂ diagnostic indicates convergence or the maximum number of iterations is reached.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `epoch_size`: Number of iterations per chain per convergence check
- `epoch_limit`: Maximum number of total epochs

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs before convergence checking. Defaults to 5.
- `maximum_R̂`: Convergence threshold for Gelman-Rubin diagnostic. Defaults to 1.2
$(DifferentialEvolutionMetropolis.deMCzs_kwargs)
$(DifferentialEvolutionMetropolis.template_chains_kwargs)
$(DifferentialEvolutionMetropolis.generic_de_kwargs)
$(DifferentialEvolutionMetropolis.abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example deMCzs_r
using DifferentialEvolutionMetropolis, Random, Distributions, MCMCDiagnosticTools

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run differential evolution MCMC
result = deMCzs(model_wrapper, 1000; n_chains = 3, maximum_R̂ = 1.1)
```

# Notes
$(DifferentialEvolutionMetropolis.generic_notes)

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`r̂_stopping_criteria`](@ref).
"""
function DifferentialEvolutionMetropolis.deMCzs(
        model_wrapper::LogDensityModel, epoch_size::Int, epoch_limit::Int;
        warmup_epochs::Int = 5, save_burnt::Bool = false, kwargs...
    )

    maximum_iterations, minimum_iterations, num_warmup = set_iterations(
        save_burnt, epoch_size, warmup_epochs, epoch_limit
    )

    return DifferentialEvolutionMetropolis._deMCzs(
        model_wrapper,
        DifferentialEvolutionMetropolis.r̂_stopping_criteria,
        num_warmup,
        save_burnt;
        minimum_iterations = minimum_iterations,
        maximum_iterations = maximum_iterations,
        check_every = epoch_size,
        kwargs...
    )
end

"""
    DREAMz(model_wrapper, epoch_size, epoch_limit; kwargs...)

$(DifferentialEvolutionMetropolis.DREAMz_description)

The algorithm runs until the R̂ diagnostic indicates convergence or the maximum number of iterations is reached.

# Arguments
- `model_wrapper`: LogDensityModel containing the target log-density function
- `epoch_size`: Number of iterations per chain per convergence check
- `epoch_limit`: Maximum number of total epochs

# Keyword Arguments
- `warmup_epochs`: Number of warm-up epochs before convergence checking. Defaults to 5.
- `maximum_R̂`: Convergence threshold for Gelman-Rubin diagnostic. Defaults to 1.2
$(DifferentialEvolutionMetropolis.DREAMz_kwargs)
$(DifferentialEvolutionMetropolis.template_chains_kwargs)
$(DifferentialEvolutionMetropolis.generic_de_kwargs)
$(DifferentialEvolutionMetropolis.abstract_mcmc_kwargs)

# Returns
- depends on `chain_type`, and `save_final_state`

# Example
```@example DREAMz_r
using DifferentialEvolutionMetropolis, Random, Distributions, MCMCDiagnosticTools

# Define a simple log-density function
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)

# Run DREAM with subspace sampling
result = DREAMz(model_wrapper, 1000; n_chains = 10, memory = false)
```

# Notes
$(DifferentialEvolutionMetropolis.generic_notes)

See also [`deMCzs`](@ref), [`DREAMz`](@ref), [`setup_subspace_sampling`](@ref), [`r̂_stopping_criteria`](@ref).
"""
function DifferentialEvolutionMetropolis.DREAMz(
        model_wrapper::LogDensityModel, epoch_size::Int, epoch_limit::Int;
        warmup_epochs::Int = 5, save_burnt::Bool = false, kwargs...
    )

    maximum_iterations, minimum_iterations, num_warmup = set_iterations(
        save_burnt, epoch_size, warmup_epochs, epoch_limit
    )

    return DifferentialEvolutionMetropolis._DREAMz(
        model_wrapper,
        DifferentialEvolutionMetropolis.r̂_stopping_criteria,
        num_warmup,
        save_burnt;
        minimum_iterations = minimum_iterations,
        maximum_iterations = maximum_iterations,
        check_every = epoch_size,
        kwargs...
    )
end

function set_iterations(
        save_burnt::Bool, epoch_size::Int, warmup_epochs::Int, epoch_limit::Int
    )
    num_warmup = epoch_size * warmup_epochs

    if save_burnt
        minimum_iterations = num_warmup + 1 #so we don't check during warmup
        maximum_iterations = (epoch_size * epoch_limit) + num_warmup
    else
        minimum_iterations = 0
        maximum_iterations = epoch_size * epoch_limit
    end

    return maximum_iterations, minimum_iterations, num_warmup
end

end
