# Customizing your sampler

This document describes how to extend `DifferentialEvolutionMetropolis.jl` with your own custom components. You can define custom stopping criteria, diagnostic checks, and proposal distributions (updates).

## Custom Stopping Criteria

To create a custom stopping criterion, you need to define a function that follows the AbstractMCMC interface, similar to [`r̂_stopping_criteria`](@ref) which is available if `MCMCDiagnosticTools.jl` is loaded. The function will be called during sampling to determine when to stop. See the [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/dev/api/#Sampling-a-single-chain) for details.

The stopping criterion function has the following signature:
```julia
function your_stopping_criteria(
    rng::AbstractRNG,
    model::AbstractModel, 
    sampler::AbstractDifferentialEvolutionSampler,
    samples::Vector{DifferentialEvolutionSample},
    state::DifferentialEvolutionState,
    iteration::Int;
    kwargs...
)
```

- `rng`: Random number generator (required by AbstractMCMC interface, usually unused)
- `model`: The model being sampled (required by interface, usually unused)  
- `sampler`: The differential evolution sampler (required by interface, usually unused)
- `samples`: Vector of all collected samples from all chains
- `state`: Current sampler state (required by interface, usually unused)
- `iteration`: Current iteration number
- `kwargs...`: Additional keyword arguments passed via `AbstractMCMC.sample`

The function should return `true` if sampling should stop, and `false` otherwise.

Here is an example of a very simple stopping criterion that stops sampling after a maximum number of iterations has been reached:

```julia
using DifferentialEvolutionMetropolis, AbstractMCMC

function max_iterations_stopping(
    rng::AbstractRNG,
    model::AbstractModel,
    sampler::AbstractDifferentialEvolutionSampler, 
    samples::Vector{DifferentialEvolutionSample{V, VV}},
    state::DifferentialEvolutionState{T, V, VV, A},
    iteration::Int;
    max_iterations::Int = 10000,
    kwargs...
) where {T<:Real, V<:AbstractVector{T}, VV<:AbstractVector{V}, A<:AbstractDifferentialEvolutionAdaptiveState{T}}
    if iteration >= max_iterations
        println("Reached maximum iterations ($max_iterations), stopping.")
        return true
    end
    return false
end

# Usage with AbstractMCMC.sample
using LogDensityProblems

model = LogDensityModel(your_log_density)
sampler = setup_de_update()

result = sample(
    rng, 
    model, 
    sampler, 
    max_iterations_stopping;
    max_iterations = 5000,  # passed as keyword argument
    n_chains = 4
)
```

## Custom Proposal Distributions
You can create your own proposal distributions by defining a new sampler type that subtypes `AbstractDifferentialEvolutionSampler` and implementing the `proposal` method.

The method signature for the proposal is:
```julia
function proposal!(
    state::DifferentialEvolutionMetropolis.DifferentialEvolutionState, 
    sampler::YourSampler, 
    current_state::Int
)
```

- `state`: The current state containing all chain positions, log-densities, and chain specific rngs
- `sampler`: An instance of your custom sampler struct
- `current_state`: The index of the chain to be updated


The function should modify `state.xₚ[current_state] = proposed_position` and return a named tuple with at least `(offset = hastings_correction)` where:
- `proposed_position`: The proposed new position (vector)
- `offset`: Hastings ratio correction in log-space (typically 0.0 for symmetric proposals)

Here is an example of a simple Metropolis-Hastings random walk update with a fixed step size:

```@example MHSampler
using DifferentialEvolutionMetropolis, Distributions, Random

# Define the struct for the sampler
struct MetropolisHastingsUpdate <: DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionSampler
    proposal_distribution::MvNormal
end

# Implement the proposal function
function DifferentialEvolutionMetropolis.proposal!(
    state::DifferentialEvolutionMetropolis.DifferentialEvolutionState,
    sampler::MetropolisHastingsUpdate,
    current_state::Int
)
    # Get the current position of this chain
    x_current = state.x[current_state]
    
    # Propose a new point (stored in state) using a random walk
    state.xₚ[current_state] .= x_current .+ rand(state.rngs[current_state], sampler.proposal_distribution)
    
    # The proposal is symmetric, so no Hastings correction needed
    return (offset = 0.0)
end
```

### Adaptive Proposals with step_warmup

For proposals that require adaptation during warm-up, you need to implement the `step_warmup` as well. This is called during the warm-up phase. Unless you want your sampler to be always adaptive then you must implement `step`.

You'll also need to define adaptive state structures and methods. Here's an example of an adaptive Metropolis-Hastings sampler:

```@example MHSampler
using AbstractMCMC, DifferentialEvolutionMetropolis
# Define adaptive state
struct AdaptiveMetropolisState{T<:Real} <:DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionAdaptiveState{T}
    proposal_cov::Matrix{T}
    adaptation_count::Int
    running_mean::Vector{T}
    running_cov::Matrix{T}
end

# Define the adaptive sampler  
struct AdaptiveMetropolisUpdate{T<:Real} <: DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionSampler
    initial_cov::Matrix{T}
    adapt_after::Int
    adapt_every::Int
    adapt_scale::T
end

# Constructor
function AdaptiveMetropolisUpdate(
    n_params::Int;
    initial_std::Float64 = 0.1,
    adapt_after::Int = 200,
    adapt_every::Int = 100,
    adapt_scale::Float64 = 2.38^2
)
    initial_cov = (initial_std^2) * I(n_params)
    return AdaptiveMetropolisUpdate{Float64}(initial_cov, adapt_after, adapt_every, adapt_scale / n_params)
end

# Initialize adaptive state
function DifferentialEvolutionMetropolis.initialize_adaptive_state(sampler::AdaptiveMetropolisUpdate{T}, model_wrapper::AbstractMCMC.LogDensityModel, n_chains::Int) where {T}
    n_params = dimension(model_wrapper.logdensity)
    return AdaptiveMetropolisState{T}(
        copy(sampler.initial_cov),
        0,
        zeros(T, n_params),
        copy(sampler.initial_cov)
    )
end

# Fix sampler (convert adaptive to non-adaptive)
function DifferentialEvolutionMetropolis.fix_sampler(sampler::AdaptiveMetropolisUpdate{T}, adaptive_state::AdaptiveMetropolisState{T}) where {T}
    return MetropolisHastingsUpdate(MvNormal(zeros(T, size(adaptive_state.proposal_cov, 1)), adaptive_state.proposal_cov))
end

# Proposal method (same as non-adaptive)
function DifferentialEvolutionMetropolis.proposal!(
    state::DifferentialEvolutionMetropolis.DifferentialEvolutionState,
    sampler::AdaptiveMetropolisUpdate,
    current_state::Int
)
    x_current = state.x[current_state]
    # Use current proposal covariance from adaptive state
    state.xₚ[current_state] .= rand(rng, MvNormal(x_current, state.adaptive_state.proposal_cov))
    return (offset = 0.0)
end

# Adaptive step during warm-up
function step_warmup(
    rng::AbstractRNG,
    model_wrapper::AbstractMCMC.LogDensityModel,
    sampler::AdaptiveMetropolisUpdate{T},
    state::DifferentialEvolutionMetropolis.DifferentialEvolutionState{T, AdaptiveMetropolisState{T}};
    parallel::Bool = false,
    kwargs...
) where {T<:Real}
    
    # Perform regular step
    sample, new_state = step(rng, model_wrapper, sampler, state; parallel = parallel, kwargs...)
    
    # Update adaptive parameters
    adapt_state = new_state.adaptive_state
    new_count = adapt_state.adaptation_count + 1
    
    # Only adapt after burn-in period and at specified intervals
    if new_count > sampler.adapt_after && new_count % sampler.adapt_every == 0
        
        # Compute empirical covariance from current chain positions
        positions = reduce(hcat, new_state.x)'  # Convert to matrix
        empirical_cov = cov(positions)
        
        # Update proposal covariance with regularization
        new_proposal_cov = sampler.adapt_scale * empirical_cov + 1e-6 * I
        
        # Update adaptive state
        new_adaptive_state = AdaptiveMetropolisState{T}(
            new_proposal_cov,
            new_count,
            adapt_state.running_mean,  # Could update these too
            adapt_state.running_cov
        )
        
        return sample, update_state(new_state; adaptive_state = new_adaptive_state)
    else
        # Just update the count
        new_adaptive_state = AdaptiveMetropolisState{T}(
            adapt_state.proposal_cov,
            new_count,
            adapt_state.running_mean,
            adapt_state.running_cov
        )
        return sample, update_state(new_state; adaptive_state = new_adaptive_state)
    end
end
```

## Example: Using Custom Components

Here is a complete example that shows how to use custom components with the new AbstractMCMC interface:

```@example MHSampler
using Distributions, TransformedLogDensities, LinearAlgebra, TransformVariables, Plots

# Set up a simple log-density to sample from (a 2D standard normal distribution)

ld = TransformedLogDensity(as(Array, 2), x -> -sum(x.^2) / 2)
dimension(ld) = 2
model = AbstractMCMC.LogDensityModel(ld)

# Create custom samplers
simple_mh = MetropolisHastingsUpdate(MvNormal([0.0, 0.0], 0.1 * I))
adaptive_mh = AdaptiveMetropolisUpdate(2; initial_std = 0.1)

# Create a composite sampler scheme
my_sampler_scheme = setup_sampler_scheme(
    simple_mh, 
    adaptive_mh;
    w = [0.3, 0.7]  # Use adaptive sampler 70% of the time
)

# Sample using AbstractMCMC.sample 
result = sample(
    Random.default_rng(),
    model,
    my_sampler_scheme,
    5000;
    n_chains = 6,
    num_warmup = 10000, #adaptive steps
    memory = true,
    parallel = false,
    chain_type = DifferentialEvolutionOutput
)

plot(result.samples[:, :, 1])
plot(result.samples[:, :, 2])
```