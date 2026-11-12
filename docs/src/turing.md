# Using a Turing model

An existing [Turing.jl](https://turinglang.org/) model can be sampled directly.
Its modelling backend, `DynamicPPL.jl`, allows the construction of a `LogDensityProblem` that this package accepts.

## Converting your model

The example below uses `DynamicPPL.jl` and does not explicitly call `Turing` (due to compatibility problems with `Pathfinder`, used elsewhere in these examples) but the `@model` can be defined identically using `Turing`.

```@example turing
using DynamicPPL, Distributions, Random, AbstractMCMC
using DifferentialEvolutionMetropolis, LogDensityProblems

@model function scale_model(y)
    σ ~ Exponential(1)
    for i in eachindex(y)
        y[i] ~ Normal(0, σ)
    end
    return σ
end

model = scale_model([-1.0, 0.5, 1.5])
ld = DynamicPPL.LogDensityFunction(
    model, #your turing model
    DynamicPPL.getlogjoint_internal, #likelihood, priors, and corrections
    DynamicPPL.LinkAll() #resulting logdensity should be unconstrained (accepts real values)
)

x = [log(2.0)]
# LinkAll maps positive σ to log(σ), so the target includes the Jacobian.
expected = logpdf(Exponential(1), 2.0) +
    sum(logpdf(Normal(0, 2.0), yi) for yi in [-1.0, 0.5, 1.5]) + log(2.0)
@assert LogDensityProblems.logdensity(ld, x) ≈ expected

rng = Xoshiro(79)
n_chains = 6
initial_position = [rand(rng, ld) for _ in 1:n_chains]
result = deMC(
    AbstractMCMC.LogDensityModel(ld), 500;
    rng, n_chains, initial_position, n_burnin = 1000,
    parallel = false, silent = true, progress = false
)
@assert size(result.samples) == (500, n_chains, 1)
@assert all(isfinite, result.ld)
size(result.samples)
```

## Recovering model-space values

`result.samples` has axes `(iteration, chain, parameter)` and contains **linked** parameters (meaning real-valued unconstrained values).
`result.ld` is the linked log joint, not the original model-space density.
**Do not** assume vector order matches model declaration order.
DynamicPPL can reconstruct named, constrained values (as you would get from a Turing output) using the same log-density object:

```@example turing
x = vec(result.samples[1, 1, :]) #select parameters from the first iteration and chain
params = DynamicPPL.ParamsWithStats(x, ld).params
σ_draw = params[DynamicPPL.@varname(σ)]
@assert σ_draw > 0
@assert σ_draw ≈ exp(only(x))
σ_draw
```

Apply this conversion to each iteration and chain when constructing summaries.
Converting directly to `MCMCChains.Chains` or `FlexiChains` within `sample` will still give linked values.
Use fixed-dimension continuous models for this recipe; these samplers do not support discrete values or models with changing dimensions.

See the [DynamicPPL log-density API](https://turinglang.org/DynamicPPL.jl/stable/api/).

## Can this be used as a Turing external sampler?

Not really.
`Turing` can only accept one set of positions instead of the ensemble, meaning any `Turing.externalsampler` implementation would be quite wasteful, as only one DEM position could be recorded or exposed to other samplers.
Most `Turing` integrated samplers like NUTS and HMC are already accessible (see [Combining HMC with differential evolution](@ref)) or would be simple enough to implement (see [Customizing your sampler](@ref)).
Implementing differential evolution updates within a Gibbs update or alongside particle-Gibbs (or any particle filter or reversible jump MCMC) is not straightforward, due to adaptation to past parameter spaces and/or changing model dimensions, and is not implemented here.
