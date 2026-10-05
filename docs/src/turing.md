# Using a Turing model

An existing [Turing.jl](https://turinglang.org/) model can be sampled without rewriting its priors or likelihood.
Its modelling backend, DynamicPPL, exposes `LogDensityProblems` targets that this package already accepts.
No Turing-specific sampler extension is needed for this route.

## Model → log density → ensemble

The example below uses DynamicPPL 0.42 (the backend used by Turing 0.49).
For an existing Turing model, keep your `using Turing` and `@model` definition; start at the `LogDensityFunction` construction instead.
Add DynamicPPL as a direct dependency of your environment to use its API explicitly.
Older DynamicPPL releases have different constructors, so check their versioned documentation.

This recipe does not need Pathfinder.
Turing 0.49 currently restricts its optional Pathfinder integration to versions up to 0.7, while this package supports Pathfinder 0.10.
Installing all three together therefore produces a dependency conflict; use the direct log-density route without Pathfinder in that environment.
The documentation environment loads only DynamicPPL, so it can still build the separate Pathfinder examples.

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

# Illustrative observations, not a data file.
model = scale_model([-1.0, 0.5, 1.5])
ld = DynamicPPL.LogDensityFunction(
    model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll()
)

# LinkAll maps positive σ to log(σ); the target includes the Jacobian.
x = [log(2.0)]
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

Use `getlogjoint_internal`, not just the likelihood: it includes priors and the change-of-variables correction when linked.
The default `LogDensityFunction(model)` is **unlinked** in this DynamicPPL version.
Omitting `LinkAll()` would leave positive or bounded parameters constrained, which is unsuitable for unrestricted DE proposals.
`rand(rng, ld)` generates prior initial positions in the same linked coordinates.
Memory-based `deMCzs` and `DREAMz` accept the same wrapper; supply enough initial positions to populate their initial memory if you want that memory drawn from the prior rather than the default standard normal.

## Recovering model-space values

`result.samples` has axes `(iteration, chain, parameter)` and contains **linked** parameters.
`result.ld` is the linked log joint, not the original model-space density.
Do not assume vector order matches model declaration order for a general model.
DynamicPPL can reconstruct named, constrained values using the same log-density object:

```@example turing
x = vec(result.samples[1, 1, :])
params = DynamicPPL.ParamsWithStats(x, ld).params
σ_draw = params[DynamicPPL.@varname(σ)]
@assert σ_draw > 0
@assert σ_draw ≈ exp(only(x))
σ_draw
```

Apply this conversion to each iteration and chain when constructing summaries.
Converting directly to `MCMCChains.Chains` with this package's extension changes the container, not the coordinate system: values remain linked unless you explicitly transform them.
Use fixed-dimensional continuous models for this recipe.
Discrete latent variables, parameter-dependent model structure, or custom transforms need separate consideration; this is not a general Gibbs integration.

## Can this be a Turing external sampler?

Not directly.
In Turing 0.49 the wrapper is named `Turing.externalsampler` (the spelling/API differs in older releases).
Both packages use AbstractMCMC, so the log-density and stepping interfaces are already close, but their sampling units differ:

- This package: an interacting ensemble of positions, log densities, memory and adaptation.
- Turing external sampler: one parameter vector for one Turing chain.

Turing extracts a single vector via `AbstractMCMC.getparams(model, state)` and statistics via `AbstractMCMC.getstats(state)`.
This package has neither extraction method for its ensemble state.
Turing also supplies one vector as `initial_params`, whereas this package uses `initial_position`, a vector of vectors.
Merely adding a `getparams` method that returns the ensemble matrix cannot satisfy that contract.

A feasible **separate adapter** would hold the full ensemble internally, initialise all auxiliary chains (respecting minimum chain counts and memory requirements), and expose one designated cold chain as the Turing chain.
It would need to translate initialisation, route warm-up correctly, declare unconstrained coordinates, and return one vector plus appropriate statistics.
The auxiliary chains would not appear in Turing's output; running several Turing chains would create several independent ensembles, with the corresponding extra cost.
Exposing *every* ensemble member as Turing chains instead requires custom output bundling rather than the ordinary external-sampler wrapper.

This is achievable, but is more than a wrapper alias and is not implemented here.
Gibbs support adds another problem: reconditioning must refresh cached log densities and invalidate or rebuild history/adaptation based on the old target.
The simple adapter should explicitly disallow Gibbs until that behaviour is designed and tested.
For now, the direct log-density route above preserves all ensemble draws and avoids these adapter semantics.

See the [DynamicPPL log-density API](https://turinglang.org/DynamicPPL.jl/stable/api/) and [Turing external-sampler interface](https://turinglang.org/Turing.jl/stable/api/Inference/).
