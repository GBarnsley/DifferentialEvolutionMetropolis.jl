# Initialising from Pathfinder

[Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) fits a multivariate normal
approximation to a target by following an L-BFGS optimisation path. It is cheap relative
to MCMC, and its draws make good starting points. This is recommended over the default `randn` initialisation
method.

When `Pathfinder` is loaded, any sampler in this package accepts a Pathfinder result as
`initial_position`. The pathfinder object is used to generate the starting position of every chain
and, for memory-based samplers, to populate the initial memory.

## Single-path Pathfinder

```@example pathfinder
using DifferentialEvolutionMetropolis, Pathfinder, AbstractMCMC, ForwardDiff
using LogDensityProblems, Distributions, LinearAlgebra, Random

# specify a gaussian far from the origin that the normal `randn` initial positions would be far form
struct FarGaussian end
LogDensityProblems.dimension(::FarGaussian) = 10
LogDensityProblems.capabilities(::Type{FarGaussian}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::FarGaussian, x) = logpdf(MvNormal(fill(50.0, 10), 0.01^2 * I), x)

ld = FarGaussian()
pf = pathfinder(ld; rng = Xoshiro(1))

result = deMCzs(
    AbstractMCMC.LogDensityModel(ld), 1000;
    initial_position = pf, N₀ = 60, silent = true, progress = false
)
extrema(result.samples)
```

The number of pathfinder draws don't need to match the number of positions used in DEMCMC sampler.
These draws become starting positions for both hot chains (if used), cold chains and fill the initial memory (specified by `N₀`).

## Multimodal targets

A single Pathfinder run converges to a single mode. For multimodal targets `multipathfinder`, whose runs start from different points, can land in different modes:

```@example pathfinder
struct Bimodal end
LogDensityProblems.dimension(::Bimodal) = 2
LogDensityProblems.capabilities(::Type{Bimodal}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::Bimodal, x) = logpdf(MixtureModel([MvNormal([-6.0, 0.0], I), MvNormal([6.0, 0.0], I)]), x)

bimodal = Bimodal()
mpf = multipathfinder(bimodal, 1000; nruns = 8, rng = Xoshiro(1))
result = DREAMz(
    AbstractMCMC.LogDensityModel(bimodal), 5000;
    initial_position = mpf, N₀ = 100, silent = true, progress = false
)
# should be near 50%
mean(result.samples[:, :, 1] .> 0)
```

By default, chain starting positions are stratified across the mixture components: chain
`i` starts from a draw of component `((i - 1) mod K) + 1`, where `K` is the number of
components. Every component gets at least one chain whenever
`n_chains + n_hot_chains ≥ K`.

Memory positions are plain draws from the mixture, importance-resampled when `importance = true` (sampling without replacement, via `resample` should be used if Pareto shape diagnostics are poor, to improve initial position diversity).

Stratification ignores the mixture weights. With `stratify_initial_position = false` starting positions from the mixture as well:

```@example pathfinder
result = DREAMz(
    AbstractMCMC.LogDensityModel(bimodal), 5000;
    initial_position = mpf, N₀ = 100, stratify_initial_position = false,
    silent = true, progress = false
)
mean(result.samples[:, :, 1] .> 0)
```

The share of draws in each mode reflects how many runs landed in that mode, not the mode's
posterior mass. This does not bias the sampler, after burn-in either the chain positions will no longer be dominated by the initial positions or the memory will largely be sample positions rather than pathfinder positions.
However, a mode that is not found by `multipathfinder` will not be represented and will have to be uncovered through the MCMC sampling process.

Separately obtained results can also be combined. A tuple or vector of `PathfinderResult`s
is treated as a uniform mixture of their fitted normals, with one component per result, and
chain starting positions are stratified in the same way:

```@example pathfinder
pf_left = pathfinder(bimodal; init = [-5.0, 0.0], rng = Xoshiro(2))
pf_right = pathfinder(bimodal; init = [5.0, 0.0], rng = Xoshiro(3))
result = deMCzs(
    AbstractMCMC.LogDensityModel(bimodal), 1000;
    initial_position = (pf_left, pf_right), silent = true, progress = false
)
mean(result.samples[:, :, 1] .> 0)
```

## Keyword arguments

- `initial_position`: a `PathfinderResult`, a `MultiPathfinderResult`, or a tuple/vector of
  `PathfinderResult`s.
- `stratify_initial_position`: if `true` (default), chain starting positions cycle through the
  mixture components. If `false`, they are plain draws from the mixture. Only affects
  `MultiPathfinderResult` and tuple/vector inputs.
