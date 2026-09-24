# Initialising from Pathfinder

!!! warning "Proposed interface"
    This page describes the planned `PathfinderExt` extension (issue #102). It is not yet
    implemented and is not included in the built documentation.

[Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) fits a multivariate normal
approximation to a target by following an L-BFGS optimisation path. It is cheap relative
to MCMC, and its draws make good starting points: they sit in regions of high posterior
mass rather than around the origin, which is where the default `randn` initialisation
places chains.

When `Pathfinder` is loaded, any sampler in this package accepts a Pathfinder result as
`initial_position`. The result is used to generate the starting position of every chain
and, for memory-based samplers, the `N₀` positions of the initial memory.

## Single-path Pathfinder

```julia
using DifferentialEvolutionMetropolis, Pathfinder, AbstractMCMC
using LogDensityProblems, Distributions, LinearAlgebra, Random

struct FarGaussian end
LogDensityProblems.dimension(::FarGaussian) = 10
LogDensityProblems.capabilities(::Type{FarGaussian}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.logdensity(::FarGaussian, x) = logpdf(MvNormal(fill(50.0, 10), 0.01^2 * I), x)

ld = FarGaussian()
pf = pathfinder(ld; rng = Xoshiro(1))

result = deMCzs(AbstractMCMC.LogDensityModel(ld), 1000; initial_position = pf, N₀ = 60)
```

Pathfinder differentiates order-0 `LogDensityProblems` targets itself (ForwardDiff by
default, or pass `adtype`), so no `LogDensityProblemsAD` wrapper is needed.

`n_chains + n_hot_chains + N₀` positions are drawn from `pf.fit_distribution`, so the
number of draws stored in `pf` does not need to match. The first `n_chains + n_hot_chains`
become the chain starting positions and the remaining `N₀` fill the initial memory.

## Multimodal targets

A single Pathfinder run converges to one mode. For multimodal targets use
`multipathfinder`, whose runs start from different points and can land in different
modes:

```julia
mpf = multipathfinder(ld, 1000; nruns = 8, rng = Xoshiro(1))
result = DREAMz(AbstractMCMC.LogDensityModel(ld), 5000; initial_position = mpf, N₀ = 100)
```

Starting positions are drawn from `mpf.draws` (importance-resampled when `importance = true`)
and topped up from `mpf.fit_distribution` if more positions are needed.

The share of draws in each mode reflects how many runs landed in that mode, not the mode's
posterior mass. This does not bias the sampler, which corrects the weights, but a mode that
no run found will not be represented. Increase `nruns` or supply your own starting points
through `init` when modes are hard to find.

Separately obtained results can also be combined. A tuple or vector of `PathfinderResult`s
is treated as a uniform mixture of their fitted normals:

```julia
pf_left = pathfinder(ld; init = fill(-5.0, 10))
pf_right = pathfinder(ld; init = fill(5.0, 10))
result = deMCzs(AbstractMCMC.LogDensityModel(ld), 1000; initial_position = (pf_left, pf_right))
```

## Keyword arguments

- `initial_position`: a `PathfinderResult`, a `MultiPathfinderResult`, or a tuple/vector of
  `PathfinderResult`s.
- `N₀`: number of Pathfinder draws placed in the initial memory. Ignored for memoryless samplers.
