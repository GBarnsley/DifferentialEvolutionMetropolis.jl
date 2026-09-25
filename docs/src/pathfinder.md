# Initialising from Pathfinder

[Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) fits a multivariate normal
approximation to a target by following an L-BFGS optimisation path. It is cheap relative
to MCMC, and its draws make good starting points: they sit in regions of high posterior
mass rather than around the origin, which is where the default `randn` initialisation
places chains.

When `Pathfinder` is loaded, any sampler in this package accepts a Pathfinder result as
`initial_position`. The result is used to generate the starting position of every chain
and, for memory-based samplers, the `N₀` positions of the initial memory.

## Single-path Pathfinder

```@example pathfinder
using DifferentialEvolutionMetropolis, Pathfinder, AbstractMCMC, ForwardDiff
using LogDensityProblems, Distributions, LinearAlgebra, Random

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

Pathfinder differentiates order-0 `LogDensityProblems` targets itself (ForwardDiff by
default, or pass `adtype`), so no `LogDensityProblemsAD` wrapper is needed.

`n_chains + n_hot_chains + N₀` positions are drawn from `pf.fit_distribution`, so the
number of draws stored in `pf` does not need to match. The first `n_chains + n_hot_chains`
become the chain starting positions and the remaining `N₀` fill the initial memory. If `N₀`
is not given, the memory holds the chain starting positions plus `n_chains + n_hot_chains`
further draws, matching the default `N₀ = 2 * (n_chains + n_hot_chains)`.

## Multimodal targets

A single Pathfinder run converges to one mode. For multimodal targets use
`multipathfinder`, whose runs start from different points and can land in different
modes:

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
mean(result.samples[:, :, 1] .> 0)
```

By default, chain starting positions are stratified across the mixture components: chain
`i` starts from a draw of component `((i - 1) mod K) + 1`, where `K` is the number of
components. Every component gets at least one chain whenever
`n_chains + n_hot_chains ≥ K`. The `N₀` memory positions are plain draws from the mixture:
taken from `mpf.draws` (importance-resampled when `importance = true`) and topped up from
`mpf.fit_distribution` if more positions are needed.

Stratification ignores the mixture weights. Pass `stratify_initial_position = false` to draw
the chain starting positions from the mixture as well:

```@example pathfinder
result = DREAMz(
    AbstractMCMC.LogDensityModel(bimodal), 5000;
    initial_position = mpf, N₀ = 100, stratify_initial_position = false,
    silent = true, progress = false
)
mean(result.samples[:, :, 1] .> 0)
```

The share of draws in each mode reflects how many runs landed in that mode, not the mode's
posterior mass. This does not bias the sampler, which corrects the weights, but a mode that
no run found will not be represented. Increase `nruns` or supply your own starting points
through `init` when modes are hard to find.

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
- `N₀`: number of Pathfinder draws placed in the initial memory. Ignored for memoryless samplers.
- `stratify_initial_position`: if `true` (default), chain starting positions cycle through the
  mixture components. If `false`, they are plain draws from the mixture. Only affects
  `MultiPathfinderResult` and tuple/vector inputs.
