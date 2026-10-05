# Initialising from Pathfinder

[Pathfinder.jl](https://github.com/mlcolab/Pathfinder.jl) fits a multivariate normal approximation to a target by following an L-BFGS optimisation path.
It is cheap relative to MCMC, and its draws make good starting points.
This is recommended over the default `randn` initialisation method.

When `Pathfinder` is loaded, any sampler in this package accepts a Pathfinder result as `initial_position`.
The pathfinder object is used to generate the starting position of every chain and, for memory-based samplers, to populate the initial memory.

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

A single Pathfinder run converges to a single mode.
For multimodal targets `multipathfinder`, whose runs start from different points, can land in different modes:

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

By default, chain starting positions are stratified across the mixture components: chain `i` starts from a draw of component `((i - 1) mod K) + 1`, where `K` is the number of components.
Every component gets at least one chain whenever `n_chains + n_hot_chains ≥ K`.

Memory positions are plain draws from the mixture, importance-resampled when `importance = true` (sampling without replacement, via `resample` should be used if Pareto shape diagnostics are poor, to improve initial position diversity).

Stratification ignores the mixture weights.
With `stratify_initial_position = false` starting positions from the mixture as well:

```@example pathfinder
result = DREAMz(
    AbstractMCMC.LogDensityModel(bimodal), 5000;
    initial_position = mpf, N₀ = 100, stratify_initial_position = false,
    silent = true, progress = false
)
mean(result.samples[:, :, 1] .> 0)
```

The share of draws in each mode reflects how many runs landed in that mode, not the mode's posterior mass.
This does not bias the sampler, after burn-in either the chain positions will no longer be dominated by the initial positions or the memory will largely be sample positions rather than pathfinder positions.
However, a mode that is not found by `multipathfinder` will not be represented and will have to be uncovered through the MCMC sampling process.

Separately obtained results can also be combined.
A tuple or vector of `PathfinderResult`s is treated as a uniform mixture of their fitted normals, with one component per result, and chain starting positions are stratified in the same way:

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

- `initial_position`: a `PathfinderResult`, a `MultiPathfinderResult`, or a tuple/vector of `PathfinderResult`s.
- `stratify_initial_position`: if `true` (default), chain starting positions cycle through the mixture components.
  If `false`, they are plain draws from the mixture.
  Only affects `MultiPathfinderResult` and tuple/vector inputs.

## Should warm-up prefer high-log-density donors?

Density-weighted donor selection is not currently implemented.
For now, prefer multipath Pathfinder initialisation plus the existing DREAM crossover adaptation.
This is a design assessment, not a measured speedup claim: density weighting could help a poorly initialised population, but there is no evidence here that it improves on a well-initialised one.

### What the proposed weighting would change

- **Snooker hinge selection:** a hinge in a well-fitting region changes the projection direction.
  The signed projected donor difference can move toward **or away from** the hinge; it is not a guaranteed pull toward a mode.
- **DREAM difference-vector donors:** donors concentrated in one region can yield smaller differences.
  High density does not imply nearby donors across modes; concentration can erase useful diversity.
- **DREAM crossover selection:** existing adaptation favours crossover values with larger variance-normalised accepted squared jumps.
  This selects how many coordinates to update, not high-density parents or individual coordinate importance.

Small coordinate updates do not necessarily follow a narrow diagonal ridge.
In a strongly correlated target, changing one coordinate can leave the ridge even when the starting point lies on it.
High acceptance from tiny steps is not itself good mixing, so acceptance rate alone is not a justification for weighting.

Multipath Pathfinder already supplies both starting chains and historical donors in fitted regions with local covariance information.
Stratified starts preserve representation of the fitted components.
Weighting those donors again by density can duplicate the initialisation benefit while disproportionately favouring a narrow, high-density mode over a broad mode with more posterior mass.
It cannot recover modes that Pathfinder never found.
Pathfinder's importance-resampled draws and raw log-density-weighted selection are also different: importance weights account for the proposal density, whereas raw density weights do not.

There remains a plausible niche when Pathfinder is unavailable (for example, without usable gradients), misses important local geometry, or leaves a few chains far from the fitted regions.
Treat density weighting there as an optional tuning experiment, not a default or a substitute for checking initialisation and geometry.

### How an experiment would fit the package

No new dependency is needed: `StatsBase` and `AliasTables` already provide weighted selection building blocks.
However, simply replacing `pick_chains` is insufficient:

- Historical memory in `src/memory.jl` stores positions, not log densities.
  A density cache would have to be populated for initial memory and updated with exactly the same thinning, refill, and growth operations as the positions.
  Re-evaluating all donors on every proposal would be expensive; chain-specific models also require deciding which model defines a donor's score.
- Preserve distinct donors, exclusion of the current chain in memoryless sampling, per-chain RNGs, and immutable selection weights during a threaded sweep.
  Define behaviour for non-finite densities and insufficient positive-weight donors.
- A tempered, uniform-mixed score such as `w_i = (1 - alpha) / M + alpha * softmax(beta * ld)_i` would limit concentration.
  This is a proposed policy, not an existing keyword; both strengths would need validation and diagnostics for effective donor count and mode coverage.
- Keep hinge weighting separate from DREAM parent weighting.
  Keep selection of crossover probabilities in the existing jump-distance adaptation rather than claiming that donor fitness identifies useful coordinates.
- Scope the policy explicitly to `step_warmup`.
  The internal `fix_sampler` is used *during* adaptive DREAM steps as well as when transitioning to ordinary `step`, so stripping a warm-up policy there unconditionally would disable it too early.
  Ordinary `step` must use uniform selection, including resumed sampling and composite samplers.
- The current snooker acceptance offset is a geometric norm ratio.
  State-dependent donor probabilities can introduce additional proposal asymmetry; that offset does not automatically correct it.
  A warm-up-only heuristic must be described as potentially non-stationary and its draws discarded.
  Any retained-sampling variant needs a detailed-balance argument and appropriate proposal correction.
- Biased warm-up can leave concentrated historical memory and crossover statistics after weights become uniform.
  Allow a subsequent uniform warm-up phase to refresh memory and retune crossover probabilities before retaining draws; merely switching off weighting does not instantly remove its influence.

Before adding this API, compare uniform and weighted warm-up with and without multipath Pathfinder on correlated Gaussian, curved-ridge, and unequal-mode targets.
Use repeated seeds, the same density/gradient evaluation budget, and held-out uniform sampling.
Measure effective sample size per evaluation, convergence, mode coverage, wall time, and donor diversity, not only warm-up acceptance.
Include memoryless, thinned/refilling memory, serial/threaded, and warm-up-to-sampling transition checks.
Only a reproducible improvement beyond Pathfinder plus crossover adaptation would justify implementing the extra cache and policy machinery.
