# Combining HMC with differential evolution

Hamiltonian Monte Carlo (HMC, and its self-tuning variant NUTS) uses gradients to glide
through awkward local geometry, but a single Hamiltonian trajectory will not cross a deep
low-density barrier, so a lone HMC chain tends to stay in whichever mode it started in.

The differential evolution updates in this package have a complementary strength. They move
each chain using *differences* between members of the population, which lets them hop between
separated modes cheaply — but only when those modes look alike up to a shift, and in **high
dimensions** their *within*-mode mixing slows down. HMC is the reverse: with a well-chosen mass
matrix it explores a correlated, high-dimensional mode efficiently, but a single trajectory will
not cross the gap between separated modes. The hard case for either method alone is therefore a
**high-dimensional posterior with separated modes of different shape** — one shared mass matrix
cannot precondition them all at once.

This is exactly the situation where it pays to combine the two: run an HMC kernel on each
chain for gradient-guided local exploration, and a [`setup_subspace_sampling`](@ref)
(DREAM-z) update for population moves. When `AdvancedHMC.jl` is loaded this package exposes
[`setup_hmc_update`](@ref), which wraps an AdvancedHMC sampler so it can act as one update
inside a [`setup_sampler_scheme`](@ref) composite, sharing a single, population-pooled metric
across all of the chains — or, as we will see, a metric estimated separately for each mode.

## A high-dimensional bimodal target

We use a `D`-dimensional mixture of two Gaussians, **separated in location** so the population's
difference moves can jump between them, but with **opposite correlation structure**: the `+` mode is
an AR(1) covariance with `ρ = +0.9` (neighbouring coordinates positively correlated), the `−` mode
an AR(1) with `ρ = −0.9` (neighbouring coordinates anti-correlated). In every coordinate plane the
two modes are ellipses tilted in opposite directions, so a single dense mass matrix is a compromise
that fits neither — the situation [`per_cluster_metric`](@ref) is built for. The dimension is what
makes this bite: in high `D` the difference proposals mix slowly *within* a mode, so the
per-mode-preconditioned HMC kernel does the local work.

HMC needs gradients, so the log density must expose them. We write it against the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface and wrap it
with [LogDensityProblemsAD](https://github.com/tpapp/LogDensityProblemsAD.jl) so that
`ForwardDiff` supplies the gradient.

```@example HMC
using DifferentialEvolutionMetropolis, AbstractMCMC
using AdvancedHMC, ForwardDiff, LogDensityProblemsAD, LogDensityProblems
using Distributions, Random, LinearAlgebra

const D = 20                          # high enough that within-mode mixing needs HMC
const SEP = 4.0                       # the two modes are centred at ±SEP in every coordinate
const W_PLUS = 0.6                    # weight of the + mode

ar1(ρ) = Symmetric([ρ^abs(i - j) for i in 1:D, j in 1:D])
const Σ_PLUS  = ar1(0.9)              # + mode: positively correlated neighbours
const Σ_MINUS = ar1(-0.9)            # − mode: anti-correlated neighbours
const μ_PLUS  = fill(SEP, D)
const μ_MINUS = fill(-SEP, D)
const P_PLUS  = inv(Σ_PLUS);  const LD_PLUS  = logdet(Σ_PLUS)
const P_MINUS = inv(Σ_MINUS); const LD_MINUS = logdet(Σ_MINUS)

struct CorrMixture end
LogDensityProblems.dimension(::CorrMixture) = D
LogDensityProblems.capabilities(::Type{CorrMixture}) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(::CorrMixture, x)
    dp = x .- μ_PLUS; dm = x .- μ_MINUS
    lpP = log(W_PLUS)     - 0.5 * dot(dp, P_PLUS, dp)  - 0.5 * LD_PLUS
    lpM = log(1 - W_PLUS) - 0.5 * dot(dm, P_MINUS, dm) - 0.5 * LD_MINUS
    c = max(lpP, lpM)
    return c + log(exp(lpP - c) + exp(lpM - c))      # mixture of the two Gaussians
end

model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, CorrMixture()))
```

The modes are `2·SEP` apart in every coordinate, so difference-based jumps can bridge them, yet
their correlations are opposite, so no single mass matrix preconditions both — exactly the regime
where giving each mode its own metric pays off.

## Building the HMC update

[`setup_hmc_update`](@ref) takes any AdvancedHMC sampler and decomposes it into the modular
pieces (metric, integrator, kernel, adaptor) that the harness drives directly. The simplest
way to construct one is the `NUTS` wrapper. `NUTS`/`HMC`/`HMCDA` store their metric as a
*symbol* and only size it once the model is known, so here you must tell `setup_hmc_update`
the number of parameters via `n_dims`:

```@example HMC
hmc_update = setup_hmc_update(NUTS(0.8); n_dims = D)
nothing # hide
```

If you would rather assemble the kernel yourself, `setup_hmc_update` also accepts a
fully-built `HMCSampler`. Because that already carries a correctly-sized metric, `n_dims` is
not needed:

```@example HMC
metric = DiagEuclideanMetric(D)
integrator = Leapfrog(0.1)
κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

custom_hmc_update = setup_hmc_update(HMCSampler(κ, metric, adaptor))
nothing # hide
```

Both styles produce the same kind of update; the example below uses the `NUTS()` form.

### Estimating the metric from the memory archive

!!! warning "Experimental"
    The archive-based metric strategies described here — [`memory_metric`](@ref),
    [`cluster_pooled_metric`](@ref), and [`per_cluster_metric`](@ref) — are **experimental**; their
    behaviour and interface may change in future releases, and their effectiveness depends on the
    target's mode structure (see the example below).

By default the HMC update tunes its mass matrix the usual AdvancedHMC way: every chain's
position is fed into a windowed Welford estimator once per warm-up step, and the mass matrix is
committed at the end of each adaptation window (the estimator resetting between windows). That
already pools across the whole population, but it only ever sees the *current* window's draws.
Because this package also keeps a historical archive when `memory = true`, the metric can
instead be estimated from that much larger pool. Passing [`memory_metric`](@ref) as the
`metric_strategy` replaces the windowed adaptor with a covariance computed in one batch over the
memory archive (thousands of past positions), recomputed wholesale every `every` HMC steps
rather than accumulated online. Because the archive keeps growing, the metric is recomputed on
that cadence throughout **both warm-up and sampling**, so it never goes stale (changing the mass
matrix between trajectories is valid — each HMC step is correct for whatever metric it uses); the
step size adapts through dual averaging during warm-up and is then fixed.

Because it reads the archive, [`memory_metric`](@ref) **requires `memory = true`** and is
**incompatible with parallel tempering** (there the archive interleaves hot-chain positions, so
the covariance would mix temperatures); both raise an error on the first warm-up step. Annealing
is fine — its chains all cool to the cold target, so the archive ends up a clean cold-target
sample.

```@example HMC
mem_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = memory_metric()
)
nothing # hide
```

The estimate takes the *shape* of the metric the update was built with. With the default
`NUTS(0.8)` metric it is diagonal (per-coordinate variances). Asking for a dense metric instead
gives the **full covariance**, so the metric captures correlations between parameters — the main
reason to estimate from a large archive in the first place, since thousands of stored draws make
a full `D×D` covariance well-conditioned where a single chain's warm-up window would not. A dense
memory metric is built simply by handing `setup_hmc_update` a dense AdvancedHMC sampler:

```@example HMC
dense_mem_update = setup_hmc_update(
    NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = memory_metric()
)
nothing # hide
```

The dense metric is safe under `parallel = true`: each chain runs its trajectory against a
private-scratch copy of the metric, so the shared scratch buffer a `DenseEuclideanMetric` carries
cannot race. Two keywords tune either shape:

- `shrinkage` (λ ∈ [0, 1]) blends the estimate toward an isotropic target (a scaled identity),
  bounding the metric's condition number. This is worth raising when the population is small or
  some direction is nearly degenerate (e.g. a strongly correlated parameter pair), where a raw
  covariance can be noisy, extreme, or — for the dense metric — singular. A small ridge is always
  added so the metric stays positive-definite.
- `every` recomputes the covariance every `every` HMC steps (in both warm-up and sampling),
  trading freshness for the cost of the covariance pass.

```@example HMC
# A more conservative variant: shrink toward isotropy and recompute every 10 warm-up steps.
conservative = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = memory_metric(; shrinkage = 0.25, every = 10)
)
nothing # hide
```

### Multimodal targets: the cluster-pooled metric

The plain [`memory_metric`](@ref) breaks down when the posterior has well-separated modes: the
archive then spans every mode, so its covariance is dominated by the *between-mode* spread rather
than the shape of any individual mode, and the resulting metric fits none of them. [`cluster_pooled_metric`](@ref)
fixes this while still producing a single, position-independent metric. It clusters the archive,
subtracts each point's cluster mean, and pools the centred deviations into one covariance —
stripping the between-mode spread but keeping the (shared) within-mode shape.

```@example HMC
cluster_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = cluster_pooled_metric()
)
nothing # hide
```

It carries the same requirements as [`memory_metric`](@ref) (`memory = true`, no parallel
tempering) and takes the same `shrinkage`/`every` keywords, plus `kmax` to cap the cluster count
(chosen automatically up to that bound). The adjustment is gated on a cheap multimodality test, so
on a unimodal target it costs almost nothing and reduces exactly to [`memory_metric`](@ref); it is
worth reaching for only when modes are expected to separate.

### Per-mode metrics for modes of different shape

[`cluster_pooled_metric`](@ref) still pools the clusters into *one* metric, so it can only carry a
single shape — fine when the modes differ in *location* but share a shape, useless when they differ
in shape too. [`per_cluster_metric`](@ref) keeps the cluster covariances **separate**, one metric
per mode, and routes each chain to its current mode's metric (by nearest cluster centre to the
chain's position). Each mode then gets its own preconditioner.

```@example HMC
per_cluster_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = per_cluster_metric()
)
nothing # hide
```

The K metrics (and their centres) are recomputed only on the recompute cadence, but each chain is
relabelled to its nearest mode's metric *before every HMC step*, off its current position — so a
chain that has wandered into another mode immediately runs under that mode's shape. For
**well-separated** modes this label is constant throughout each basin, so the per-mode metric is the
optimal preconditioner and the choice is independent of the step's own trajectory (the setting it is
designed for). For **close or connected** modes, whose trajectories can bridge
clusters, picking the metric from the step's own starting position trades a small boundary bias for
always matching the chain's current mode. It carries the same requirements and
`shrinkage`/`every`/`kmax` keywords as [`cluster_pooled_metric`](@ref), and gates to one cluster
(= [`memory_metric`](@ref)) on a unimodal archive.

## Seeding the memory and sampling

These are the **z-variant** updates: with `memory = true` the difference vectors come from a
historical archive `Z`, so only a few chains are needed (we use 3). But `Z` must *contain* both
modes from the start — the default fill adds random points near the origin, which in high `D` sit far
from either mode and make the difference vectors useless. We seed `Z` with draws from both modes
(50/50) by passing them as extra `initial_position` entries beyond the 3 chains and setting `N₀` to
keep them.

Crossing two separated modes also needs a **full-dimensional jump**: a `γ = 1` subspace update with
`cr = 1` moves *every* coordinate by a full archive difference at once. (A partial-subspace jump
moves only some coordinates, landing a chain between the modes where the density is negligible, so it
is rejected.)

```@example HMC
# 3 starting chains followed by a memory archive seeded 50/50 across the two modes.
function seeded_positions(rng; n_mem = 60)
    Lp = cholesky(Σ_PLUS).L
    Lm = cholesky(Σ_MINUS).L
    plus  = [μ_PLUS  .+ Lp * randn(rng, D) for _ in 1:(n_mem ÷ 2)]
    minus = [μ_MINUS .+ Lm * randn(rng, D) for _ in 1:(n_mem ÷ 2)]
    chains = [plus[1], minus[1], plus[2]]               # the 3 chains span both modes
    return vcat(chains, plus[3:end], minus[2:end])      # the rest seed the memory
end

scheme = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = per_cluster_metric()),
    setup_subspace_sampling(),                  # local population moves
    setup_subspace_sampling(γ = 1.0, cr = 1.0); # full-dimensional mode jump
    w = [0.5, 0.3, 0.2],
)

out = sample(
    Random.Xoshiro(1), model, scheme, 5000;
    n_chains = 3, num_warmup = 2000,
    initial_position = seeded_positions(Random.Xoshiro(11)), N₀ = 60,
    memory = true, progress = false, chain_type = DifferentialEvolutionOutput,
)
nothing # hide
```

## What it recovers

Both modes are represented, and — the payoff of per-mode metrics — each mode's *distinct* correlation
is recovered, even though the two are opposite:

```@example HMC
using Statistics

draws = reshape(out.samples, :, D)
in_plus = vec(sum(draws; dims = 2)) .> 0

println("P(+ mode)            = ", round(mean(in_plus); digits = 2), "   (true ", W_PLUS, ")")
println("corr(x1, x2)  + mode = ", round(cor(draws[in_plus, 1],  draws[in_plus, 2]);  digits = 2), "   (true +0.9)")
println("corr(x1, x2)  − mode = ", round(cor(draws[.!in_plus, 1], draws[.!in_plus, 2]); digits = 2), "   (true −0.9)")
```

A scatter of the first two coordinates shows the two modes recovered with their opposite tilts:

```@example HMC
using Plots

scatter(draws[in_plus, 1], draws[in_plus, 2];
    label = "+ mode", ms = 2, alpha = 0.3, color = :steelblue)
scatter!(draws[.!in_plus, 1], draws[.!in_plus, 2];
    label = "− mode", ms = 2, alpha = 0.3, color = :firebrick)
plot!(xlabel = "x₁", ylabel = "x₂", title = "Two 20-D modes with opposite correlation")
```

Two honest points. First, recovering each mode's *shape* is where the per-mode metric earns its
keep: a single pooled metric would have to compromise between `+0.9` and `−0.9` and fit neither,
whereas `per_cluster_metric` gives each mode its own dense preconditioner and reproduces both
correlations to within a few percent. Second, the mode *weight* `P(+)` is the genuinely hard part:
even with the memory seeded across both modes and full-dimensional jumps, crossing between two
well-separated 20-dimensional modes is rare, so `P(+)` moves only slowly away from where the chains
began. That between-mode mixing is the difficult part of high-dimensional multimodality, and it is
largely independent of which metric you choose — the metric governs how efficiently each mode is
explored *once a chain is in it*.

The broader point: differential-evolution moves and HMC are complementary — the population's
difference proposals jump between separated modes, while HMC, with a per-mode metric, explores each
high-dimensional mode efficiently. The archive-based metric strategies are an experimental lever for
that second part: [`memory_metric`](@ref) for a single mode (or several alike up to a shift),
[`cluster_pooled_metric`](@ref) for modes that differ in *location* but share a shape, and
[`per_cluster_metric`](@ref) for modes that differ in *shape*, as here. As always, benchmark on your
own posterior.
