# Combining HMC with differential evolution

Hamiltonian Monte Carlo (HMC, and its self-tuning variant NUTS) uses gradients to glide
through awkward local geometry, but a single Hamiltonian trajectory will not cross a deep
low-density barrier, so a lone HMC chain tends to stay in whichever mode it started in.

The differential evolution updates in this package have a complementary strength. They move
each chain using *differences* between members of the population, which lets them hop between
modes cheaply — but only when those modes look alike up to a shift. The hard case for either
method alone is a posterior whose modes are **separated** (so a lone HMC chain is stuck in
whichever it started in) yet **differently scaled** (so a difference vector drawn from one mode
is the wrong size for another, and a single mass matrix cannot precondition both).

This is exactly the situation where it pays to combine the two: run an HMC kernel on each
chain for gradient-guided local exploration, and a [`setup_subspace_sampling`](@ref)
(DREAM-z) update for population moves. When `AdvancedHMC.jl` is loaded this package exposes
[`setup_hmc_update`](@ref), which wraps an AdvancedHMC sampler so it can act as one update
inside a [`setup_sampler_scheme`](@ref) composite, sharing a single, population-pooled metric
across all of the chains — or, as we will see, a metric estimated separately for each mode.

## A two-mode funnel mixture

We use a **two-component mixture of Neal's funnels**. Each component is a funnel — a log-scale
parameter `v` and `D-1` coordinates whose standard deviation is `exp(v/2)` — but the two
components are *separated*: one sits at `v ≈ +A` with its coordinates centred at `+L`, the other
at `v ≈ -A` centred at `-L`. The two modes are thus alike up to a **shift** (so the
difference-based population moves can hop between them) yet have **different scales** (`exp(±A/2)`,
a factor of `exp(A) ≈ 9` apart), so no single mass matrix preconditions both at once. Because we
know the mixture exactly, the marginal posterior of `v` — a two-component Gaussian mixture with
`P(v > 0) = W_PLUS` — is a known quantity we can hold each sampler up against.

HMC needs gradients, so the log density must expose them. We write it against the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface and wrap it
with [LogDensityProblemsAD](https://github.com/tpapp/LogDensityProblemsAD.jl) so that
`ForwardDiff` supplies the gradient.

```@example HMC
using DifferentialEvolutionMetropolis, AbstractMCMC
using AdvancedHMC, ForwardDiff, LogDensityProblemsAD, LogDensityProblems
using Distributions, Random

const D = 5            # one log-scale parameter v plus D-1 funnel coordinates
const A = 2.2          # the two modes sit at v ≈ ±A (scale ratio exp(A) ≈ 9)
const SV = 0.5         # width of the v marginal within a mode
const L = 4.0          # the two modes are shifted to ±L in every coordinate
const W_PLUS = 0.6     # weight of the + mode (v ≈ +A)

struct FunnelMixture end
LogDensityProblems.dimension(::FunnelMixture) = D
LogDensityProblems.capabilities(::Type{FunnelMixture}) = LogDensityProblems.LogDensityOrder{0}()
# Log density of one funnel mode, shifted to ±A in v and ±L in every other coordinate.
function mode_logdensity(x, sgn)
    v = x[1]
    lp = logpdf(Normal(sgn * A, SV), v)             # this mode's scale mouth
    s = exp(v / 2)
    @inbounds for i in 2:D
        lp += logpdf(Normal(sgn * L, s), x[i])      # coordinates centred at ±L, width exp(v/2)
    end
    return lp
end
function LogDensityProblems.logdensity(::FunnelMixture, x)
    lpR = log(W_PLUS)     + mode_logdensity(x, +1)
    lpL = log(1 - W_PLUS) + mode_logdensity(x, -1)
    m = max(lpR, lpL)
    return m + log(exp(lpR - m) + exp(lpL - m))     # mixture of the two funnels
end

model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, FunnelMixture()))
```

The two modes differ by a factor of `exp(A) ≈ 9` in scale but only by a *shift* in location, so a
difference vector drawn from the population can carry a chain from one to the other — while no
single mass matrix fits both scales. That is exactly the regime where giving each mode its own
metric can pay off.

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

Both styles produce the same kind of update; the comparison below uses the `NUTS()` form.

### Estimating the metric from the memory archive

!!! warning "Experimental"
    The archive-based metric strategies described here — [`memory_metric`](@ref),
    [`cluster_pooled_metric`](@ref), and [`per_cluster_metric`](@ref) — are **experimental**; their
    behaviour and interface may change in future releases, and their effectiveness depends on the
    target's mode structure (see the comparison below).

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
  some direction is nearly degenerate (e.g. a tight funnel neck), where a raw covariance can be
  noisy, extreme, or — for the dense metric — singular. A small ridge is always added so the
  metric stays positive-definite.
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

## The samplers

Two ingredients matter for the comparison. First, because `memory = true` makes these the
**z-variant** updates — difference vectors are drawn from the historical archive `Z`, not the live
population — we run only **3 chains**; the archive supplies the diversity, so a large population is
unnecessary. Second, crossing between two separated modes needs a full-size jump, so each scheme
pairs the ordinary adaptive-`γ` subspace update with a **separate `γ = 1` subspace update** (the
move the package's own [`DREAMz`](@ref) template uses for mode hopping). Without it the chains never
leave their starting mode.

Against that fixed population/jump setup we vary only the HMC metric: the bare HMC update, the
subspace updates on their own (no HMC), the composite of the two, and three composites whose HMC
kernel estimates its metric from the archive — [`memory_metric`](@ref),
[`cluster_pooled_metric`](@ref), and [`per_cluster_metric`](@ref). All use `memory = true` and no
tempering, which the archive strategies require. All are ordinary [`setup_sampler_scheme`](@ref)
objects.

```@example HMC
# Population moves: adaptive-γ subspace update + a separate γ = 1 update for full mode-jumps.
subspace_moves() = (setup_subspace_sampling(), setup_subspace_sampling(γ = 1.0))

nuts_only = setup_sampler_scheme(setup_hmc_update(NUTS(0.8); n_dims = D))
subspace_only = setup_sampler_scheme(subspace_moves()...; w = [0.6, 0.4])
composite = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8); n_dims = D), subspace_moves()...; w = [0.5, 0.3, 0.2]
)
mem_composite = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = memory_metric()),
    subspace_moves()...; w = [0.5, 0.3, 0.2]
)
cluster_composite = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = cluster_pooled_metric()),
    subspace_moves()...; w = [0.5, 0.3, 0.2]
)
per_cluster_composite = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = per_cluster_metric()),
    subspace_moves()...; w = [0.5, 0.3, 0.2]
)
nothing # hide
```

We seed both modes (two chains in the `+A` mode, one in the `−A` mode) so the archive holds both
from the start; the question is whether each sampler then keeps the population *shuttling* between
them at the right balance, rather than letting chains lock into one mode.

```@example HMC
const N_CHAINS = 3

function init_positions()
    [vcat(s * A, s * L .* ones(D - 1)) for s in (1.0, -1.0, 1.0)]
end
nothing # hide
```

## Recovering the scale parameter

For each scheme we report the recovered weight of the `+A` mode, `P(v > 0)` (true value `0.6`).
Estimating a mode probability from three chains is intrinsically a high-variance Monte Carlo
problem — it depends on how often the chains cross between the modes during a run — so we report the
mean and the run-to-run standard deviation over several seeds, and accumulate the `v` draws for the
plot below.

```@example HMC
using Statistics

const SEEDS = 1:6

function weight_estimate(scheme; iters = 8000, warm = 2000, seeds = SEEDS)
    vs = Float64[]
    weights = Float64[]
    for seed in seeds
        out = sample(
            Random.Xoshiro(seed), model, scheme, iters;
            n_chains = N_CHAINS, num_warmup = warm, initial_position = init_positions(),
            memory = true, progress = false, chain_type = DifferentialEvolutionOutput
        )
        v = out.samples[:, :, 1]
        append!(vs, vec(v))
        push!(weights, mean(v .> 0))                 # truth is W_PLUS = 0.6
    end
    return (; v = vs, p = mean(weights), sd = std(weights))
end

results = (
    NUTS = weight_estimate(nuts_only),
    subspace = weight_estimate(subspace_only),
    composite = weight_estimate(composite),
    memory = weight_estimate(mem_composite),
    cluster = weight_estimate(cluster_composite),
    per_cluster = weight_estimate(per_cluster_composite),
)

println("true P(v>0) = ", W_PLUS, "   (mean ± sd over ", length(SEEDS), " seeds)\n")
println(rpad("sampler", 13), rpad("P(v>0)", 11), "sd")
for name in (:NUTS, :subspace, :composite, :memory, :cluster, :per_cluster)
    r = results[name]
    println(rpad(string(name), 13), rpad(round(r.p; digits = 3), 11), round(r.sd; digits = 3))
end
```

The weight only tells you where the population *ended up*; the per-chain occupancy tells you whether
the chains are actually *crossing*. For one run we report, for each chain, the fraction of its draws
in the `+A` mode — a value near `0` or `1` means that chain never left a mode, while intermediate
values mean it shuttled between them.

```@example HMC
function occupancy(scheme; iters = 8000, warm = 2000, seed = 2)
    out = sample(
        Random.Xoshiro(seed), model, scheme, iters;
        n_chains = N_CHAINS, num_warmup = warm, initial_position = init_positions(),
        memory = true, progress = false, chain_type = DifferentialEvolutionOutput
    )
    return round.(vec(mean(out.samples[:, :, 1] .> 0; dims = 1)); digits = 2)
end

println("per-chain fraction in the +A mode (one run):")
for (name, scheme) in (("composite", composite), ("memory", mem_composite),
                       ("cluster", cluster_composite), ("per_cluster", per_cluster_composite))
    println("  ", rpad(name, 13), occupancy(scheme))
end
```

Two things stand out. First, the population moves do the heavy lifting. **NUTS alone** cannot cross
between the modes at all: each of its three chains stays in whichever mode it started in, so
`P(v > 0)` simply echoes the initial split and never moves (its per-chain occupancies are all `0` or
`1`). **subspace alone** has the jumps but no gradient-guided local exploration, and it swings
wildly — routinely collapsing onto a single mode (a low mean with a large spread).

Second, the composites all share the same HMC + subspace + `γ = 1` machinery and differ *only* in
the HMC metric. On the recovered *weight* they are hard to separate: all four land near the true
`0.6`, but with a large run-to-run spread (the `sd` column), and at three chains their differences
sit well inside that spread — estimating a mode probability from three chains is inherently noisy.
The clearer signal is the per-chain occupancy, which isolates the metric's effect on mixing: under
**`per_cluster_metric`** more of the chains spend time in *both* modes, because a chain is always
preconditioned at the scale of whichever mode it currently occupies, so its local HMC steps stay
well-sized and it is less likely to be stranded after a jump. Under a single shared metric
(`composite`, `memory`, `cluster`) the metric is a compromise between the two scales, and chains
more often lock into one mode for the whole run.

So the metric is a secondary lever here: it is the population moves and the `γ = 1` jump — not the
choice of mass matrix — that rescue the weight from the baselines' failure, and the metric only
nudges how freely the chains then cross. The whole effect also lives within a limited regime: place
the modes far enough apart that the `γ = 1` jumps stop landing in the opposite mode and the
population collapses regardless of metric, because the binding constraint becomes the jump rather
than the within-mode preconditioning.

## Comparing the scale-parameter posteriors

Plotting the recovered marginal of `v` (pooled across the seeds) against the truth makes the
baseline failures plain: NUTS is frozen at its initial split, and subspace has collapsed onto a
single mode. The composite schemes all recover *both* modes near the true balance; they are close
enough to one another that the differences between them sit within the run-to-run spread.

```@example HMC
using Plots

vgrid = range(-5, 5; length = 400)
true_density = @. W_PLUS * pdf(Normal(A, SV), vgrid) + (1 - W_PLUS) * pdf(Normal(-A, SV), vgrid)

plt = plot(
    vgrid, true_density;
    lw = 3, color = :black, label = "truth",
    xlabel = "v  (log scale)", ylabel = "density",
    title = "Posterior of the scale parameter",
)
for (name, c) in zip(
        (:NUTS, :subspace, :composite, :memory, :cluster, :per_cluster),
        (:orange, :red, :green, :purple, :blue, :brown),
    )
    stephist!(plt, results[name].v; normalize = :pdf, lw = 2, color = c, label = string(name))
end
plt
```

The broader lesson is about matching the sampler — and the metric — to the geometry. Differential
evolution moves (here a small population of `z`-variant chains drawing from the archive, with a
`γ = 1` jump for mode hopping) excel when modes are alike up to a shift, and gradient information is
what lets a sampler follow a scale that changes across the space; the HMC update brings both to bear
in one scheme. Estimating the metric from the memory archive is a further (experimental) lever, with
a ladder of strategies: `memory_metric` pools everything (best on unimodal targets, or several modes
alike up to a shift), `cluster_pooled_metric` strips the between-mode spread (for modes that differ
in *location* but share a shape), and `per_cluster_metric` gives each mode its own geometry (for
modes that are *separated* and differ in *shape*, as here). The benefit holds only while the
population can still move between the modes; once they are separated far enough that hopping fails,
every strategy collapses together. We recommend benchmarking on your own posterior: gradient
evaluations carry a cost, on smoother targets the population moves may already suffice, these
archive-metric strategies remain experimental, and the most effective one depends on whether your
archive samples a single mode, several modes alike up to a shift, or several separated modes of
different shape.
