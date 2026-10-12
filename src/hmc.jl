"""
    setup_hmc_update(sampler; metric_strategy = ...)

Wrap an AdvancedHMC sampler (`NUTS`, `HMC`, `HMCDA`, or a hand-built `HMCSampler`) so it
can participate as one update inside a [`setup_sampler_scheme`](@ref) composite, running
the same HMC kernel on each chain in turn.

The implementation lives in the `AdvancedHMCExt` package extension; calling this requires
`AdvancedHMC.jl` to be loaded.
"""
function setup_hmc_update(args...; kwargs...)
    return error("`setup_hmc_update` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end

"""
    memory_metric(; shrinkage = 0.0, every = 100)

!!! warning "Experimental"
    Archive-based metric strategies ([`memory_metric`](@ref), [`cluster_pooled_metric`](@ref),
    [`per_cluster_metric`](@ref)) are experimental; their behaviour and interface may change in
    future releases.

Metric strategy for [`setup_hmc_update`](@ref) that estimates the HMC mass matrix from the
sampler's memory archive rather than from AdvancedHMC's windowed adaptor. The estimate is a
batch covariance over the history of sample positions, recomputed wholesale every `every`
HMC steps so it keeps tracking the growing archive throughout **both warm-up and sampling**
(recomputing the mass matrix between trajectories is valid: each HMC step is correct for
whatever metric it uses). The step size adapts through dual averaging during warm-up and is
then fixed.

Pass it through `setup_hmc_update`:

```julia
setup_hmc_update(NUTS(0.8); n_dims = d, metric_strategy = memory_metric())
```

This strategy requires the scheme to run **with memory** and is **incompatible with parallel
tempering** — the memory archive interleaves hot-chain positions there, so the covariance would
mix temperatures. Both cases raise an error on the first warmup step. Annealing *is* supported:
when well-specified its chains all cool, so the archive trends to a clean cold-target sample.

Keyword arguments:
- `shrinkage`: λ ∈ [0, 1]. Blends the estimate toward an isotropic target (`v̄·I`) to bound the
  metric's condition number; `0` keeps the raw sample covariance. A small ridge/floor is always
  applied so the metric stays positive-definite.
- `every`: recompute the covariance every `every` HMC steps (≥ 1), in both warm-up and
  sampling. More frequent recomputes track the archive more closely at the cost of more
  covariance passes.

The estimate matches the metric the update was built with: a diagonal metric (the
`NUTS`/`HMC`/`HMCDA` default) gets per-coordinate variances, while a dense metric
(`NUTS(0.8; metric = :dense)`) gets the full covariance, capturing correlations between
parameters. Requires `AdvancedHMC.jl` to be loaded.
"""
function memory_metric(args...; kwargs...)
    return error("`memory_metric` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end

"""
    cluster_pooled_metric(; shrinkage = 0.0, every = 100, kmax = 10)

!!! warning "Experimental"
    Archive-based metric strategies ([`memory_metric`](@ref), [`cluster_pooled_metric`](@ref),
    [`per_cluster_metric`](@ref)) are experimental; their behaviour and interface may change in
    future releases.

Metric strategy for [`setup_hmc_update`](@ref) that estimates the HMC mass matrix from the
*within-cluster-pooled* covariance of the memory archive. It is a multimodal-robust
refinement of [`memory_metric`](@ref): the archive is clustered, each point is centred on
its cluster mean, and the centred deviations are pooled into a single covariance. Removing
the per-cluster means strips the between-mode spread that otherwise inflates the raw memory
covariance for separated modes, while still producing one position-independent metric shared
by every chain.

Pass it through `setup_hmc_update`:

```julia
setup_hmc_update(NUTS(0.8); n_dims = d, metric_strategy = cluster_pooled_metric())
```

The adjustment is gated so it costs nothing on unimodal targets: a cheap per-axis
multimodality test runs first, and clustering only happens when an axis looks multimodal and
the between-mode variance is a material fraction of the total. When neither fires the metric
is exactly [`memory_metric`](@ref)'s raw memory covariance, so this strategy weakly dominates
it — equal when unimodal, better when modes separate.

It carries the same requirements as [`memory_metric`](@ref): the scheme must run **with
memory** and is **incompatible with parallel tempering** (pure annealing is fine). Both are
checked on the first warmup step.

Keyword arguments:
- `shrinkage`: λ ∈ [0, 1], blended into the pooled estimate exactly as in [`memory_metric`](@ref).
- `every`: recompute the metric (and re-cluster) every `every` HMC steps (≥ 1), in both
  warm-up and sampling, exactly as in [`memory_metric`](@ref).
- `kmax`: upper bound on the number of clusters (≥ 1). The cluster count is chosen up to this
  cap by growing it while each added cluster explains a worthwhile share of the variance, so a
  too-large `kmax` only costs a little extra compute.

The estimate matches the metric the update was built with (diagonal or dense), as for
[`memory_metric`](@ref). Requires `AdvancedHMC.jl` to be loaded.
"""
function cluster_pooled_metric(args...; kwargs...)
    return error("`cluster_pooled_metric` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end

"""
    per_cluster_metric(; shrinkage = 0.0, every = 100, kmax = 10)

!!! warning "Experimental"
    Archive-based metric strategies ([`memory_metric`](@ref), [`cluster_pooled_metric`](@ref),
    [`per_cluster_metric`](@ref)) are experimental; their behaviour and interface may change in
    future releases.

Metric strategy for [`setup_hmc_update`](@ref) that gives each posterior **mode its own** HMC
mass matrix. It is the heteroscedastic refinement of [`cluster_pooled_metric`](@ref): where the
pooled strategy clusters the memory archive and collapses the per-cluster covariances into one
shared metric (correct only when the modes share a shape), this one keeps the cluster covariances
**separate** — one metric per cluster — and routes each chain to its current mode's metric. It is
the right choice for **separated** modes of genuinely different shape, which a single pooled metric
cannot fit at once.

Pass it through `setup_hmc_update`:

```julia
setup_hmc_update(NUTS(0.8); n_dims = d, metric_strategy = per_cluster_metric())
```

The K cluster metrics (and their centres) are recomputed only on the recompute cadence (every
`every` steps), but each chain is **relabelled to its nearest mode's metric before every HMC
step**, off its current position — so a chain that crosses into another mode immediately runs under
that mode's shape. For **separated** modes the nearest centre is constant throughout each basin, so
the per-mode metric is the optimal preconditioner and the choice does not depend on the step's own
trajectory; for **close** modes whose trajectories can bridge clusters, choosing the metric from
the step's own starting position introduces a small boundary bias in exchange for always matching
the chain's current mode. Reparameterising the modes to a common shape (so
[`cluster_pooled_metric`](@ref) suffices) is preferable when it is available.

It carries the same requirements as [`memory_metric`](@ref) (`memory = true`, no parallel
tempering; pure annealing is fine), checked on the first warmup step. The adjustment is gated like
[`cluster_pooled_metric`](@ref): when the archive looks unimodal it reduces to one cluster and the
metric is exactly [`memory_metric`](@ref)'s.

Keyword arguments `shrinkage`, `every`, and `kmax` behave exactly as in
[`cluster_pooled_metric`](@ref). The estimate matches the metric the update was built with
(diagonal or dense). Requires `AdvancedHMC.jl` to be loaded.
"""
function per_cluster_metric(args...; kwargs...)
    return error("`per_cluster_metric` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end
