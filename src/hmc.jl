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
