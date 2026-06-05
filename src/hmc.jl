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
batch covariance over history of sample positions, recomputed wholesale every
`every` warmup steps; the step size keeps adapting through dual averaging.

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
- `every`: recompute the covariance every `every` warmup steps (≥ 1). The step size still
  adapts on every step. More re-computations means more compute in the warm-up but more quickly tunes the covariance.

The estimate matches the metric the update was built with: a diagonal metric (the
`NUTS`/`HMC`/`HMCDA` default) gets per-coordinate variances, while a dense metric
(`NUTS(0.8; metric = :dense)`) gets the full covariance, capturing correlations between
parameters. Requires `AdvancedHMC.jl` to be loaded.
"""
function memory_metric(args...; kwargs...)
    return error("`memory_metric` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end
