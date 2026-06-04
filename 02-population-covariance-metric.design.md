# Stage 2 — Population/memory covariance as the metric

## Goal

Add a metric strategy that estimates the mass matrix directly from the sampler's
**population** — and, when the DE sampler is memory-based, from the **memory archive**
(1000s of stored positions) rather than the live chain positions. Keep dual-averaging for
the step size ε.

This is a new `AbstractDifferentialEvolutionMetricStrategy`, dispatched through the exact
extension points Stage 1 already established. It must not touch `run_trajectory!` /
`run_trajectories!`, the RNG-seeding, the per-worker model copies, lazy init, or the
warmup/sampling step methods. The only new code is a strategy type plus an `adapt_metric!`
method (and a couple of small helpers it calls).

## Prerequisites

- Stage 1 (`AdvancedHMCExt`) complete and passing. Concretely you are extending:
  - `abstract type AbstractDifferentialEvolutionMetricStrategy`
  - `adapt_metric!(strategy, model_wrapper, state, astate, α)` — the dispatch seam
  - `mutable struct HMCAdaptiveState{T}` with fields `metric, integrator, κ, adaptor, initialized`
  - the trajectory loop, which reads `state.x[i]`, writes `state.xₚ[i]`/`state.ldₚ[i]`,
    and uses `state.rngs[i]` / `state.chain_models[i]`.
- A host accessor for the population/memory. **Confirm** how the archive is exposed on
  `state` (e.g. a memory buffer of past accepted positions) and how to tell whether memory
  is enabled. When memory is enabled, the archive is the informative set; otherwise the
  informative set is the live positions `state.x` (typically ~3) and the estimate will be
  poor — that is expected, see fallback below.

## Data-shape note (differs from earlier drafts)

Positions in this package are a **vector of per-chain vectors** (`state.x[i]` is chain `i`'s
position vector), not a `D×N` matrix. Any covariance routine must assemble a `D×M`
(or `D×N`) view/copy from that representation (and from the archive's representation),
rather than assuming a matrix is already in hand. Keep this conversion in one helper.

## Non-goals

- No clustering (Stage 3 adds within-cluster centering; Stage 4 adds per-cluster metrics).
- No per-chain or per-cluster metric. Stage 2 installs **one** shared metric into
  `astate.metric`, exactly as the stock strategy leaves one shared metric — only its
  *source* changes.

## Key design decisions (settled)

### The strategy type and the dispatch seam

```julia
"""Estimate the metric (M⁻¹ = Σ) from the population/memory covariance."""
struct DifferentialEvolutionPopulationMetric <: AbstractDifferentialEvolutionMetricStrategy end
```

Stage 1 routes adaptation through `adapt_metric!(sampler.metric_strategy, model_wrapper,
state, astate, α)` inside `step_warmup`. Stage 2 supplies a new method on that strategy:

```julia
function adapt_metric!(
        ::DifferentialEvolutionPopulationMetric, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    # 1. ε: keep the stock dual-averaging update so the step size still tracks the metric.
    # 2. metric: rebuild astate.metric from the population/memory covariance.
end
```

Everything upstream (the threaded trajectory loop, the freeze via `fix_sampler`, the
`step_warmup`→`step` transition) is inherited untouched.

### Step size still comes from dual averaging — keep `astate.adaptor.ssa` alive

Do **not** drop the adaptor wholesale. The stock strategy gets ε from the
`StanHMCAdaptor`'s `ssa` (Nesterov dual averaging), which Stage 1 re-anchors in
`refine_step_size!` (`ssa.state.ϵ = ϵ; reset!(ssa)`) and advances via `pooled_adapt!`.
Stage 2 should still drive that **step-size** update each warmup step from the pooled
acceptance rate, because ε must keep tracking the metric you are now setting yourself.

What Stage 2 stops using is the **mass-matrix** half of the adaptor (`adaptor.pc` and its
windowed schedule): you are estimating Σ from the population instead. Two clean options —
pick one and document it:

- **(A) Keep `StanHMCAdaptor`, feed only the step size.** Each warmup step, do the ε part
  of `pooled_adapt!` (the single `adapt!(adaptor.ssa, …, mα)` plus the window
  `reset!(adaptor.ssa)` bookkeeping) and skip the `adaptor.pc` pushes entirely. Then set
  `astate.metric` from the population covariance and push ε into the integrator/κ with
  `astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)` (this pulls ε from `ssa`).
  Lowest-risk: reuses Stage 1's proven ε plumbing verbatim, only the `pc` pushes are gone.
- **(B) Replace the adaptor with a bare `StepSizeAdaptor`.** Cleaner conceptually, but
  changes `setup_hmc_update`/`make_adaptor` and the `refine_step_size!` assumptions, so it
  touches Stage-1 code. Prefer (A) unless (B) buys something concrete.

Recommend **(A)**.

### Memory reframes "population" — and is why the mass-matrix adaptor is droppable

With memory enabled, the metric is a **batch covariance over the archive**, not a running
window: thousands of within-mode draws give a well-conditioned estimate immediately. That
is the justification for not using `adaptor.pc` — the archive is a strictly better,
instantaneous covariance estimator than the single-chain-style Welford window the stock
strategy runs. (When memory is off, you fall back to the live positions; see below.)

### M⁻¹ = Σ (covariance), NOT its inverse — and match the existing metric type

HMC draws momentum from `N(0, M)`; the inverse metric equals the position covariance.
AdvancedHMC's Euclidean metrics store **M⁻¹** and its factor:

- `DiagEuclideanMetric(diag_Σ)` — pass the per-coordinate variances (the diagonal of Σ).
- `DenseEuclideanMetric(Σ)` — pass the covariance matrix Σ.

Pass **Σ** (the covariance), never `inv(Σ)`. **Verify against the installed version** what
each constructor expects and that the field you read in Stage 1's
`refresh_pieces_from_adaptor!` (`.metric`) is consistent with what you now build. Construct
the new metric to the **same concrete type** Stage 1 sized in `setup_hmc_update`
(`make_metric(sampler.metric, T, n_dims)`), so dispatch and sizes stay consistent.

### THREADING HAZARD — diagonal is safe to share, dense is NOT (carried from Stage 1)

Stage 1's `run_trajectory!` documents this explicitly: the shared metric is read-only
across threads for the **diagonal and unit** metrics (only `M⁻¹`/`sqrtM⁻¹` are read), but
`DenseEuclideanMetric` carries a `_temp` scratch buffer that **would race if shared across
the threaded path**. Stage 2 is the first strategy that can produce a *dense* metric, so
this is now load-bearing, not hypothetical. Resolve it before shipping dense:

- **Default to a diagonal metric.** `DiagEuclideanMetric` from `diag(Σ)` is race-free on
  the threaded path, robust, and the right default. Ship this first.
- **If you offer dense**, you must remove the shared-`_temp` race. Options: give each
  chunk/worker its own metric copy (mirror `state.chain_models`), or confirm whether the
  installed AdvancedHMC version made dense-metric momentum generation allocate locally. Do
  **not** share one `DenseEuclideanMetric` across `Threads.@threads` — that reintroduces
  exactly the data race Stage 1's per-worker model copies were added to prevent.
- The serial path (`parallel = false`) shares one metric already and is unaffected.

Make diagonal-vs-dense a field on the strategy (`DifferentialEvolutionPopulationMetric`),
defaulting to diagonal, so the dense path is opt-in and clearly gated on the threading fix.

### Regularize / shrink before constructing the metric

A raw archive covariance can be ill-conditioned (near-degenerate directions; dense-in-high
D). Shrink toward a diagonal target before building the metric — Ledoit–Wolf, or a fixed
`λ·Σ + (1−λ)·diag(Σ)`. Cheap insurance against an unstable integrator. For the diagonal
metric this reduces to flooring small variances; for dense it conditions the matrix. Make
the shrinkage intensity configurable on the strategy.

### Estimation cadence and the step-boundary invariant

Mode/scale geometry moves slowly, so you need not recompute Σ every warmup step — a
schedule is fine (and cheaper with a large archive). But the **step-boundary invariant**
from Stage 1 is mandatory: `astate.metric` is read-only inside the threaded trajectory loop
and may only be replaced **between** steps (in `adapt_metric!`, which Stage 1 calls after
`run_trajectories!` returns). Never swap the metric mid-trajectory or inside the
`@threads` region. The implicit barrier at the end of `run_trajectories!` guarantees the
next step sees the new metric. Freezing at the warmup→sampling boundary is already handled
by Stage 1's `fix_sampler` (it snapshots `astate.metric`/`κ` into the static sampler).

## Implementation sketch

```julia
struct DifferentialEvolutionPopulationMetric{T} <: AbstractDifferentialEvolutionMetricStrategy
    dense::Bool          # default false; true requires the dense threading fix above
    shrinkage::T         # λ for diag-target shrinkage
    every::Int           # recompute Σ every `every` warmup steps (>=1)
end

function adapt_metric!(
        ms::DifferentialEvolutionPopulationMetric, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    # --- step size: reuse Stage 1's dual-averaging, minus the mass-matrix pushes ---
    step_size_only_adapt!(astate.adaptor, α)          # ssa update + window reset bookkeeping
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)  # pulls ε into integrator/κ
    astate.integrator = astate.κ.τ.integrator

    # --- metric: from population/memory covariance, on schedule ---
    if due_this_step(astate, ms.every)
        P = population_or_memory_columns(state)        # D×M (memory) or D×N (live)
        Σ = cov_columns(P)                             # columns are samples
        Σ = shrink(Σ, ms.shrinkage)                    # diagonal-target shrinkage
        astate.metric = ms.dense ? make_dense(Σ) : make_diag(diag(Σ))  # M⁻¹ = Σ, NOT inv
    end
    return nothing
end
```

`population_or_memory_columns` is the single place that (a) prefers the archive when memory
is enabled, (b) falls back to `state.x` otherwise, and (c) converts the vector-of-vectors /
archive representation into a `D×M` array.

## Validation (must pass before Stage 3)

The metric-quality ladder is the oracle. **Stage 2 must be no worse than the Stage-1 stock
strategy on unimodal targets**, ideally better:

1. **Unimodal Gaussian (D = 10, plus an anisotropic/correlated variant).** Stage 2's metric
   must roughly match Stage 1's adapted metric (both estimate the same Σ). ESS and
   acceptance at least as good as Stage 1. **If Stage 2 is worse on a unimodal Gaussian,
   the bug is in the Σ estimation, the shrinkage, or the covariance-vs-inverse / metric-type
   handling** — catch it here, before clustering enters.
2. **Convergence speed.** With memory enabled, Stage 2 should reach a good metric in fewer
   HMC warmup steps than the stock window (the archive supplies more samples). Measure
   warmup-to-good-metric.
3. **Threading equivalence + race check.** Same seed, 1 thread vs many, with the
   **diagonal** metric: statistically identical, no races. If you implement dense, repeat
   the check for dense and confirm the per-worker-metric fix actually removed the `_temp`
   race (run repeatedly / under a sanitizer).
4. **Ill-conditioning.** A target with a near-degenerate direction must not destabilize the
   integrator — confirm shrinkage prevents divergence spikes.
5. **Memory-off fallback.** With memory disabled (Σ from live positions only), Stage 2 must
   still run without crashing; the estimate will be poor/degenerate (expected — memory is
   the intended mode), so shrink hard and prefer the diagonal metric there.

## Known limitations / carried to Stage 3

- **Raw pooled Σ is wrong for separated multimodality.** `Σ_total = Σ_within + Σ_between`;
  for modes separated by Δ on an axis with weights `p, 1−p`, the between term contributes
  `~p(1−p)Δ²`, so when `Δ ≫ within-mode sd`, Σ is dominated by separation and the metric
  over-scales momentum along separating axes (divergences, or ε driven so low everything
  else under-steps). Stage 3 removes this between-mode inflation while still producing one
  shared metric. Stage 2 is correct only when modes overlap/are close or the target is
  unimodal.
- One shared ε from dual averaging (unchanged from Stage 1). Per-cluster ε remains a
  Stage-4 deferred decision.
- Dense metric on the threaded path is gated on the `_temp`-race fix; until then, diagonal
  is the only threaded-safe population metric.
