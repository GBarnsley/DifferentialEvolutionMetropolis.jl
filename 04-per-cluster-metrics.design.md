# Stage 4 — Per-cluster metrics (per-mode geometry)

## Goal

Give each posterior **mode** its own metric, for genuinely different-shaped
(heteroscedastic) modes that a single within-cluster-pooled metric (Stage 3) cannot fit at
once. This reuses Stage 3's clustering wholesale; the new part is keeping the K cluster
covariances **separate** (one metric per cluster) and routing each chain to **its current
mode's** metric through the existing `chain_metrics` slots.

This is the last stage, it is **optional**, and it is correct only in a specific regime.
Read the next two sections before anything else.

## Why the label must track position (a frozen-label design is broken)

The label means "which mode is this chain in." In this sampler, **DE's whole job is to move
chains between modes** — it fires ~15% of the time precisely to relocate chains across the
posterior. So any label fixed at warmup end describes where the chain was *at that instant*
and is wrong within a few DE moves. A frozen-label scheme therefore runs HMC on a chain that
is physically in mode B under mode A's metric, and — because relocation is what the sampler
*does* — that mismatch is the **common case**, not an edge case. It is π-invariant (HMC under
any fixed metric is) but delivers anti-geometry: mode A's tight covariance applied to a chain
now in broad mode B. That is **strictly worse than Stage 3**, whose single pooled metric is at
least a sensible cross-mode compromise. Do not implement frozen labels. The assignment must be
a function of the chain's **current position**.

## What is unbiased, what is merely suboptimal (read carefully)

A **fixed** metric makes an HMC step π-invariant *regardless of where the trajectory goes*.
Momentum is drawn from `N(0, M_k)`, the leapfrog runs under `M_k`, and the acceptance uses the
`M_k` Hamiltonian — one fixed matrix throughout. That is a standard MH step targeting
`π(θ)·N(p; 0, M_k)`, whose θ-marginal is π. So a chain carrying `M_j` whose trajectory bridges
into a neighbouring cluster is **suboptimal, not biased**: the mis-scaled metric just lowers
acceptance / mixing on that step. This is the key fact and it is *not* a detailed-balance
problem.

The **only** way bias enters is selecting a step's metric as a function of *that step's own
current starting position* — i.e. recomputing the assignment every HMC step. We do not do that.
Assignment changes on a slow cadence (see below), held fixed across the HMC steps in between, so
every HMC step is a fixed-metric, unbiased step.

Two operations, kept distinct:

- **Reclustering** — recomputing the K centres and the K metrics from the archive — happens every
  `every` steps. It changes the *set* of metrics, never which one a given step is selected by
  mid-flight.
- **Reassignment** — which of the K fixed metrics a chain carries — changes on the assignment
  cadence (e.g. on DE moves), *not* every HMC step. DE never reads the label, so DE's detailed
  balance is untouched; and because the label is then constant across the following HMC steps,
  those steps are fixed-metric and unbiased.

### Exactness for separated modes; a second-order edge for close modes

For **well-separated** modes the assignment is constant over every region a trajectory can reach
(`nearest_center` is constant across a basin, and a trajectory under a reasonable `M_j` stays in
basin *j*). So `M(θ_start) = M(θ_end)` and the scheme is **exact**, and the per-mode metric is the
*optimal* preconditioner — this is the regime Stage 4 is for.

For **close** modes a single trajectory can bridge two clusters. The dominant effect is the
**inefficiency** above (a fixed metric that fits neither bridged region well). There is a narrow
second-order point worth being explicit about: the reassignment reads current position, and at
`p(HMC) ≈ 0.85` the step right after a reassignment usually starts from that same position, so its
metric is `M_{nearest(θ_start)}` — correlated with its own start. For separated modes this is
harmless (start and end share a basin). For close modes, on a boundary-crossing trajectory, that
correlation is the one channel by which a small bias could enter; it is dominated by the
inefficiency and vanishes as separation grows. If you want it provably gone, the assignment for a
step must not be a function of that step's start position (e.g. reassign off a position the step
does not then start from) — but for the separated regime this is moot, since there is no crossing
to correct. There is no cheap exact treatment of genuinely close, differently-shaped modes:
RHMC is exact but position-dependent everywhere and far more expensive; a full MH correction needs
the reverse-proposal density, intractable for NUTS; **reparameterization** is the only thing that
dissolves it.

## Reassignment cadence (on DE moves, not every HMC step)

Reassignment is the slow operation: a chain's carried label changes when DE relocates it, and
stays fixed across the HMC steps in between. This is what keeps every HMC step a fixed-metric,
unbiased step (the previous section). Implementing "on DE moves" cleanly needs a hook the HMC
extension does not currently have (it does not observe DE moves); absent that hook, the fallback
is to refresh the carried label on the existing `metric_steps` cadence (every `every` steps, the
same cadence as reclustering) rather than every HMC step. Do **not** reassign every HMC step from
current position — that is the one thing that turns the position-dependence into a real bias on
close modes. For separated modes the timing is immaterial: `nearest_center` is constant within a
basin, so the label a chain carries equals its basin no matter when it was last set.

## Within-step rule (the one invariant to enforce)

Use the chain's **carried** metric for its whole trajectory (momentum draw + leapfrog +
acceptance); do **not** recompute the metric mid-trajectory and do **not** re-derive it from the
step's current position. Within-trajectory constancy is what makes each step a fixed-metric,
π-invariant proposal — and that holds *regardless of where the trajectory goes*. If the trajectory
bridges into a neighbouring cluster the step is suboptimal (poor acceptance), never biased. The
carried label is refreshed only on the reassignment cadence above, never inside the trajectory and
never per HMC step from current position.

## Prerequisites (reuse Stage 3 — do not duplicate)

- `cluster_archive(positions, kmax, warm)` → `(labels, centers)`. Returns `K = 1` when
  `multimodal_gate` says unimodal and grows K only while each cluster explains
  `EXPLAINED_VARIANCE_GAIN` of the variance. **When K = 1, Stage 4 collapses to a single
  metric** (= Stage 3/2). Keep that fall-through.
- `astate.prev_centers::Matrix{T}` — warm-start centres **and** the assignment reference
  (`nearest_center` uses these). Already present.
- `astate.metric_steps`, `metric_due`, `advance_memory_metric!`, the `ArchiveMetricStrategy`
  union, `track_sampling_metric!`/`sampling_pieces` — the cadence + sampling-phase tracking.
  Stage 4 joins the union and adds a `refresh_memory_metric!` method. Note the metric *values*
  keep adapting from the growing archive during sampling, same as Stage 3 (same adaptive-MCMC
  caveat, no new bias — the value update is global, not per-chain-position).
- `memory_inverse_metric` (+ `diagonal_/dense_inverse_metric`, the floor/ridge). Stage 4 calls
  it **per cluster** on each cluster's points.
- `require_estimable_metric` / `require_memory_untempered` — inherited: diag/dense-only, and
  the **parallel-tempering ban**. With PT banned, `cold` = all chains and `hot` = ∅, so
  per-chain assignment runs over every chain and needs no Stage-4 cold/hot handling.
- `needs_private_scratch` / `prepare_chain_metrics!` — the dense `_temp` race handling; Stage 4
  needs a per-label, position-assigned variant (below).

## Types and fields to add

```julia
struct DifferentialEvolutionPerClusterMetric{T <: Real} <: AbstractDifferentialEvolutionMetricStrategy
    shrinkage::T
    every::Int
    kmax::Int
end

function DEM.per_cluster_metric(; shrinkage = 0.0, every = 100, kmax = 10)
    # validate ranges exactly like cluster_pooled_metric
end

validate_metric_strategy(::DifferentialEvolutionPerClusterMetric, m) = require_estimable_metric(m, "per_cluster_metric")
validate_metric_state(::DifferentialEvolutionPerClusterMetric, s)    = require_memory_untempered(s, "per_cluster_metric")
```

On `HMCAdaptiveState`, add one field (parametric-stable, like `prev_centers`):

- `cluster_metrics::Vector{M}` — the K per-cluster metrics (length K, K ≥ 1).

No per-chain `labels` field and **no `labels_frozen`** — labels are derived from position each
step, never stored across the warmup boundary. `chain_metrics` (already present, length
n_chains) stays the per-chain trajectory slot, filled by position-assignment each step.

## Estimation, assignment, fan-out

```julia
# K separate metrics: each cluster's own covariance through Stage 3's shrinkage/floor path.
function per_cluster_inverse_metrics(ms, astate, positions)
    labels, centers = cluster_archive(positions, ms.kmax, astate.prev_centers)
    astate.prev_centers = centers
    K = size(centers, 2)
    Ms = map(1:K) do k
        pts = [positions[i] for i in eachindex(positions) if labels[i] == k]
        # guard tiny clusters: fall back to the global/pooled metric if too few points.
        # (With an archive of hundreds and small K, n/K >= D holds — the live-only degeneracy
        # bound is gone because memory supplies many points per mode.)
        AdvancedHMC.renew(astate.metric, memory_inverse_metric(astate.metric, pts, ms.shrinkage))
    end
    return Ms, centers
end

nearest_center(θ, centers) = argmin(j -> column_sqdist(θ, centers, j), 1:size(centers, 2))

# Assign each chain by CURRENT position, then fan out into the trajectory slots.
# Diagonal/unit: share the cluster metric object (read-only across threads).
function assign_chain_metrics!(chain_metrics, cluster_metrics, centers, x)
    for i in eachindex(chain_metrics)
        chain_metrics[i] = cluster_metrics[nearest_center(x[i], centers)]
    end
end
# Dense: chain_metrics[i] is a private-scratch copy; copy in its cluster's M⁻¹/chol, leave _temp.
function assign_chain_metrics!(chain_metrics::Vector{<:DenseEuclideanMetric}, cluster_metrics, centers, x)
    for i in eachindex(chain_metrics)
        src = cluster_metrics[nearest_center(x[i], centers)]
        chain_metrics[i].M⁻¹ .= src.M⁻¹
        chain_metrics[i].cholM⁻¹.data .= src.cholM⁻¹.data
    end
end
```

Two cadences, both already in the architecture:

- **Metric values** (the K `cluster_metrics` + `centers`): recomputed on the `metric_steps` /
  `every` cadence via `refresh_memory_metric!`, warmup and sampling alike (archive grows).
- **Assignment** (`nearest_center` per chain): refreshed on the reassignment cadence (on DE
  moves if a hook exists, else the `every` cadence) — **not** every HMC step. It is `N × K` and
  cheap. The carried label is then used, fixed, by the HMC steps until the next refresh.

```julia
function refresh_memory_metric!(ms::DifferentialEvolutionPerClusterMetric, astate, state)
    astate.cluster_metrics, astate.prev_centers = per_cluster_inverse_metrics(ms, astate, informative_positions(state))
    return nothing
end
```

`run_trajectories!` is otherwise untouched: it already reads `chain_metrics[i]` per chain. The
only Stage-4 insertion is the `assign_chain_metrics!` call (replacing the single-metric
`prepare_chain_metrics!` fan-out) immediately before the trajectory loop, in both `step` and
`step_warmup`, so assignment is by current position in both phases. With K = 1 this reduces to
the Stage 3 single-metric fan-out.

## Deferred: shared vs per-cluster ε

Still one shared ε. Per-cluster metrics fix mode *shape*; a residual *scale* difference a tight
vs broad mode wants can land in ε. Start shared; if Stage-4 acceptance rates split by cluster,
per-cluster ε rides along at no extra cost (the chain's cluster is already looked up). Known
limitation.

## Validation

Ladder ordering plus the Stage-4-specific correctness tests:

1. **Unimodal Gaussian:** all four stages agree; Stage 4 returns K = 1 and matches Stage 3.
2. **Separated, equal-shape modes:** Stages 3 & 4 beat 1 & 2; **3 ≈ 4** (pooling already
   suffices when shapes match).
3. **Separated, different-shape modes (heteroscedastic):** **Stage 4 beats Stage 3** — the
   case Stage 4 exists for, and where its per-mode geometry pays off.
4. **Exactness on separated modes:** marginals/quantiles must match a trusted long reference.
   This is the core claim and the justification for Stage 4: with separated modes and fixed
   per-chain metrics it is exact.
5. **Efficiency (not correctness) on close modes:** as separation shrinks and trajectories
   bridge clusters, expect acceptance/ESS to degrade (the fixed metric fits neither bridged
   region) — confirm this manifests as *inefficiency*, and that reparameterization (collapsing
   the shape difference so Stage 3 suffices) restores efficiency. Separately confirm you are
   **not** reassigning every HMC step from current position, the one setting that would convert
   the close-mode case into a genuine (small) bias.
6. **Tempering ban + memoryless ban** still fire (inherited).
7. **Assignment behavior:** confirm a chain's assigned metric is constant through pure-HMC
   stretches and changes only after a DE move crosses a basin (the emergent "switch on DE
   moves"). Confirm the within-trajectory metric is constant (assert in a debug build).

## Where Stage 4 sits

**Reparameterization first** — if a transform makes heteroscedastic modes homoscedastic, Stage
3 suffices and the position-dependence question never arises. Failing that, Stage 4 is the
correct choice for **separated** heteroscedastic modes: position-based per-mode assignment is
exact there and delivers genuine per-mode geometry, with no frozen labels and no DE hook. For
**close, differently-shaped** modes no Euclidean per-cluster scheme is safe — reparameterize or
go to RHMC, out of scope here. For separated modes the per-mode metric is exact and optimal;
the close-mode cost is inefficiency, not bias, provided assignment is never recomputed per HMC
step from current position. Stage 3's single within-cluster-pooled metric remains the default
and the K = 1 / tiny-cluster fall-through throughout.
