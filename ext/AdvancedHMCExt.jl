module AdvancedHMCExt

import DifferentialEvolutionMetropolis as DEM
import AdvancedHMC
import AdvancedHMC: AbstractHMCSampler, AbstractMetric, AbstractIntegrator,
    AbstractMCMCKernel, Hamiltonian, phasepoint, transition, StanHMCAdaptor, find_good_stepsize
const AHMCAdapt = AdvancedHMC.Adaptation
import AdvancedHMC.Adaptation: AbstractAdaptor
import AbstractMCMC
import AbstractMCMC: LogDensityModel, step, step_warmup
import LogDensityProblems
import Random
import Random: AbstractRNG

# Selects how the metric is produced during adaptation. Dispatched on by `adapt_metric!`,
# so a metric scheme can be swapped without touching the trajectory loop.
abstract type AbstractDifferentialEvolutionMetricStrategy end

"""Produce the metric by driving AdvancedHMC's `StanHMCAdaptor` directly."""
struct DifferentialEvolutionStockAdaptorMetric <: AbstractDifferentialEvolutionMetricStrategy end

# Reject a strategy/metric combination at construction time (`setup_hmc_update`), before any
# sampling, so an incompatible pairing fails fast rather than mid-warmup. The default accepts
# every metric; strategies with metric-type requirements override it (see Stage 2).
validate_metric_strategy(::AbstractDifferentialEvolutionMetricStrategy, ::AbstractMetric) = nothing

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionHMCSampler(metric_strategy, metric, integrator, κ, adaptor, chain_metrics)

One HMC update for a composite scheme. It stores the AdvancedHMC modular pieces directly
— `metric`, `integrator`, kernel `κ`, and `adaptor` — rather than a `NUTS`/`HMC`/`HMCDA`
wrapper, because every per-step operation (running a trajectory, refining the step size,
refreshing the mass matrix) works on these pieces. The wrapper is decomposed into them
once in `setup_hmc_update` and then discarded.

Immutable, like every sampler in this package. The pieces are built at their final size in
`setup_hmc_update` (which is why it needs `n_dims`); only the step *size* is unknown until a
position exists, so it is refined on the first warmup step. `fix_sampler` returns another
`DifferentialEvolutionHMCSampler` carrying the *adapted* pieces, which is how the frozen
metric/`κ` reach the post-warmup `step`.
"""
struct DifferentialEvolutionHMCSampler{
        MS <: AbstractDifferentialEvolutionMetricStrategy,
        M <: AbstractMetric, I <: AbstractIntegrator,
        K <: AbstractMCMCKernel, A <: AbstractAdaptor,
    } <: DEM.AbstractDifferentialEvolutionSampler
    metric_strategy::MS
    metric::M
    integrator::I
    κ::K
    adaptor::A
    # Pre-allocated per-chain scratch metrics for the threaded loop (empty unless the metric
    # needs private scratch — see `needs_private_scratch`). On the frozen sampler this is the
    # same vector the adaptive state owns, passed by reference by `fix_sampler`, so the
    # post-warmup `step` reuses it without allocating.
    chain_metrics::Vector{AbstractMetric}
end

"""
    HMCAdaptiveState{T}

The shared, evolving HMC pieces — one set per HMC update, NOT one per chain. This lives in
the sampler *state* (`state.adaptive_state`) and is mutated during warmup as the metric and
step size adapt; the immutable sampler never holds it. Fields are AdvancedHMC's own
abstract types.

`metric`/`integrator`/`κ`/`adaptor` start as copies of the sampler's pieces. The metric and
adaptor are mutated/replaced as adaptation proceeds; the step size is refined on the first
HMC step (`refine_step_size!`), guarded by `initialized`.
"""
mutable struct HMCAdaptiveState{T <: Real} <: DEM.AbstractDifferentialEvolutionAdaptiveState{T}
    metric::AbstractMetric
    integrator::AbstractIntegrator
    κ::AbstractMCMCKernel
    adaptor::AbstractAdaptor
    initialized::Bool
    # Per-chain scratch metrics, allocated once in `initialize_adaptive_state` and reused every
    # step (never rebuilt in the loop). Empty unless the metric needs private scratch.
    chain_metrics::Vector{AbstractMetric}
end

# -----------------------------------------------------------------------------
# Construction + gradient requirement
# -----------------------------------------------------------------------------

function DEM.setup_hmc_update(
        sampler::AbstractHMCSampler;
        n_dims::Int = 0,
        metric_strategy::AbstractDifferentialEvolutionMetricStrategy = DifferentialEvolutionStockAdaptorMetric()
    )
    # Decompose the wrapper into its modular pieces here, then discard it. AdvancedHMC's
    # `NUTS`/`HMC`/`HMCDA` store the metric as a *symbol* (e.g. `:diagonal`) and only size it
    # once the model is known; we size it up front instead, so `n_dims` is required for those.
    # A hand-built `HMCSampler` already carries a sized metric, so `n_dims` is optional there.
    if sampler.metric isa Symbol && n_dims ≤ 0
        error(
            "setup_hmc_update needs the parameter dimension to size the metric for a " *
                "`$(sampler.metric)` metric; pass `n_dims = <number of parameters>`."
        )
    end
    T = AdvancedHMC.sampler_eltype(sampler)
    metric = AdvancedHMC.make_metric(sampler.metric, T, n_dims)
    validate_metric_strategy(metric_strategy, metric)
    integrator = AdvancedHMC.make_integrator(sampler, oneunit(T))
    κ = AdvancedHMC.make_kernel(sampler, integrator)
    adaptor = AdvancedHMC.make_adaptor(sampler, metric, integrator)
    # No chains exist yet, so the per-chain scratch is empty; it is sized in
    # `initialize_adaptive_state` and reaches the frozen sampler through `fix_sampler`.
    return DifferentialEvolutionHMCSampler(
        metric_strategy, metric, integrator, κ, adaptor, AbstractMetric[]
    )
end

"""
    initialize_adaptive_state(::DifferentialEvolutionHMCSampler, model_wrapper, n_chains)

Build a fresh `HMCAdaptiveState` for this run from copies of the sampler's pieces (the
metric and adaptor carry mutable adaptation state, so each run needs its own). Enforce that
the metric was sized for this model and that the target is differentiable.

The gradient requirement is checked here: DE is gradient-free, but HMC needs a first-order
target, so we fail loudly if the model is order-0 rather than silently assuming the wrapper
carries a gradient.
"""
function DEM.initialize_adaptive_state(
        sampler::DifferentialEvolutionHMCSampler, model_wrapper::LogDensityModel, n_chains::Int
    )
    ℓ = model_wrapper.logdensity
    cap = LogDensityProblems.capabilities(ℓ)
    cap === nothing && error(
        "setup_hmc_update: the model does not implement the LogDensityProblems interface."
    )
    if cap < LogDensityProblems.LogDensityOrder{1}()
        error(
            "setup_hmc_update requires a first-order (gradient) LogDensityProblems target, " *
                "but the supplied model is order-0. Wrap it with an AD backend before sampling, " *
                "e.g. `LogDensityProblemsAD.ADgradient(:ForwardDiff, model)`. The gradient model " *
                "is also a valid target for the DE updates in the same scheme."
        )
    end
    d = LogDensityProblems.dimension(ℓ)
    size(sampler.metric, 1) == d || error(
        "setup_hmc_update was given `n_dims = $(size(sampler.metric, 1))` but the model has " *
            "$d parameters; pass the matching `n_dims`."
    )
    # Pre-allocate one private-scratch metric per chain when the metric needs it (a dense
    # metric's `_temp` would otherwise race across the threaded loop). Allocated once here and
    # refreshed in place each step — never rebuilt inside the trajectory loop. Diagonal/unit
    # metrics are safe to share, so they get no copies (an empty vector).
    chain_metrics = needs_private_scratch(sampler.metric) ?
        AbstractMetric[deepcopy(sampler.metric) for _ in 1:n_chains] : AbstractMetric[]
    return HMCAdaptiveState{Float64}(
        deepcopy(sampler.metric), sampler.integrator, sampler.κ, deepcopy(sampler.adaptor),
        false, chain_metrics
    )
end

"""
    fix_sampler(::DifferentialEvolutionHMCSampler, ::HMCAdaptiveState)

Return a `DifferentialEvolutionHMCSampler` carrying the adaptive state's current pieces by
reference (they are immutable, so no copy). During warmup the adaptive state is still
evolving, so this snapshot only matters post-warmup, when it is the frozen result the
sampling `step` reads.
"""
function DEM.fix_sampler(sampler::DifferentialEvolutionHMCSampler, astate::HMCAdaptiveState)
    # `astate.chain_metrics` is carried by reference (the same pre-allocated vector), so the
    # frozen sampler the post-warmup `step` reads reuses it with no per-step allocation.
    return DifferentialEvolutionHMCSampler(
        sampler.metric_strategy, astate.metric, astate.integrator, astate.κ, astate.adaptor,
        astate.chain_metrics
    )
end

# -----------------------------------------------------------------------------
# Step-size refinement and metric refresh (operate on the pieces, never the wrapper)
# -----------------------------------------------------------------------------

# Representative point for `find_good_stepsize`. The mean is adequate for a unimodal
# ensemble; a multimodal ensemble would want a more robust choice (e.g. a per-coordinate
# median), as the mean can fall in a low-density valley between modes.
ensemble_mean(x) = sum(x) ./ length(x)

# Refresh the metric (M⁻¹) and the kernel's step size from the current adaptor state,
# after each warmup adaptation.
function refresh_pieces_from_adaptor!(astate::HMCAdaptiveState, ℓ)
    astate.metric = AdvancedHMC.update(Hamiltonian(astate.metric, ℓ), astate.adaptor).metric
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    return nothing
end

# First-step lazy work: replace the placeholder step size with `find_good_stepsize`
# evaluated at the ensemble, re-anchor the configured step-size adaptor on it, push it into
# the integrator/κ, and lay out the adaptor's windowed schedule.
function refine_step_size!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, astate::HMCAdaptiveState, state;
        n_adapts::Int
    )
    ℓ = model_wrapper.logdensity
    h = Hamiltonian(astate.metric, ℓ)
    ϵ = find_good_stepsize(rng, h, ensemble_mean(state.x))
    # Re-anchor the user's step-size adaptor in place rather than rebuilding it, so its type
    # and dual-averaging settings are preserved: set its step size and `reset!`, which
    # recomputes the dual-averaging target μ from the new ϵ.
    ssa = astate.adaptor.ssa
    ssa.state.ϵ = ϵ
    AHMCAdapt.reset!(ssa)
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    # `n_adapts` only lays out the windowed mass-matrix schedule; the cursor advances per
    # `pooled_adapt!` call (once per HMC step), so it is in HMC-step units. The total warmup
    # length is a safe upper bound — we may walk only a prefix of it.
    AHMCAdapt.initialize!(astate.adaptor, n_adapts)
    astate.initialized = true
    return nothing
end

# -----------------------------------------------------------------------------
# The trajectory loop (shared by warmup and sampling)
#
# For each chain: build a Hamiltonian from the shared metric, draw a fresh phase point at
# the chain's current position (full momentum refresh), take one HMC transition, and read
# the accepted position / log-density / acceptance-rate back. HMC does its own Metropolis
# accept/reject inside `transition`, so we write the accepted result straight into
# `xₚ`/`ldₚ` (proposal-and-accept in one) — NO outer DEM MH correction, which would
# double-count.
# -----------------------------------------------------------------------------

# Most metrics are safe to share read-only across the threaded loop: their `M⁻¹`/`sqrtM⁻¹`/
# `cholM⁻¹` are only read. `DenseEuclideanMetric` is the exception — `neg_energy` writes its
# `_temp` scratch buffer on every leapfrog step (AdvancedHMC `hamiltonian.jl`), so one dense
# metric shared across threads races on `_temp`. `needs_private_scratch` flags that case; such
# metrics get one pre-allocated copy per chain (`HMCAdaptiveState.chain_metrics`).
needs_private_scratch(::AbstractMetric) = false
needs_private_scratch(::AdvancedHMC.DenseEuclideanMetric) = true

# Refresh the pre-allocated per-chain scratch metrics from the current shared metric, in place
# (no allocation). Only the read-only `M⁻¹`/`cholM⁻¹` are copied — identical across chains and
# fixed for the duration of a step; each scratch keeps its own `_temp`, which is the only field
# written (and hence raced on) during a trajectory.
function sync_scratch_metrics!(scratch, metric::AdvancedHMC.DenseEuclideanMetric)
    for m in scratch
        m.M⁻¹ .= metric.M⁻¹
        m.cholM⁻¹.data .= metric.cholM⁻¹.data
    end
    return nothing
end

function run_trajectory!(state, metric, κ, i::Int, model)
    # `metric` is this chain's metric: the one metric shared read-only across chains (safe for
    # the diagonal/unit metrics, whose fields are only read), or — for a dense metric, whose
    # `_temp` scratch would otherwise race — a per-chain copy with private scratch prepared by
    # `run_trajectories!`. Either way `M⁻¹` is only read here.
    h = Hamiltonian(metric, model)
    z = phasepoint(state.rngs[i], state.x[i], h)
    t = transition(state.rngs[i], h, κ, z)
    state.xₚ[i] .= t.z.θ
    state.ldₚ[i] = t.z.ℓπ.value
    return Float64(t.stat.acceptance_rate)
end

function run_trajectories!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, state,
        metric::AbstractMetric, κ::AbstractMCMCKernel,
        chain_metrics::Vector{AbstractMetric}, parallel::Bool
    )
    # Derive per-chain RNGs from the master rng so `step` depends only on `rng` and `state`,
    # mirroring the DE updates and keeping each thread on its own RNG.
    for i in eachindex(state.rngs)
        Random.seed!(state.rngs[i], rand(rng, UInt))
    end
    n = length(state.x)
    α = Vector{Float64}(undef, n)
    if parallel
        # Per-worker model copies (`state.chain_models[i]`) make concurrent gradient evaluation
        # safe, and κ is read-only inside the region. The metric is shared read-only when that
        # is safe; a dense metric instead uses the pre-allocated per-chain copies, whose
        # read-only data is refreshed in place here (no allocation in the loop).
        if isempty(chain_metrics)
            Threads.@threads for i in 1:n
                α[i] = run_trajectory!(state, metric, κ, i, state.chain_models[i])
            end
        else
            sync_scratch_metrics!(chain_metrics, metric)
            Threads.@threads for i in 1:n
                α[i] = run_trajectory!(state, chain_metrics[i], κ, i, state.chain_models[i])
            end
        end
    else
        # Serial path intentionally shares one `ℓ` across chains (no thread hazard),
        # mirroring the base DE samplers' serial branch.
        ℓ = model_wrapper.logdensity
        for i in 1:n
            α[i] = run_trajectory!(state, metric, κ, i, ℓ)
        end
    end
    return α
end

"""
    pooled_adapt!(adaptor, Xₚ, α)

Drive a `StanHMCAdaptor` from the whole `N`-chain population in one step, producing a single
pooled metric shared by every chain.

Two stock calling conventions are avoided. `adapt!(adaptor, X::Matrix, α)` resizes the
preconditioner to `D×N` and estimates `N` independent per-chain metrics, not one pooled
metric. Looping `adapt!(adaptor, xᵢ, αᵢ)` once per chain advances the dual-averaging
step-size schedule `N×` per step, collapsing ϵ to an early, too-small value.

Instead this replicates `StanHMCAdaptor`'s windowing once per HMC step: one dual-averaging
update from the mean acceptance rate, and `N` position pushes into the mass-matrix Welford
estimator (which stays `D`-dimensional). The window cursor ticks once per HMC step, so
`n_adapts` counts HMC steps.
"""
# Mean acceptance rate across the chain population — the single statistic both step-size
# adaptation paths (stock `pooled_adapt!` and Stage 2's `step_size_only_adapt!`) drive ϵ from.
pooled_acceptance(α) = sum(α) / length(α)

function pooled_adapt!(adaptor::StanHMCAdaptor, Xₚ, α)
    adaptor.state.i += 1
    mα = pooled_acceptance(α)
    # Step size: a single Nesterov dual-averaging update from the pooled acceptance.
    AHMCAdapt.adapt!(adaptor.ssa, Xₚ[1], mα)
    # Mass matrix: ingest every chain's position, but only commit M⁻¹ at window end.
    if AHMCAdapt.is_in_window(adaptor)
        is_update = AHMCAdapt.is_window_end(adaptor)
        n = length(Xₚ)
        for (k, x) in enumerate(Xₚ)
            AHMCAdapt.adapt!(adaptor.pc, x, mα, is_update && k == n)
        end
    end
    if AHMCAdapt.is_window_end(adaptor)
        AHMCAdapt.reset!(adaptor.ssa)
        AHMCAdapt.reset!(adaptor.pc)
    end
    return nothing
end

function adapt_metric!(
        ::DifferentialEvolutionStockAdaptorMetric, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    pooled_adapt!(astate.adaptor, state.xₚ, α)
    refresh_pieces_from_adaptor!(astate, model_wrapper.logdensity)
    return nothing
end

# -----------------------------------------------------------------------------
# Population/memory covariance metric (Stage 2)
#
# Estimate the mass matrix directly from the sampler's own population instead of from
# AdvancedHMC's windowed Welford estimator. With memory enabled the informative set is the
# memory archive (1000s of within-mode draws → a well-conditioned covariance immediately);
# otherwise it is the live chain positions. The step size keeps coming from dual averaging.
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionPopulationMetric{T}

Metric strategy that rebuilds `astate.metric` from the population/memory covariance each
warmup step (on a schedule). See [`population_metric`](@ref) for the user-facing constructor
and the documented keyword arguments.

The estimate's *shape* follows the target metric, so the same strategy serves both kinds: a
`DiagEuclideanMetric` gets per-coordinate variances (`diag Σ`), a `DenseEuclideanMetric` gets
the full covariance `Σ` (off-diagonals and all). Dense is safe on the threaded trajectory loop
because `run_trajectories!` hands each chain a private-scratch copy of the metric (a shared
`DenseEuclideanMetric` would race on its `_temp` buffer); see `needs_private_scratch`.
"""
struct DifferentialEvolutionPopulationMetric{T <: Real} <: AbstractDifferentialEvolutionMetricStrategy
    shrinkage::T
    every::Int
end

function DEM.population_metric(; shrinkage::Real = 0.0, every::Integer = 100)
    0 ≤ shrinkage ≤ 1 ||
        error("population_metric: `shrinkage` must be in [0, 1], got $shrinkage.")
    every ≥ 1 || error("population_metric: `every` must be ≥ 1, got $every.")
    return DifferentialEvolutionPopulationMetric(float(shrinkage), Int(every))
end

# Accept the two Euclidean metrics this strategy can populate (diagonal variances or a full
# covariance) and reject anything else at setup time — e.g. a `UnitEuclideanMetric`, which has
# no estimable mass matrix. Dense is allowed because the threaded loop gives each chain private
# scratch (`needs_private_scratch`), so the shared-`_temp` race is already handled.
function validate_metric_strategy(::DifferentialEvolutionPopulationMetric, metric::AbstractMetric)
    (metric isa AdvancedHMC.DiagEuclideanMetric || metric isa AdvancedHMC.DenseEuclideanMetric) || error(
        "population_metric estimates a diagonal or dense covariance, so it requires a " *
            "`DiagEuclideanMetric` or `DenseEuclideanMetric`, but the HMC update was set up with a " *
            "`$(nameof(typeof(metric)))`. Use a diagonal (the `NUTS`/`HMC`/`HMCDA` default) or " *
            "dense metric (`NUTS(0.8; metric = :dense)`)."
    )
    return nothing
end

# The informative set of positions for the covariance estimate. This is the single place
# that prefers the memory archive when memory is enabled and falls back to the live positions
# otherwise, mirroring the source/count that `pick_chains` reads. The archive is one step
# stale here (the current step's positions are written into memory later, in `update_state`),
# which is immaterial against 1000s of stored draws.
informative_positions(state) = informative_positions(state.memory, state)
# Without memory, `state.xₚ` holds the freshly accepted positions at adapt time (one set per
# chain, typically only a handful — the estimate is correspondingly noisy, so shrink hard).
informative_positions(::DEM.DifferentialEvolutionMemoryless, state) = state.xₚ
informative_positions(mem::DEM.DifferentialEvolutionMemoryRefill, state) = mem.mem_x
function informative_positions(mem::DEM.DifferentialEvolutionMemoryFill, state)
    # Only the filled prefix is meaningful; the tail is preallocated, uninitialised storage.
    return view(mem.mem_x, 1:mem.fill.position)
end

# Floor each variance at this fraction of the mean variance. This bounds the metric's
# condition number (and so guards the integrator) without distorting already well-scaled
# coordinates, for which the floor never binds. Distinct from the configurable `shrinkage`,
# which is an optional extra pull toward the mean variance.
const VARIANCE_FLOOR_FRACTION = 1.0e-6

# Per-coordinate variances of a set of position vectors, returned as the diagonal inverse
# metric M⁻¹ = diag(Σ). Two passes (mean, then squared deviations) over the vector-of-vectors
# representation; no `D×n` matrix is materialised. Shrinkage blends toward the mean variance
# and a floor keeps every entry strictly positive so the metric stays positive-definite.
function diagonal_inverse_metric(positions, shrinkage::Real)
    T = eltype(first(positions))
    D = length(first(positions))
    n = length(positions)
    μ = zeros(T, D)
    for p in positions
        μ .+= p
    end
    μ ./= n
    v = zeros(T, D)
    for p in positions
        @inbounds for d in 1:D
            v[d] += abs2(p[d] - μ[d])
        end
    end
    v ./= max(n - 1, 1)
    v̄ = sum(v) / D
    λ = T(shrinkage)
    if λ > 0
        @. v = (1 - λ) * v + λ * v̄
    end
    var_floor = T(VARIANCE_FLOOR_FRACTION) * v̄ + floatmin(T)
    @. v = max(v, var_floor)
    return v
end

# Full covariance of a set of position vectors, returned as the dense inverse metric M⁻¹ = Σ
# (off-diagonals carry the correlations a diagonal metric throws away). The sample covariance
# is positive-semidefinite but can be singular (e.g. fewer samples than dimensions); shrinkage
# pulls it toward the isotropic target `v̄·I` and a small ridge is always added, so the result
# is strictly positive-definite (and well-conditioned) — `renew` can take its Cholesky factor.
# Runs once per recompute (between trajectory loops), never inside the threaded hot loop.
function dense_inverse_metric(positions, shrinkage::Real)
    T = eltype(first(positions))
    D = length(first(positions))
    n = length(positions)
    μ = zeros(T, D)
    for p in positions
        μ .+= p
    end
    μ ./= n
    Σ = zeros(T, D, D)
    d = Vector{T}(undef, D)
    for p in positions
        @inbounds for i in 1:D
            d[i] = p[i] - μ[i]
        end
        @inbounds for j in 1:D, i in 1:D
            Σ[i, j] += d[i] * d[j]
        end
    end
    Σ ./= max(n - 1, 1)
    v̄ = zero(T)
    @inbounds for i in 1:D
        v̄ += Σ[i, i]
    end
    v̄ /= D
    λ = T(shrinkage)
    if λ > 0
        # Shrink toward the isotropic target v̄·I: scale the whole matrix and lift the diagonal.
        @. Σ *= (1 - λ)
        @inbounds for i in 1:D
            Σ[i, i] += λ * v̄
        end
    end
    # Ridge guarantees positive-definiteness even at λ = 0 (raw Σ may be singular); negligible
    # against well-scaled directions, mirroring the diagonal metric's variance floor.
    ridge = T(VARIANCE_FLOOR_FRACTION) * v̄ + floatmin(T)
    @inbounds for i in 1:D
        Σ[i, i] += ridge
    end
    return Σ
end

# The population covariance, shaped to match the target metric: a diagonal metric wants the
# per-coordinate variances, a dense metric the full covariance. Dispatching on the metric type
# means the strategy needs no separate "dense" flag — the metric the update was built with
# (`:diagonal` vs `:dense`) already says which to produce.
population_inverse_metric(::AdvancedHMC.DiagEuclideanMetric, positions, shrinkage::Real) =
    diagonal_inverse_metric(positions, shrinkage)
population_inverse_metric(::AdvancedHMC.DenseEuclideanMetric, positions, shrinkage::Real) =
    dense_inverse_metric(positions, shrinkage)

# Advance only the dual-averaging step size from the pooled acceptance rate. This reproduces
# the ε half of `pooled_adapt!` (window cursor + per-window dual-averaging reset) but skips the
# mass-matrix pushes into `adaptor.pc`, because Stage 2 estimates Σ from the population. The
# step size still flows out through `update(κ, adaptor)`, which reads only `getϵ(ssa)`.
function step_size_only_adapt!(adaptor::StanHMCAdaptor, α)
    adaptor.state.i += 1
    AHMCAdapt.adapt_stepsize!(adaptor.ssa, pooled_acceptance(α))
    if AHMCAdapt.is_window_end(adaptor)
        AHMCAdapt.reset!(adaptor.ssa)
    end
    return nothing
end

# Recompute the covariance on the first warmup step (`i == 1`, where the archive already holds
# the initial positions) and every `ms.every` steps thereafter. Computing on the first step
# guarantees the population metric is installed at least once even when `every` exceeds the
# warmup budget — otherwise the run would silently keep the placeholder identity metric.
metric_due(astate::HMCAdaptiveState, every::Int) =
    (astate.adaptor.state.i == 1) || (mod(astate.adaptor.state.i, every) == 0)

function adapt_metric!(
        ms::DifferentialEvolutionPopulationMetric, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    # Step size: dual averaging from the pooled acceptance, no mass-matrix pushes. (The metric
    # type was already validated against this strategy in `setup_hmc_update`.)
    step_size_only_adapt!(astate.adaptor, α)
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    # Metric: rebuilt from the population/memory covariance on schedule, shaped (diagonal or
    # dense) to match the metric the update was built with.
    if metric_due(astate, ms.every)
        M⁻¹ = population_inverse_metric(astate.metric, informative_positions(state), ms.shrinkage)
        astate.metric = AdvancedHMC.renew(astate.metric, M⁻¹)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Dispatch into the DEM stepping interface
# -----------------------------------------------------------------------------

# Warmup: threaded trajectory loop off the live `HMCAdaptiveState`, then adapt it. Reached
# both for direct use and for the composite warmup path (where the sampler is a frozen
# snapshot but the live pieces still come from the state).
function step_warmup(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, <:HMCAdaptiveState};
        parallel::Bool = false, update_memory::Bool = true,
        num_warmup::Int = 1000, kwargs...
    ) where {T <: Real}
    astate = state.adaptive_state
    if !astate.initialized
        refine_step_size!(rng, model_wrapper, astate, state; n_adapts = num_warmup)
    end
    α = run_trajectories!(
        rng, model_wrapper, state, astate.metric, astate.κ, astate.chain_metrics, parallel
    )
    adapt_metric!(sampler.metric_strategy, model_wrapper, state, astate, α)
    return DEM.create_sample(state),
        DEM.update_state(
            state; x = state.xₚ, ld = state.ldₚ, xₚ = state.x, ldₚ = state.ld,
            adaptive_state = astate, update_memory = update_memory
        )
end

# Post-warmup (frozen): threaded trajectory loop, no adaptation, reading the frozen
# metric/κ off the sampler. The state's adaptive field is the static one here, because the
# generic `fix_sampler_state`/composite machinery has already fixed the sampler and
# collapsed the state — mirroring how the DE subspace sampler reaches this same point.
function step(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, DEM.DifferentialEvolutionAdaptiveStatic{T}};
        parallel::Bool = false, update_memory::Bool = true, kwargs...
    ) where {T <: Real}
    run_trajectories!(
        rng, model_wrapper, state, sampler.metric, sampler.κ, sampler.chain_metrics, parallel
    )
    return DEM.create_sample(state),
        DEM.update_state(
            state; x = state.xₚ, ld = state.ldₚ, xₚ = state.x, ldₚ = state.ld,
            update_memory = update_memory
        )
end

end
