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

# Selects how the metric is produced during adaptation.
abstract type AbstractDifferentialEvolutionMetricStrategy end

"""Produce the metric by driving AdvancedHMC's `StanHMCAdaptor` directly."""
struct DifferentialEvolutionStockAdaptorMetric <: AbstractDifferentialEvolutionMetricStrategy end

# Reject an incompatible strategy/metric pairing at construction; the default accepts every metric.
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
`DifferentialEvolutionHMCSampler` carrying the *adapted* pieces, which is how the adapted
metric/`κ` reach the post-warmup `step`. That fixed sampler also carries the live
`HMCAdaptiveState` (`astate`), so the memory-based strategies keep tracking the growing
archive during sampling; before warmup `astate` is `nothing`.
"""
struct DifferentialEvolutionHMCSampler{
        MS <: AbstractDifferentialEvolutionMetricStrategy,
        M <: AbstractMetric, I <: AbstractIntegrator,
        K <: AbstractMCMCKernel, A <: AbstractAdaptor, AST,
    } <: DEM.AbstractDifferentialEvolutionSampler
    metric_strategy::MS
    metric::M
    integrator::I
    κ::K
    adaptor::A
    # Per-chain metric slots; cold/hot are the chains HMC advances / carries over unchanged.
    chain_metrics::Vector{M}
    cold::Vector{Int}
    hot::Vector{Int}
    astate::AST
end

"""
    HMCAdaptiveState{T}

The shared, evolving HMC pieces — one set per HMC update, NOT one per chain. This lives in
the sampler *state* (`state.adaptive_state`); the immutable sampler holds it only by reference,
threaded through `fix_sampler` so memory-based strategies keep mutating it during sampling.
Concretely parametrised on the AdvancedHMC piece types, which `renew`/`update` preserve across
adaptation.

`metric`/`integrator`/`κ`/`adaptor` start as copies of the sampler's pieces. The metric and
adaptor are mutated/replaced as adaptation proceeds; the step size is refined on the first
HMC step (`refine_step_size!`), guarded by `initialized`.
"""
mutable struct HMCAdaptiveState{
        T <: Real, M <: AbstractMetric, I <: AbstractIntegrator,
        K <: AbstractMCMCKernel, A <: AbstractAdaptor,
    } <: DEM.AbstractDifferentialEvolutionAdaptiveState{T}
    metric::M
    integrator::I
    κ::K
    adaptor::A
    initialized::Bool
    # Per-chain metric slots; cold/hot chain indices (empty until resolved on the first warmup step).
    chain_metrics::Vector{M}
    cold::Vector{Int}
    hot::Vector{Int}
    # Previous cluster centres (D×K), warm-starting the clustering strategies; D×0 otherwise.
    prev_centers::Matrix{T}
    # HMC steps taken (warmup and sampling), driving the metric-recompute cadence.
    metric_steps::Int
    # Per-cluster metrics (length K ≥ 1) for the per-cluster strategy; empty until first recompute.
    cluster_metrics::Vector{M}
    # Centres (D×K) aligned with `cluster_metrics`, for relabelling chains to their nearest mode
    # before every HMC step; D×0 until the first recompute.
    cluster_centers::Matrix{T}
end

# `T` has no field, so it cannot be inferred; take it explicitly and infer the piece types.
function HMCAdaptiveState{T}(
        metric::M, integrator::I, κ::K, adaptor::A, initialized::Bool,
        chain_metrics::Vector{M}, cold::Vector{Int}, hot::Vector{Int}, prev_centers::Matrix{T},
        metric_steps::Int, cluster_metrics::Vector{M}, cluster_centers::Matrix{T}
    ) where {T <: Real, M <: AbstractMetric, I <: AbstractIntegrator, K <: AbstractMCMCKernel, A <: AbstractAdaptor}
    return HMCAdaptiveState{T, M, I, K, A}(
        metric, integrator, κ, adaptor, initialized, chain_metrics, cold, hot, prev_centers,
        metric_steps, cluster_metrics, cluster_centers
    )
end

# -----------------------------------------------------------------------------
# Construction + gradient requirement
# -----------------------------------------------------------------------------

function DEM.setup_hmc_update(
        sampler::AbstractHMCSampler;
        n_dims::Int = 0,
        metric_strategy::AbstractDifferentialEvolutionMetricStrategy = DifferentialEvolutionStockAdaptorMetric()
    )
    # `NUTS`/`HMC`/`HMCDA` store the metric as a symbol and size it lazily, so they need `n_dims`;
    # a hand-built `HMCSampler` already carries a sized metric.
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
    return DifferentialEvolutionHMCSampler(
        metric_strategy, metric, integrator, κ, adaptor, typeof(metric)[], Int[], Int[], nothing
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
    # Only a dense metric needs distinct per-chain copies (its `_temp` races); others share a slot.
    metric = deepcopy(sampler.metric)
    chain_metrics = needs_private_scratch(metric) ?
        [deepcopy(metric) for _ in 1:n_chains] :
        Vector{typeof(metric)}(undef, n_chains)
    return HMCAdaptiveState{Float64}(
        metric, sampler.integrator, sampler.κ, deepcopy(sampler.adaptor),
        false, chain_metrics, Int[], Int[], Matrix{Float64}(undef, d, 0), 0, typeof(metric)[],
        Matrix{Float64}(undef, d, 0)
    )
end

"""
    fix_sampler(::DifferentialEvolutionHMCSampler, ::HMCAdaptiveState)

Return a `DifferentialEvolutionHMCSampler` carrying the adaptive state's current pieces by
reference (they are immutable, so no copy), plus the live `astate` itself. The sampling
`step` reads the pieces, and — for memory-based strategies — keeps mutating `astate` to track
the growing archive (the stock strategy leaves it frozen).
"""
function DEM.fix_sampler(sampler::DifferentialEvolutionHMCSampler, astate::HMCAdaptiveState)
    return DifferentialEvolutionHMCSampler(
        sampler.metric_strategy, astate.metric, astate.integrator, astate.κ, astate.adaptor,
        astate.chain_metrics, astate.cold, astate.hot, astate
    )
end

# -----------------------------------------------------------------------------
# Step-size refinement and metric refresh
# -----------------------------------------------------------------------------

# Representative point for `find_good_stepsize` (the ensemble mean; weak for a multimodal ensemble).
ensemble_mean(x) = sum(x) ./ length(x)

# Refresh the metric (M⁻¹) and step size from the current adaptor state.
function refresh_pieces_from_adaptor!(astate::HMCAdaptiveState, ℓ)
    astate.metric = AdvancedHMC.update(Hamiltonian(astate.metric, ℓ), astate.adaptor).metric
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    return nothing
end

# First-step lazy work: find a good step size at the ensemble and lay out the windowed schedule.
function refine_step_size!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, astate::HMCAdaptiveState, state;
        n_adapts::Int
    )
    ℓ = model_wrapper.logdensity
    h = Hamiltonian(astate.metric, ℓ)
    ϵ = find_good_stepsize(rng, h, ensemble_mean(state.x))
    # Re-anchor the configured step-size adaptor in place, preserving its type/settings.
    ssa = astate.adaptor.ssa
    ssa.state.ϵ = ϵ
    AHMCAdapt.reset!(ssa)
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    # The cursor advances per HMC step, so `n_adapts` is an upper bound in HMC-step units.
    AHMCAdapt.initialize!(astate.adaptor, n_adapts)
    astate.initialized = true
    return nothing
end

# -----------------------------------------------------------------------------
# The trajectory loop (shared by warmup and sampling)
# -----------------------------------------------------------------------------

# Only `DenseEuclideanMetric` writes scratch (`_temp`) per leapfrog step, so a shared dense metric
# races across threads and gets per-chain copies; diagonal/unit metrics only read and are shared.
needs_private_scratch(::AbstractMetric) = false
needs_private_scratch(::AdvancedHMC.DenseEuclideanMetric) = true

# Point each chain's slot at the metric for this step: the shared object (read-only) or, for a dense
# metric, a per-chain copy refreshed in place (its own `_temp` left alone).
prepare_chain_metrics!(chain_metrics, metric::AbstractMetric) = (fill!(chain_metrics, metric); nothing)
function prepare_chain_metrics!(chain_metrics, metric::AdvancedHMC.DenseEuclideanMetric)
    for m in chain_metrics
        m.M⁻¹ .= metric.M⁻¹
        m.cholM⁻¹.data .= metric.cholM⁻¹.data
    end
    return nothing
end

# One HMC transition: full momentum refresh, then write the accepted state straight into `xₚ`/`ldₚ`
# (HMC does its own accept/reject, so there is no outer DEM MH correction).
function run_trajectory!(state, metric, κ, i::Int, model)
    h = Hamiltonian(metric, model)
    z = phasepoint(state.rngs[i], state.x[i], h)
    t = transition(state.rngs[i], h, κ, z)
    state.xₚ[i] .= t.z.θ
    state.ldₚ[i] = t.z.ℓπ.value
    return Float64(t.stat.acceptance_rate)
end

# Chains HMC advances (cold) vs carries over (hot). Running the untempered kernel on hot chains would
# move them by the cold posterior; they mix through the tempered DE updates instead. Resolved once,
# as concrete `Vector{Int}`s for type stability; `hot` is empty without tempering.
cold_chain_indices(state) = cold_chain_indices(state.temperature_ladder, state)
cold_chain_indices(::DEM.DifferentialEvolutionNullTemperatureLadder, state) = collect(eachindex(state.x))
cold_chain_indices(ladder::DEM.AbstractDifferentialEvolutionTemperatureLadder, state) = copy(ladder.cold_chains)
hot_chain_indices(state, cold) = setdiff(eachindex(state.x), cold)

function run_trajectories!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, state, κ::AbstractMCMCKernel,
        chain_metrics::Vector{<:AbstractMetric}, cold::Vector{Int}, hot::Vector{Int}, parallel::Bool
    )
    # Reseed only the advanced chains (mirrors the base DE `step`); untempered this is every chain.
    for i in cold
        Random.seed!(state.rngs[i], rand(rng, UInt))
    end
    # Carry hot chains over unchanged so the `xₚ ↔ x` swap preserves them (empty without tempering).
    for i in hot
        state.xₚ[i] .= state.x[i]
        state.ldₚ[i] = state.ld[i]
    end
    ncold = length(cold)
    α = Vector{Float64}(undef, ncold)
    if parallel
        Threads.@threads for k in 1:ncold
            i = cold[k]
            α[k] = run_trajectory!(state, chain_metrics[i], κ, i, state.chain_models[i])
        end
    else
        ℓ = model_wrapper.logdensity
        for k in 1:ncold
            i = cold[k]
            α[k] = run_trajectory!(state, chain_metrics[i], κ, i, ℓ)
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
pooled_acceptance(α) = sum(α) / length(α)

function pooled_adapt!(adaptor::StanHMCAdaptor, Xₚ, α)
    adaptor.state.i += 1
    mα = pooled_acceptance(α)
    AHMCAdapt.adapt!(adaptor.ssa, Xₚ[1], mα)
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
    pooled_adapt!(astate.adaptor, state.xₚ_smpl_view, α)
    refresh_pieces_from_adaptor!(astate, model_wrapper.logdensity)
    return nothing
end

# -----------------------------------------------------------------------------
# Memory covariance metric (Stage 2)
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionMemoryMetric{T}

Metric strategy that rebuilds `astate.metric` from the memory-archive covariance on a schedule,
throughout both warmup and sampling (the archive keeps growing). See [`memory_metric`](@ref) for
the user-facing constructor and the documented keyword arguments.

The estimate's *shape* follows the target metric, so the same strategy serves both kinds: a
`DiagEuclideanMetric` gets per-coordinate variances (`diag Σ`), a `DenseEuclideanMetric` gets
the full covariance `Σ` (off-diagonals and all). Dense is safe on the threaded trajectory loop
because `run_trajectories!` hands each chain a private-scratch copy of the metric (a shared
`DenseEuclideanMetric` would race on its `_temp` buffer); see `needs_private_scratch`.
"""
struct DifferentialEvolutionMemoryMetric{T <: Real} <: AbstractDifferentialEvolutionMetricStrategy
    shrinkage::T
    every::Int
end

function DEM.memory_metric(; shrinkage::Real = 0.0, every::Integer = 100)
    0 ≤ shrinkage ≤ 1 ||
        error("memory_metric: `shrinkage` must be in [0, 1], got $shrinkage.")
    every ≥ 1 || error("memory_metric: `every` must be ≥ 1, got $every.")
    return DifferentialEvolutionMemoryMetric(float(shrinkage), Int(every))
end

# Accept only metrics with an estimable covariance (diagonal/dense), reject anything else (e.g. unit).
function require_estimable_metric(metric::AbstractMetric, who::AbstractString)
    (metric isa AdvancedHMC.DiagEuclideanMetric || metric isa AdvancedHMC.DenseEuclideanMetric) || error(
        "$who estimates a diagonal or dense covariance, so it requires a `DiagEuclideanMetric` " *
            "or `DenseEuclideanMetric`, but the HMC update was set up with a " *
            "`$(nameof(typeof(metric)))`. Use a diagonal (the `NUTS`/`HMC`/`HMCDA` default) or " *
            "dense metric (`NUTS(0.8; metric = :dense)`)."
    )
    return nothing
end

# The two state-level requirements shared by every memory-archive strategy.
function require_memory_untempered(state, who::AbstractString)
    state.memory isa DEM.DifferentialEvolutionMemoryless && error(
        "$who estimates the HMC mass matrix from the memory archive, but this scheme runs " *
            "memoryless. Enable memory, or use the default stock-adaptor metric."
    )
    has_hot_chains(state.temperature_ladder, state) && error(
        "$who is incompatible with parallel tempering: the memory archive interleaves hot-chain " *
            "positions, so the estimated covariance would mix temperatures. Use the default " *
            "stock-adaptor metric with parallel tempering. (Pure annealing — every chain cooling " *
            "to the cold target — is supported; annealing alongside persistent hot chains is " *
            "still parallel tempering and is not.)"
    )
    return nothing
end

validate_metric_strategy(::DifferentialEvolutionMemoryMetric, metric::AbstractMetric) =
    require_estimable_metric(metric, "memory_metric")

# Reject configs only knowable from the state; called once on the first warmup step.
validate_metric_state(::AbstractDifferentialEvolutionMetricStrategy, state) = nothing
validate_metric_state(::DifferentialEvolutionMemoryMetric, state) =
    require_memory_untempered(state, "memory_metric")

# Any chain that does not finish cold ⇒ parallel tempering (including annealing into hot chains).
has_hot_chains(::DEM.DifferentialEvolutionNullTemperatureLadder, state) = false
has_hot_chains(ladder::DEM.AbstractDifferentialEvolutionTemperatureLadder, state) =
    length(ladder.cold_chains) < length(state.x)

# Positions for the covariance estimate: the memory archive (validated present and untempered).
informative_positions(state) = informative_positions(state.memory, state)
informative_positions(mem::DEM.DifferentialEvolutionMemoryRefill, state) = mem.mem_x
function informative_positions(mem::DEM.DifferentialEvolutionMemoryFill, state)
    return view(mem.mem_x, 1:mem.fill.position)   # only the filled prefix is initialised
end

# Floor each variance at this fraction of the mean variance, bounding the condition number.
const VARIANCE_FLOOR_FRACTION = 1.0e-6

# Per-coordinate variances diag(Σ), with shrinkage toward the mean variance and a positivity floor.
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

# Full covariance Σ, with shrinkage toward isotropy and a ridge that keeps it positive-definite.
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
        @. Σ *= (1 - λ)
        @inbounds for i in 1:D
            Σ[i, i] += λ * v̄
        end
    end
    ridge = T(VARIANCE_FLOOR_FRACTION) * v̄ + floatmin(T)   # keeps Σ PD even at λ = 0
    @inbounds for i in 1:D
        Σ[i, i] += ridge
    end
    return Σ
end

# Shape the estimate to the metric: diagonal variances or full covariance.
memory_inverse_metric(::AdvancedHMC.DiagEuclideanMetric, positions, shrinkage::Real) =
    diagonal_inverse_metric(positions, shrinkage)
memory_inverse_metric(::AdvancedHMC.DenseEuclideanMetric, positions, shrinkage::Real) =
    dense_inverse_metric(positions, shrinkage)

# Advance only the dual-averaging step size (the strategy sets the mass matrix itself).
function step_size_only_adapt!(adaptor::StanHMCAdaptor, α)
    adaptor.state.i += 1
    AHMCAdapt.adapt_stepsize!(adaptor.ssa, pooled_acceptance(α))
    if AHMCAdapt.is_window_end(adaptor)
        AHMCAdapt.reset!(adaptor.ssa)
    end
    return nothing
end

# Recompute on the first HMC step (archive already seeded) and every `every` steps thereafter.
metric_due(astate::HMCAdaptiveState, every::Int) =
    (astate.metric_steps == 1) || (mod(astate.metric_steps, every) == 0)

# Rebuild the mass matrix from the current memory archive. Dispatched per strategy.
function refresh_memory_metric!(ms::DifferentialEvolutionMemoryMetric, astate::HMCAdaptiveState, state)
    M⁻¹ = memory_inverse_metric(astate.metric, informative_positions(state), ms.shrinkage)
    astate.metric = AdvancedHMC.renew(astate.metric, M⁻¹)
    return nothing
end

# -----------------------------------------------------------------------------
# Within-cluster pooled covariance metric (Stage 3)
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionClusterPooledMetric{T}

Metric strategy that clusters the memory archive, centres each archive point on its cluster
mean, and pools the centred deviations into one covariance — stripping the between-mode
inflation that makes the raw memory covariance ([`DifferentialEvolutionMemoryMetric`])
useless for separated modes. See [`cluster_pooled_metric`](@ref) for the user-facing
constructor.

The adjustment is gated: when the archive looks unimodal (or the between-mode variance is
negligible) it falls through to the raw memory covariance, so the output is identical to
[`DifferentialEvolutionMemoryMetric`] there. The result is always a single, position-
independent `M⁻¹`, fanned out to the per-chain slots exactly as Stage 2 does.
"""
struct DifferentialEvolutionClusterPooledMetric{T <: Real} <: AbstractDifferentialEvolutionMetricStrategy
    shrinkage::T
    every::Int
    kmax::Int
end

function DEM.cluster_pooled_metric(; shrinkage::Real = 0.0, every::Integer = 100, kmax::Integer = 10)
    0 ≤ shrinkage ≤ 1 ||
        error("cluster_pooled_metric: `shrinkage` must be in [0, 1], got $shrinkage.")
    every ≥ 1 || error("cluster_pooled_metric: `every` must be ≥ 1, got $every.")
    kmax ≥ 1 || error("cluster_pooled_metric: `kmax` must be ≥ 1, got $kmax.")
    return DifferentialEvolutionClusterPooledMetric(float(shrinkage), Int(every), Int(kmax))
end

validate_metric_strategy(::DifferentialEvolutionClusterPooledMetric, metric::AbstractMetric) =
    require_estimable_metric(metric, "cluster_pooled_metric")
validate_metric_state(::DifferentialEvolutionClusterPooledMetric, state) =
    require_memory_untempered(state, "cluster_pooled_metric")

# An axis flags multimodal above this (Sarle's bimodality coefficient; uniform ≈ 0.555).
const BIMODALITY_THRESHOLD = 0.555
# Cluster only when the between-mode share of some axis's variance exceeds this.
const BETWEEN_VARIANCE_THRESHOLD = 0.1
const KMEANS_MAXITER = 50

# Sarle's bimodality coefficient (finite-sample corrected); larger ⇒ more multimodal.
function bimodality_coefficient(x)
    T = float(eltype(x))
    n = length(x)
    μ = sum(x) / n
    m2 = m3 = m4 = zero(T)
    for v in x
        d = v - μ
        d2 = d * d
        m2 += d2
        m3 += d2 * d
        m4 += d2 * d2
    end
    m2 /= n
    m3 /= n
    m4 /= n
    m2 ≤ 0 && return zero(T)
    g = m3 / m2^(T(3) / 2)
    k = m4 / (m2 * m2) - 3
    return (g * g + 1) / (k + 3 * T(n - 1)^2 / ((n - 2) * (n - 3)))
end

# Cheap gate: does any marginal of the archive look multimodal?
function multimodal_gate(positions)
    n = length(positions)
    n < 4 && return false
    D = length(first(positions))
    col = Vector{Float64}(undef, n)
    for d in 1:D
        for i in 1:n
            col[i] = positions[i][d]
        end
        bimodality_coefficient(col) > BIMODALITY_THRESHOLD && return true
    end
    return false
end

function column_sqdist(p, centers, j)
    s = zero(eltype(centers))
    @inbounds for d in eachindex(p)
        s += abs2(p[d] - centers[d, j])
    end
    return s
end

# Deterministic farthest-first seeding: extreme point, then repeatedly the point farthest
# from the nearest existing centre. Reproducible, so threaded and serial runs agree.
function farthest_first_centers(positions, k)
    T = eltype(first(positions))
    D = length(first(positions))
    M = length(positions)
    centers = Matrix{T}(undef, D, k)
    μ = zeros(T, D)
    for p in positions
        μ .+= p
    end
    μ ./= M
    i1 = argmax(i -> sum(abs2, positions[i] .- μ), 1:M)
    @inbounds centers[:, 1] .= positions[i1]
    nearest = [column_sqdist(positions[i], centers, 1) for i in 1:M]
    for j in 2:k
        ij = argmax(nearest)
        @inbounds centers[:, j] .= positions[ij]
        for i in 1:M
            nearest[i] = min(nearest[i], column_sqdist(positions[i], centers, j))
        end
    end
    return centers
end

function assign_labels!(positions, centers, labels)
    k = size(centers, 2)
    changed = false
    for i in eachindex(positions)
        best = 1
        bestd = column_sqdist(positions[i], centers, 1)
        for j in 2:k
            dj = column_sqdist(positions[i], centers, j)
            if dj < bestd
                bestd = dj
                best = j
            end
        end
        if labels[i] != best
            labels[i] = best
            changed = true
        end
    end
    return changed
end

function update_centers!(positions, centers, labels)
    D, k = size(centers)
    counts = zeros(Int, k)
    fill!(centers, 0)
    for i in eachindex(positions)
        z = labels[i]
        counts[z] += 1
        @inbounds for d in 1:D
            centers[d, z] += positions[i][d]
        end
    end
    for j in 1:k
        counts[j] == 0 && continue
        @inbounds for d in 1:D
            centers[d, j] /= counts[j]
        end
    end
    return counts
end

# Reseed any emptied cluster onto the worst-fit point so all k clusters stay live.
function reseed_empty!(positions, centers, labels, counts)
    for j in eachindex(counts)
        counts[j] == 0 || continue
        worst = argmax(i -> column_sqdist(positions[i], centers, labels[i]), eachindex(positions))
        @inbounds centers[:, j] .= positions[worst]
    end
    return nothing
end

# Lloyd's k-means; warm-starts from `warm` when it matches the requested `k`, else seeds afresh.
function kmeans(positions, k, warm)
    D = length(first(positions))
    centers = (size(warm) == (D, k)) ? copy(warm) : farthest_first_centers(positions, k)
    labels = ones(Int, length(positions))
    for _ in 1:KMEANS_MAXITER
        changed = assign_labels!(positions, centers, labels)
        counts = update_centers!(positions, centers, labels)
        reseed_empty!(positions, centers, labels, counts)
        changed || break
    end
    return labels, centers
end

# A new cluster must explain at least this share of the total variance to be kept.
const EXPLAINED_VARIANCE_GAIN = 0.05

function global_mean(positions)
    T = eltype(first(positions))
    D = length(first(positions))
    μ = zeros(T, D)
    for p in positions
        μ .+= p
    end
    μ ./= length(positions)
    return μ
end

total_variance_trace(positions, μ) =
    sum(p -> sum(abs2, p .- μ), positions)

function between_variance_trace(labels, centers, μ)
    K = size(centers, 2)
    D = size(centers, 1)
    counts = zeros(Int, K)
    for z in labels
        counts[z] += 1
    end
    s = zero(eltype(centers))
    @inbounds for j in 1:K, d in 1:D
        s += counts[j] * abs2(centers[d, j] - μ[d])
    end
    return s
end

# Cluster the archive, growing K while each added cluster explains a worthwhile share of the
# total variance. For separated modes this stops at the mode count: splitting *within* a mode
# adds negligible between-mode variance, so BIC-style over-splitting (which shrinks the pooled
# metric below the true within-mode covariance) is avoided.
function cluster_archive(positions, kmax, warm)
    D = length(first(positions))
    μ = global_mean(positions)
    best_labels = ones(Int, length(positions))
    best_centers = reshape(copy(μ), D, 1)
    total = total_variance_trace(positions, μ)
    total == 0 && return best_labels, best_centers
    kcap = min(kmax, length(positions))
    prev_explained = 0.0
    for k in 2:kcap
        labels, centers = kmeans(positions, k, warm)
        explained = between_variance_trace(labels, centers, μ) / total
        explained - prev_explained < EXPLAINED_VARIANCE_GAIN && break
        prev_explained = explained
        best_labels = labels
        best_centers = centers
    end
    return best_labels, best_centers
end

# Severity diagnostic η: the largest per-axis between-mode share of the total variance.
function between_variance_fraction(positions, labels, centers)
    size(centers, 2) == 1 && return 0.0
    T = eltype(first(positions))
    D = length(first(positions))
    μ = global_mean(positions)
    counts = zeros(Int, size(centers, 2))
    for z in labels
        counts[z] += 1
    end
    between = zeros(T, D)
    @inbounds for j in 1:size(centers, 2), d in 1:D
        between[d] += counts[j] * abs2(centers[d, j] - μ[d])
    end
    total = zeros(T, D)
    @inbounds for i in eachindex(positions), d in 1:D
        total[d] += abs2(positions[i][d] - μ[d])
    end
    η = zero(T)
    @inbounds for d in 1:D
        total[d] > 0 && (η = max(η, between[d] / total[d]))
    end
    return η
end

# Pool the within-cluster deviations through Stage 2's estimator (the (M−K) vs (M−1)
# denominator difference is negligible for an archive of hundreds of points).
function within_cluster_inverse_metric(metric, positions, labels, centers, shrinkage)
    centered = [positions[i] .- view(centers, :, labels[i]) for i in eachindex(positions)]
    return memory_inverse_metric(metric, centered, shrinkage)
end

# Gate and cluster the archive — the scaffold both clustering strategies share. Returns the k-means
# labels and centres plus whether the archive collapsed to a single cluster (unimodal, or the
# between-mode variance below threshold). Warm-starts the next k-means from these centres whenever
# clustering ran, so a transient below-threshold step does not discard the warm start.
function gated_clustering(ms, astate, positions)
    multimodal_gate(positions) || return Int[], astate.prev_centers, true
    labels, centers = cluster_archive(positions, ms.kmax, astate.prev_centers)
    astate.prev_centers = centers
    gated = between_variance_fraction(positions, labels, centers) < BETWEEN_VARIANCE_THRESHOLD
    return labels, centers, gated
end

function cluster_pooled_inverse_metric(ms::DifferentialEvolutionClusterPooledMetric, astate, positions)
    labels, centers, gated = gated_clustering(ms, astate, positions)
    gated && return memory_inverse_metric(astate.metric, positions, ms.shrinkage)
    return within_cluster_inverse_metric(astate.metric, positions, labels, centers, ms.shrinkage)
end

function refresh_memory_metric!(ms::DifferentialEvolutionClusterPooledMetric, astate::HMCAdaptiveState, state)
    M⁻¹ = cluster_pooled_inverse_metric(ms, astate, informative_positions(state))
    astate.metric = AdvancedHMC.renew(astate.metric, M⁻¹)
    return nothing
end

# -----------------------------------------------------------------------------
# Per-cluster metrics (Stage 4)
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionPerClusterMetric{T}

Metric strategy that keeps the cluster covariances **separate** — one metric per mode — and
routes each chain to its current mode's metric. It is the heteroscedastic refinement of
[`DifferentialEvolutionClusterPooledMetric`]: where the pooled strategy collapses the K
within-cluster covariances into one shared metric (correct only when the modes share a shape),
this one fits each mode its own metric, for separated modes of genuinely different shape. See
[`per_cluster_metric`](@ref) for the user-facing constructor.

Like the pooled strategy it is gated: when the archive looks unimodal (or the between-mode
variance is negligible) it reduces to `K = 1`, a single metric identical to
[`DifferentialEvolutionMemoryMetric`]. The K metrics (and their centres) are recomputed only on
the recompute cadence, but each chain is **relabelled to its nearest mode's metric before every
HMC step**, off its current position. For separated modes this label is constant within each
basin, so the per-mode metric is the optimal preconditioner and the choice is independent of the
step's own trajectory; for close modes whose trajectories can bridge clusters, picking the metric
from the step's own starting position introduces a small bias on the boundary in exchange for each
chain always running under its current mode's shape.
"""
struct DifferentialEvolutionPerClusterMetric{T <: Real} <: AbstractDifferentialEvolutionMetricStrategy
    shrinkage::T
    every::Int
    kmax::Int
end

function DEM.per_cluster_metric(; shrinkage::Real = 0.0, every::Integer = 100, kmax::Integer = 10)
    0 ≤ shrinkage ≤ 1 ||
        error("per_cluster_metric: `shrinkage` must be in [0, 1], got $shrinkage.")
    every ≥ 1 || error("per_cluster_metric: `every` must be ≥ 1, got $every.")
    kmax ≥ 1 || error("per_cluster_metric: `kmax` must be ≥ 1, got $kmax.")
    return DifferentialEvolutionPerClusterMetric(float(shrinkage), Int(every), Int(kmax))
end

validate_metric_strategy(::DifferentialEvolutionPerClusterMetric, metric::AbstractMetric) =
    require_estimable_metric(metric, "per_cluster_metric")
validate_metric_state(::DifferentialEvolutionPerClusterMetric, state) =
    require_memory_untempered(state, "per_cluster_metric")

# A cluster with fewer points than this falls back to the whole-archive (pooled) covariance: a
# diagonal estimate needs a couple of points, a dense one needs more than D so the D×D sample
# covariance is full-rank rather than ridge-dominated.
min_cluster_points(::AdvancedHMC.DiagEuclideanMetric) = 2
min_cluster_points(metric::AdvancedHMC.DenseEuclideanMetric) = size(metric, 1) + 1

# One metric over the whole archive (the K = 1 / gated-off fall-through), with its single centre.
function pooled_cluster_metric(ms::DifferentialEvolutionPerClusterMetric, astate, positions)
    M⁻¹ = memory_inverse_metric(astate.metric, positions, ms.shrinkage)
    centers = reshape(global_mean(positions), length(first(positions)), 1)
    return [AdvancedHMC.renew(astate.metric, M⁻¹)], centers
end

# K separate metrics: each cluster's own covariance through Stage 2's shrinkage/floor path, plus the
# centres to assign chains by. Gated exactly like the pooled strategy (a unimodal archive returns
# K = 1 = memory_metric). The returned centres match the metrics' K — distinct from the warm-start
# `astate.prev_centers`, which `gated_clustering` keeps at the real cluster centres even when gated.
function per_cluster_inverse_metrics(ms::DifferentialEvolutionPerClusterMetric, astate, positions)
    labels, centers, gated = gated_clustering(ms, astate, positions)
    gated && return pooled_cluster_metric(ms, astate, positions)
    floor_pts = min_cluster_points(astate.metric)
    metrics = map(1:size(centers, 2)) do k
        pts = [positions[i] for i in eachindex(positions) if labels[i] == k]
        src = length(pts) ≥ floor_pts ? pts : positions
        AdvancedHMC.renew(astate.metric, memory_inverse_metric(astate.metric, src, ms.shrinkage))
    end
    return metrics, centers
end

nearest_center(θ, centers) = argmin(j -> column_sqdist(θ, centers, j), 1:size(centers, 2))

# Assign each chain its current mode's metric by nearest centre. Diagonal/unit share the cluster
# metric object (read-only across threads); dense copies into the per-chain scratch (its own `_temp`
# left alone), mirroring `prepare_chain_metrics!`.
function assign_chain_metrics!(chain_metrics, cluster_metrics, centers, x)
    for i in eachindex(chain_metrics)
        chain_metrics[i] = cluster_metrics[nearest_center(x[i], centers)]
    end
    return nothing
end
function assign_chain_metrics!(
        chain_metrics::Vector{<:AdvancedHMC.DenseEuclideanMetric}, cluster_metrics, centers, x
    )
    for i in eachindex(chain_metrics)
        src = cluster_metrics[nearest_center(x[i], centers)]
        chain_metrics[i].M⁻¹ .= src.M⁻¹
        chain_metrics[i].cholM⁻¹.data .= src.cholM⁻¹.data
    end
    return nothing
end

# Recompute the K cluster metrics (and their centres) from the archive on the recompute cadence.
# Chains are *not* reassigned here: relabelling to the nearest centre runs before every HMC step in
# `fill_chain_metrics!`, so the metric set is held fixed between recomputes while each chain still
# tracks its current mode. The stored centres match the metrics' K — distinct from the warm-start
# `astate.prev_centers`, which `gated_clustering` keeps at the real cluster centres even when gated.
function refresh_memory_metric!(ms::DifferentialEvolutionPerClusterMetric, astate::HMCAdaptiveState, state)
    metrics, centers = per_cluster_inverse_metrics(ms, astate, informative_positions(state))
    astate.cluster_metrics = metrics
    astate.cluster_centers = centers
    return nothing
end

# -----------------------------------------------------------------------------
# Dispatch into the DEM stepping interface
# -----------------------------------------------------------------------------

const ArchiveMetricStrategy = Union{
    DifferentialEvolutionMemoryMetric, DifferentialEvolutionClusterPooledMetric,
    DifferentialEvolutionPerClusterMetric,
}

# Advance the cadence counter and rebuild the metric when due. Shared by warmup and sampling so
# both phases recompute on exactly the same schedule.
function advance_memory_metric!(ms::ArchiveMetricStrategy, astate::HMCAdaptiveState, state)
    astate.metric_steps += 1
    metric_due(astate, ms.every) && refresh_memory_metric!(ms, astate, state)
    return nothing
end

# Fill the per-chain trajectory slots just before the loop. Single-metric strategies fan the one
# adapted metric out to every chain each step; the per-cluster strategy instead relabels each chain
# to its nearest mode's metric here, before every HMC step, off the centres last computed on the
# recompute cadence. Before the first recompute the K cluster metrics do not exist yet, so it seeds
# the slots with the lone seed metric.
fill_chain_metrics!(::AbstractDifferentialEvolutionMetricStrategy, astate::HMCAdaptiveState, state) =
    prepare_chain_metrics!(astate.chain_metrics, astate.metric)
function fill_chain_metrics!(::DifferentialEvolutionPerClusterMetric, astate::HMCAdaptiveState, state)
    if isempty(astate.cluster_metrics)
        prepare_chain_metrics!(astate.chain_metrics, astate.metric)
    else
        assign_chain_metrics!(astate.chain_metrics, astate.cluster_metrics, astate.cluster_centers, state.x)
    end
    return nothing
end

function adapt_metric!(
        ms::ArchiveMetricStrategy, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    step_size_only_adapt!(astate.adaptor, α)
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    advance_memory_metric!(ms, astate, state)
    return nothing
end

function step_warmup(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, <:HMCAdaptiveState};
        parallel::Bool = false, update_memory::Bool = true,
        num_warmup::Int = 1000, kwargs...
    ) where {T <: Real}
    astate = state.adaptive_state
    if !astate.initialized
        validate_metric_state(sampler.metric_strategy, state)
        astate.cold = cold_chain_indices(state)
        astate.hot = hot_chain_indices(state, astate.cold)
        refine_step_size!(rng, model_wrapper, astate, state; n_adapts = num_warmup)
    end
    fill_chain_metrics!(sampler.metric_strategy, astate, state)
    α = run_trajectories!(
        rng, model_wrapper, state, astate.κ, astate.chain_metrics,
        astate.cold, astate.hot, parallel
    )
    adapt_metric!(sampler.metric_strategy, model_wrapper, state, astate, α)
    return DEM.create_sample(state),
        DEM.update_state(
            state; swap_positions = Val(true),
            adaptive_state = astate, update_memory = update_memory
        )
end

# During sampling the memory keeps growing, so memory-based strategies keep recomputing the
# metric on the same cadence; the stock strategy and the pre-warmup sampler (`astate === nothing`)
# stay frozen.
track_sampling_metric!(::AbstractDifferentialEvolutionMetricStrategy, ::Nothing, state) = nothing
track_sampling_metric!(::DifferentialEvolutionStockAdaptorMetric, ::HMCAdaptiveState, state) = nothing
track_sampling_metric!(ms::ArchiveMetricStrategy, astate::HMCAdaptiveState, state) =
    advance_memory_metric!(ms, astate, state)

# Fill the trajectory slots and return the kernel + slots for a sampling step. Once warmed these
# come from the live adaptive state (so sampling-phase recomputes/reassignments are seen); without
# warmup (e.g. a manually fixed sampler) they come from the sampler's own frozen pieces. Dispatched
# on the carried adaptive state, whose concrete type is known, so it stays type-stable.
function prepare_sampling_metrics!(sampler::DifferentialEvolutionHMCSampler, ::Nothing, state)
    prepare_chain_metrics!(sampler.chain_metrics, sampler.metric)
    return sampler.κ, sampler.chain_metrics
end
function prepare_sampling_metrics!(
        sampler::DifferentialEvolutionHMCSampler, astate::HMCAdaptiveState, state
    )
    fill_chain_metrics!(sampler.metric_strategy, astate, state)
    return astate.κ, astate.chain_metrics
end

function step(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, DEM.DifferentialEvolutionAdaptiveStatic{T}};
        parallel::Bool = false, update_memory::Bool = true, kwargs...
    ) where {T <: Real}
    track_sampling_metric!(sampler.metric_strategy, sampler.astate, state)
    κ, chain_metrics = prepare_sampling_metrics!(sampler, sampler.astate, state)
    run_trajectories!(
        rng, model_wrapper, state, κ, chain_metrics, sampler.cold, sampler.hot, parallel
    )
    return DEM.create_sample(state),
        DEM.update_state(state; swap_positions = Val(true), update_memory = update_memory)
end

end
