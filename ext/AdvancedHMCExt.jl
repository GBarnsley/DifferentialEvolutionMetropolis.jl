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
    # Per-chain metric slots; cold/hot are the chains HMC advances / carries over unchanged.
    chain_metrics::Vector{M}
    cold::Vector{Int}
    hot::Vector{Int}
end

"""
    HMCAdaptiveState{T}

The shared, evolving HMC pieces — one set per HMC update, NOT one per chain. This lives in
the sampler *state* (`state.adaptive_state`) and is mutated during warmup as the metric and
step size adapt; the immutable sampler never holds it. Concretely parametrised on the
AdvancedHMC piece types, which `renew`/`update` preserve across adaptation.

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
end

# `T` has no field, so it cannot be inferred; take it explicitly and infer the piece types.
function HMCAdaptiveState{T}(
        metric::M, integrator::I, κ::K, adaptor::A, initialized::Bool,
        chain_metrics::Vector{M}, cold::Vector{Int}, hot::Vector{Int}
    ) where {T <: Real, M <: AbstractMetric, I <: AbstractIntegrator, K <: AbstractMCMCKernel, A <: AbstractAdaptor}
    return HMCAdaptiveState{T, M, I, K, A}(
        metric, integrator, κ, adaptor, initialized, chain_metrics, cold, hot
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
        metric_strategy, metric, integrator, κ, adaptor, typeof(metric)[], Int[], Int[]
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
        false, chain_metrics, Int[], Int[]
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
    return DifferentialEvolutionHMCSampler(
        sampler.metric_strategy, astate.metric, astate.integrator, astate.κ, astate.adaptor,
        astate.chain_metrics, astate.cold, astate.hot
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
        rng::AbstractRNG, model_wrapper::LogDensityModel, state,
        metric::AbstractMetric, κ::AbstractMCMCKernel,
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
    prepare_chain_metrics!(chain_metrics, metric)
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

Metric strategy that rebuilds `astate.metric` from the memory-archive covariance each warmup
step (on a schedule). See [`memory_metric`](@ref) for the user-facing constructor and the
documented keyword arguments.

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

# Accept the metrics with an estimable covariance (diagonal/dense), reject anything else (e.g. unit).
function validate_metric_strategy(::DifferentialEvolutionMemoryMetric, metric::AbstractMetric)
    (metric isa AdvancedHMC.DiagEuclideanMetric || metric isa AdvancedHMC.DenseEuclideanMetric) || error(
        "memory_metric estimates a diagonal or dense covariance, so it requires a " *
            "`DiagEuclideanMetric` or `DenseEuclideanMetric`, but the HMC update was set up with a " *
            "`$(nameof(typeof(metric)))`. Use a diagonal (the `NUTS`/`HMC`/`HMCDA` default) or " *
            "dense metric (`NUTS(0.8; metric = :dense)`)."
    )
    return nothing
end

# Reject configs only knowable from the state; called once on the first warmup step.
validate_metric_state(::AbstractDifferentialEvolutionMetricStrategy, state) = nothing
function validate_metric_state(::DifferentialEvolutionMemoryMetric, state)
    state.memory isa DEM.DifferentialEvolutionMemoryless && error(
        "memory_metric estimates the HMC mass matrix from the memory archive, but this scheme " *
            "runs memoryless. Enable memory, or use the default stock-adaptor metric."
    )
    has_hot_chains(state.temperature_ladder, state) && error(
        "memory_metric is incompatible with parallel tempering: the memory archive interleaves " *
            "hot-chain positions, so the estimated covariance would mix temperatures. Use the " *
            "default stock-adaptor metric with parallel tempering. (Pure annealing — every chain " *
            "cooling to the cold target — is supported; annealing alongside persistent hot chains " *
            "is still parallel tempering and is not.)"
    )
    return nothing
end

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

# Recompute on the first warmup step (archive already seeded) and every `every` steps thereafter.
metric_due(astate::HMCAdaptiveState, every::Int) =
    (astate.adaptor.state.i == 1) || (mod(astate.adaptor.state.i, every) == 0)

function adapt_metric!(
        ms::DifferentialEvolutionMemoryMetric, model_wrapper::LogDensityModel, state,
        astate::HMCAdaptiveState, α
    )
    step_size_only_adapt!(astate.adaptor, α)
    astate.κ = AdvancedHMC.update(astate.κ, astate.adaptor)
    astate.integrator = astate.κ.τ.integrator
    if metric_due(astate, ms.every)
        M⁻¹ = memory_inverse_metric(astate.metric, informative_positions(state), ms.shrinkage)
        astate.metric = AdvancedHMC.renew(astate.metric, M⁻¹)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Dispatch into the DEM stepping interface
# -----------------------------------------------------------------------------

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
    α = run_trajectories!(
        rng, model_wrapper, state, astate.metric, astate.κ, astate.chain_metrics,
        astate.cold, astate.hot, parallel
    )
    adapt_metric!(sampler.metric_strategy, model_wrapper, state, astate, α)
    return DEM.create_sample(state),
        DEM.update_state(
            state; swap_positions = Val(true),
            adaptive_state = astate, update_memory = update_memory
        )
end

function step(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, DEM.DifferentialEvolutionAdaptiveStatic{T}};
        parallel::Bool = false, update_memory::Bool = true, kwargs...
    ) where {T <: Real}
    run_trajectories!(
        rng, model_wrapper, state, sampler.metric, sampler.κ, sampler.chain_metrics,
        sampler.cold, sampler.hot, parallel
    )
    return DEM.create_sample(state),
        DEM.update_state(state; swap_positions = Val(true), update_memory = update_memory)
end

end
