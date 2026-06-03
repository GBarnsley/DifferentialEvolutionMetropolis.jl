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

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

"""
    DifferentialEvolutionHMCSampler(metric_strategy, metric, integrator, κ, adaptor)

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
    integrator = AdvancedHMC.make_integrator(sampler, oneunit(T))
    κ = AdvancedHMC.make_kernel(sampler, integrator)
    adaptor = AdvancedHMC.make_adaptor(sampler, metric, integrator)
    return DifferentialEvolutionHMCSampler(metric_strategy, metric, integrator, κ, adaptor)
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
    return HMCAdaptiveState{Float64}(
        deepcopy(sampler.metric), sampler.integrator, sampler.κ, deepcopy(sampler.adaptor), false
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
        sampler.metric_strategy, astate.metric, astate.integrator, astate.κ, astate.adaptor
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

function run_trajectory!(state, metric, κ, i::Int, model)
    # The metric is shared read-only across chains (its `M⁻¹`/`sqrtM⁻¹` are only read by
    # `neg_energy`/momentum sampling for the diagonal and unit metrics used here). A dense
    # metric carries a `_temp` scratch buffer that would race if shared — that metric is
    # not used by the threaded path.
    h = Hamiltonian(metric, model)
    z = phasepoint(state.rngs[i], state.x[i], h)
    t = transition(state.rngs[i], h, κ, z)
    state.xₚ[i] .= t.z.θ
    state.ldₚ[i] = t.z.ℓπ.value
    return Float64(t.stat.acceptance_rate)
end

function run_trajectories!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, state,
        metric::AbstractMetric, κ::AbstractMCMCKernel, parallel::Bool
    )
    # Derive per-chain RNGs from the master rng so `step` depends only on `rng` and `state`,
    # mirroring the DE updates and keeping each thread on its own RNG.
    for i in eachindex(state.rngs)
        Random.seed!(state.rngs[i], rand(rng, UInt))
    end
    n = length(state.x)
    α = Vector{Float64}(undef, n)
    if parallel
        # Per-worker model copies (`state.chain_models[i]`) make concurrent gradient
        # evaluation safe; the shared metric/κ are read-only inside the region.
        Threads.@threads for i in 1:n
            α[i] = run_trajectory!(state, metric, κ, i, state.chain_models[i])
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
function pooled_adapt!(adaptor::StanHMCAdaptor, Xₚ, α)
    adaptor.state.i += 1
    mα = sum(α) / length(α)
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
    α = run_trajectories!(rng, model_wrapper, state, astate.metric, astate.κ, parallel)
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
    run_trajectories!(rng, model_wrapper, state, sampler.metric, sampler.κ, parallel)
    return DEM.create_sample(state),
        DEM.update_state(
            state; x = state.xₚ, ld = state.ldₚ, xₚ = state.x, ldₚ = state.ld,
            update_memory = update_memory
        )
end

end # module
