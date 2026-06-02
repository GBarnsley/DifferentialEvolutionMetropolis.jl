module AdvancedHMCExt

import DifferentialEvolutionMetropolis as DEM
import AdvancedHMC
import AdvancedHMC: AbstractHMCSampler, AbstractMetric, AbstractIntegrator,
    AbstractMCMCKernel, Hamiltonian, phasepoint, transition, StanHMCAdaptor
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

abstract type AbstractDifferentialEvolutionHMCSampler <: DEM.AbstractDifferentialEvolutionSampler end

"""
    DifferentialEvolutionHMCSampler(sampler, metric_strategy)

Wraps an AdvancedHMC sampler so it can live in a composite's `updates` vector.
Constructed via `DifferentialEvolutionMetropolis.setup_hmc_update`.
"""
struct DifferentialEvolutionHMCSampler{S <: AbstractHMCSampler, MS <: AbstractDifferentialEvolutionMetricStrategy} <:
    AbstractDifferentialEvolutionHMCSampler
    sampler::S
    metric_strategy::MS
end

"""
    HMCAdaptiveState{T, MS}

The shared modular HMC pieces — one set per HMC update, NOT one per chain. This lives in
the sampler *state* (`state.adaptive_state`) and is mutated during warmup; the immutable
sampler never holds it. `metric`/`integrator`/`κ`/`adaptor` are the AdvancedHMC objects
of the same name.

They are built eagerly in `initialize_adaptive_state` (the integrator/κ/adaptor with a
placeholder step size, since only the step-size *value* needs a chain position, via
`find_good_stepsize`). The step size is refined in place on the first HMC step
(`refine_step_size!`); `initialized` guards that one-off.
"""
mutable struct HMCAdaptiveState{T <: Real, MS <: AbstractDifferentialEvolutionMetricStrategy} <:
    DEM.AbstractDifferentialEvolutionAdaptiveState{T}
    metric::AbstractMetric
    integrator::AbstractIntegrator
    κ::AbstractMCMCKernel
    adaptor::AbstractAdaptor
    metric_strategy::MS
    initialized::Bool
end

"""
    DifferentialEvolutionFixedHMCSampler(sampler, metric_strategy, metric, κ)

The immutable, frozen sampler returned by `fix_sampler`. It carries the adapted
`metric` and kernel `κ` *by reference* — they are immutable AdvancedHMC objects shared
from the final adaptive state, so no copy is made. The post-warmup `step` reads them
straight off the sampler, which is how it reaches the frozen pieces even when the
surrounding composite state hides which `sampler_id` we are. No adaptive state lives on
the sampler.
"""
struct DifferentialEvolutionFixedHMCSampler{S <: AbstractHMCSampler, MS <: AbstractDifferentialEvolutionMetricStrategy, M <: AbstractMetric, K <: AbstractMCMCKernel} <:
    AbstractDifferentialEvolutionHMCSampler
    sampler::S
    metric_strategy::MS
    metric::M
    κ::K
end

# -----------------------------------------------------------------------------
# Construction + gradient requirement
# -----------------------------------------------------------------------------

function DEM.setup_hmc_update(
        sampler::AbstractHMCSampler;
        metric_strategy::AbstractDifferentialEvolutionMetricStrategy = DifferentialEvolutionStockAdaptorMetric()
    )
    return DifferentialEvolutionHMCSampler(sampler, metric_strategy)
end

"""
    initialize_adaptive_state(::DifferentialEvolutionHMCSampler, model_wrapper, n_chains)

Allocate a fully-typed `HMCAdaptiveState`. The metric and a placeholder
integrator/kernel/adaptor are built here (their types depend only on the sampler and
the dimension); only the step-size *value* is deferred to the first HMC step, where a
chain position becomes available for `find_good_stepsize`.

This is also where the gradient requirement is enforced: DE is gradient-free, but HMC
needs a first-order target, so we fail loudly here if the model is order-0 rather than
silently assuming the wrapper carries a gradient.
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
    spl = sampler.sampler
    ms = sampler.metric_strategy
    metric = AdvancedHMC.make_metric(spl, ℓ)
    # Placeholder step size: only fixes the integrator/κ/adaptor TYPES. The value is
    # replaced by `find_good_stepsize` at the ensemble on the first step.
    ϵ₀ = oneunit(AdvancedHMC.sampler_eltype(spl))
    integrator = AdvancedHMC.make_integrator(spl, ϵ₀)
    κ = AdvancedHMC.make_kernel(spl, integrator)
    adaptor = AdvancedHMC.make_adaptor(spl, metric, integrator)
    return HMCAdaptiveState{Float64, typeof(ms)}(
        metric, integrator, κ, adaptor, ms, false
    )
end

"""
    fix_sampler(::DifferentialEvolutionHMCSampler, ::HMCAdaptiveState)

Return a `DifferentialEvolutionFixedHMCSampler` that shares the adaptive state's current
`metric` and `κ` by reference (they are immutable, so no copy). During warmup the
adaptive state is still evolving, so these snapshots are only used post-warmup, when the
state no longer changes and the snapshot is the frozen result.
"""
function DEM.fix_sampler(sampler::DifferentialEvolutionHMCSampler, adaptive_state::HMCAdaptiveState)
    return DifferentialEvolutionFixedHMCSampler(
        sampler.sampler, sampler.metric_strategy, adaptive_state.metric, adaptive_state.κ
    )
end

# -----------------------------------------------------------------------------
# Lazy initialisation of the shared modular pieces
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
# evaluated at the ensemble, rebuild the ϵ-dependent pieces (so dual averaging anchors
# on the real ϵ), and lay out the adaptor's windowed schedule.
function refine_step_size!(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::AbstractDifferentialEvolutionHMCSampler,
        state, astate::HMCAdaptiveState; n_adapts::Int
    )
    spl = sampler.sampler
    ℓ = model_wrapper.logdensity
    h = Hamiltonian(astate.metric, ℓ)
    ϵ = AdvancedHMC.make_step_size(rng, spl, h, ensemble_mean(state.x))
    astate.integrator = AdvancedHMC.make_integrator(spl, ϵ)
    astate.κ = AdvancedHMC.make_kernel(spl, astate.integrator)
    astate.adaptor = AdvancedHMC.make_adaptor(spl, astate.metric, astate.integrator)
    # `n_adapts` only lays out the windowed mass-matrix schedule; the cursor advances
    # per `pooled_adapt!` call (once per HMC step), so it is in HMC-step units. The
    # total warmup length is a safe upper bound — we may walk only a prefix of it.
    AHMCAdapt.initialize!(astate.adaptor, n_adapts)
    astate.initialized = true
    return nothing
end

# -----------------------------------------------------------------------------
# The trajectory loop (shared by warmup and sampling)
#
# For each chain: build a Hamiltonian from the shared metric, draw a fresh phase
# point at the chain's current position (full momentum refresh), take one HMC
# transition, and read the accepted position / log-density / acceptance-rate back.
# HMC does its own Metropolis accept/reject inside `transition`, so we write the
# accepted result straight into `xₚ`/`ldₚ` (proposal-and-accept in one) — NO outer
# DEM MH correction, which would double-count.
# -----------------------------------------------------------------------------

function run_trajectory!(state, metric, κ, i::Int, model)
    # The metric is shared read-only across chains (its `M⁻¹`/`sqrtM⁻¹` are only read by
    # `neg_energy`/momentum sampling for the diagonal and unit metrics used here). A
    # dense metric carries a `_temp` scratch buffer that would race if shared — that
    # metric is not used by the threaded path.
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
    # Derive per-chain RNGs from the master rng so `step` depends only on `rng` and
    # `state`, mirroring the DE updates and keeping each thread on its own RNG.
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

Drive a `StanHMCAdaptor` from the whole `N`-chain population in one step, producing a
single pooled metric shared by every chain.

Two stock calling conventions are avoided. `adapt!(adaptor, X::Matrix, α)` resizes the
preconditioner to `D×N` and estimates `N` independent per-chain metrics, not one pooled
metric. Looping `adapt!(adaptor, xᵢ, αᵢ)` once per chain advances the dual-averaging
step-size schedule `N×` per step, collapsing ϵ to an early, too-small value.

Instead this replicates `StanHMCAdaptor`'s windowing once per HMC step: one dual-
averaging update from the mean acceptance rate, and `N` position pushes into the
mass-matrix Welford estimator (which stays `D`-dimensional). The window cursor ticks
once per HMC step, so `n_adapts` counts HMC steps.
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

# Warmup: threaded trajectory loop, then adapt. Reached for both a bare
# `DifferentialEvolutionHMCSampler` (direct use) and a `DifferentialEvolutionFixedHMCSampler`
# (the composite warmup path); in both cases the live `HMCAdaptiveState` is in the state,
# and the trajectory + adaptation run off that.
function step_warmup(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::AbstractDifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, <:HMCAdaptiveState};
        parallel::Bool = false, update_memory::Bool = true,
        num_warmup::Int = 1000, kwargs...
    ) where {T <: Real}
    astate = state.adaptive_state
    if !astate.initialized
        refine_step_size!(rng, model_wrapper, sampler, state, astate; n_adapts = num_warmup)
    end
    α = run_trajectories!(rng, model_wrapper, state, astate.metric, astate.κ, parallel)
    adapt_metric!(astate.metric_strategy, model_wrapper, state, astate, α)
    return DEM.create_sample(state),
        DEM.update_state(
            state; x = state.xₚ, ld = state.ldₚ, xₚ = state.x, ldₚ = state.ld,
            adaptive_state = astate, update_memory = update_memory
        )
end

# Direct (non-composite) post-warmup use: the state still carries the `HMCAdaptiveState`,
# so freeze its pieces into a `DifferentialEvolutionFixedHMCSampler` and step with that.
# (If warmup never ran, pin the step size first.) This bypasses the generic
# `fix_sampler_state` path, which would otherwise collide with the static `step`.
function step(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionHMCSampler,
        state::DEM.DifferentialEvolutionState{T, <:HMCAdaptiveState};
        kwargs...
    ) where {T <: Real}
    astate = state.adaptive_state
    if !astate.initialized
        refine_step_size!(rng, model_wrapper, sampler, state, astate; n_adapts = 0)
    end
    return step(rng, model_wrapper, DEM.fix_sampler(sampler, astate), state; kwargs...)
end

# Post-warmup (frozen): threaded trajectory loop, no adaptation, reading the frozen
# `metric`/`κ` straight off the immutable sampler. The state's adaptive field is the
# composite adaptive state (composite path) or the `HMCAdaptiveState` (direct path) —
# never the static one, so this cannot collide with the static sampling `step`.
function step(
        rng::AbstractRNG, model_wrapper::LogDensityModel, sampler::DifferentialEvolutionFixedHMCSampler,
        state::DEM.DifferentialEvolutionState{
            T, <:Union{HMCAdaptiveState{T}, DEM.DifferentialEvolutionAdaptiveComposite{T}},
        };
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
