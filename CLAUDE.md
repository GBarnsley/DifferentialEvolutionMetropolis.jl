# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```julia
using Pkg; Pkg.test("DifferentialEvolutionMetropolis")
```

**Run a single test file** (from a Julia REPL with the package loaded):
```julia
include("test/test_chains.jl")
```

**Use the Julia MCP server** when possible to avoid recompilation overhead during development.

## Architecture

This package implements DE-MC family samplers (deMC, deMCzs, DREAMz) on top of `AbstractMCMC.jl`. The flow is:

### AbstractMCMC integration

The package integrates via three dispatch points:
- `step(rng, model, sampler)` — **initialisation** (no state arg): allocates `DifferentialEvolutionState`, evaluates initial log-densities, sets up memory and per-chain RNGs.
- `step(rng, model, sampler, state)` — **post-warmup step**: calls `proposal!` for each chain then `update_chain!` (which calls `logdensity`), supports `parallel=true` via `Threads.@threads`.
- `step_warmup(rng, model, sampler, state)` — **warmup step**: same as above but also updates adaptive state.
- `bundle_samples` / `chainsstack` handle output formatting.

### Core state

`DifferentialEvolutionState` is the central struct (defined in `DifferentialEvolutionMetropolis.jl`). It carries:
- `x` / `xₚ` — current and proposed positions (pre-allocated, swapped each step via `update_state`)
- `ld` / `ldₚ` — corresponding log-densities
- `rngs` — **per-chain** `AbstractRNG` instances (required for thread safety and reproducibility)
- `adaptive_state` — subtype of `AbstractDifferentialEvolutionAdaptiveState`
- `temperature_ladder` — subtype of `AbstractDifferentialEvolutionTemperatureLadder`
- `memory` — subtype of `AbstractDifferentialEvolutionMemory`

### Update types (each in its own file)

All update types implement `proposal!(state, sampler, chain_index) → (offset, ...)`. The `step` function calls `proposal!` then `update_chain!` (MH accept/reject + `logdensity` eval).

- `DifferentialEvolutionSampler` (`differential_evolution_update.jl`) — classic DE-MC: picks 2 chains, adds scaled difference + noise.
- `DifferentialEvolutionSnookerSampler` (`snooker_update.jl`) — snooker update: projects onto line connecting current to a third chain.
- `DifferentialEvolutionSubspaceSampler` / `DifferentialEvolutionSubspaceSamplerFixedGamma` (`subspace_update.jl`) — DREAM-z: updates a random subspace of parameters using multiple difference vectors.
- Adaptive variants live in `subspace_adaptive_update.jl` (adapts crossover probabilities during warmup).

### Composite sampler (`composite_sampler.jl`)

`DifferentialEvolutionCompositeSampler` holds a vector of update types and weight-samples among them each step. When all component adaptive states are static, the composite collapses to a static adaptive state too.

### Memory (`memory.jl`)

Two regimes controlled at initialisation:
- `DifferentialEvolutionMemoryless` — `pick_chains` samples from the live chain positions `state.x`.
- `DifferentialEvolutionMemoryFill` / `DifferentialEvolutionMemoryRefill` — `pick_chains` samples from a historical pool `mem_x`, which grows or cycles over time.

### Threading

`step` with `parallel=true` uses `Threads.@threads` over chains. Thread safety relies on:
1. Per-chain RNGs in `state.rngs` (each chain uses `state.rngs[i]` exclusively).
2. Per-chain pre-allocated proposal buffers (`state.xₚ[i]`).
3. **`logdensity` is called with the shared `model` object** — if the model holds mutable scratch buffers, this races. The safe fix is per-worker `deepcopy(model)` in the parallel branch of `update_chain!` / `step`.

### Template functions (`templates.jl`)

`deMC`, `deMCzs`, `DREAMz` are high-level entry points that assemble sampler schemes and call `AbstractMCMC.sample`. They delegate to `_deMC`, `_deMCzs`, `_DREAMz` which accept either an `Int` (fixed iterations) or a stopping criterion.

### Testing conventions

- `test/runtests.jl` defines shared model structs (`IsotropicNormalModel`, `BendyBananaModel`) used across test files — do not redefine them in individual test files.
- Correctness is validated via a sequential rank-uniformity test in `test_correct.jl` (simulation-based, slow).
- `Aqua.test_all` runs ambiguity/piracy checks.
- `@info` is suppressed globally in tests via `disable_logging(Logging.Info)`.
