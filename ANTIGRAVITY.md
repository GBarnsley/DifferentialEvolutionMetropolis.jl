# ANTIGRAVITY.md

This file provides guidance for Antigravity (and other AI coding assistants) when working with code in this repository.

## Commands

### Running Tests
Always use the Julia MCP server (`julia_eval`) or project environment to execute tests to avoid startup overhead:

**Run all tests:**
```julia
using Pkg; Pkg.test("DifferentialEvolutionMetropolis")
```

**Run a single test file:**
Individual test files (e.g., `test/test_differential_evolution_update.jl`) depend on mock models and helpers defined in [test/runtests.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/test/runtests.jl). To run an individual test file:
```julia
# 1. Run the setup and mock definitions in runtests.jl (or define them in the REPL)
# 2. Include the specific test file:
include("test/test_differential_evolution_update.jl")
```

---

## Repository Structure

### Core Source Code (`src/`)
- [DifferentialEvolutionMetropolis.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/DifferentialEvolutionMetropolis.jl) — Main entry point defining exports, module imports, and core types (`DifferentialEvolutionState`, `DifferentialEvolutionOutput`).
- [chains.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/chains.jl) — Core step-logic implementing `AbstractMCMC` integration, state updates (`update_state`), and thread-parallel step logic.
- [templates.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/templates.jl) — High-level user entry points (`deMC`, `deMCzs`, `DREAMz`) wrapping `AbstractMCMC.sample`.
- [composite_sampler.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/composite_sampler.jl) — Implements `DifferentialEvolutionCompositeSampler`, weight-sampling component updates.
- [differential_evolution_update.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/differential_evolution_update.jl) — Classic DE-MC update proposal (`DifferentialEvolutionSampler`).
- [snooker_update.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/snooker_update.jl) — Snooker update proposal (`DifferentialEvolutionSnookerSampler`).
- [subspace_update.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/subspace_update.jl) — DREAM-z subspace update proposal (`DifferentialEvolutionSubspaceSampler`).
- [subspace_adaptive_update.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/subspace_adaptive_update.jl) — Crossover probability adaptation logic during warmup.
- [memory.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/memory.jl) — Memory buffer formats: `DifferentialEvolutionMemoryless`, `DifferentialEvolutionMemoryFill`, `DifferentialEvolutionMemoryRefill`.
- [temperature.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/temperature.jl) — Temperature ladders for parallel tempering and simulated annealing.
- [fast_sample.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/fast_sample.jl) — Pre-allocated and optimized index/chain sampling utilities.
- [convergence.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/convergence.jl) — Diagnostics and stopping rules (e.g., `r̂_stopping_criteria`).
- [utilities.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/utilities.jl) — General mathematical/helper utilities.
- [docs.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/src/docs.jl) — Shared docstrings.

### Package Extensions (`ext/`)
Weak dependencies and their integration:
- [FlexiChainsExt.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/ext/FlexiChainsExt.jl) — Output formatting integration for `FlexiChains.jl`.
- [MCMCChainsExt.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/ext/MCMCChainsExt.jl) — Output formatting integration for `MCMCChains.jl`.
- [MCMCDiagnosticToolsExt.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/ext/MCMCDiagnosticToolsExt.jl) — Convergence diagnostic checking via `MCMCDiagnosticTools.jl`.

### Test Suite (`test/`)
- [runtests.jl](file:///home/gregbarnsley/.julia/dev/DifferentialEvolutionMetropolis.jl/test/runtests.jl) — Main test runner. Declares mock models (`IsotropicNormalModel`, `BendyBananaModel`) used by individual test files.
- `test_*.jl` — Independent test suites focusing on specific samplers, states, or diagnostics (e.g. `test_differential_evolution_update.jl`, `test_snooker_update.jl`, `test_subspace_update.jl`, etc.). Note that `test_diagnostics.jl` is currently excluded/commented-out in `runtests.jl` due to external dependency requirements.

---

## Code Architecture

### AbstractMCMC Integration
The package hooks into the `AbstractMCMC` interface by dispatching the following methods (defined in `src/chains.jl` and `src/composite_sampler.jl`):
1. `step(rng, model, sampler; kwargs...)` — **Initialization**: Sets up the initial `DifferentialEvolutionState`, pre-allocates vectors, and initializes the per-chain random number generators (RNGs).
2. `step(rng, model, sampler, state; kwargs...)` — **Normal sampling step**: Executes proposal generation (`proposal!`) and Metropolis-Hastings acceptance checks across all chains.
3. `step_warmup(rng, model, sampler, state; kwargs...)` — **Warmup step**: Performs normal steps while updating adaptive attributes.
4. `bundle_samples` / `chainsstack` — Handles output conversion into `DifferentialEvolutionOutput` or structures defined by loaded extensions (`MCMCChains`, `FlexiChains`).

### Core State Management
The central data structure is `DifferentialEvolutionState` which holds:
- `x` / `xₚ` — Current and proposed parameter vectors.
- `ld` / `ldₚ` — Current and proposed log-density values.
- `rngs` — A vector of independent `AbstractRNG` objects (one per chain) for thread-safe concurrent sampling.
- `adaptive_state` — Adaptive options (`AbstractDifferentialEvolutionAdaptiveState`).
- `temperature_ladder` — Holds temperatures for cold/hot chains (`AbstractDifferentialEvolutionTemperatureLadder`).
- `memory` — Reference history database (`AbstractDifferentialEvolutionMemory`).

### Parallelization & Thread Safety
When running with `parallel=true` in `step()`, the sampler divides chain updates using `Threads.@threads`.
- **RNGs**: Chain `i` uses `state.rngs[i]` exclusively, preventing state race conditions.
- **Buffers**: Proposals are written to chain-specific pre-allocated arrays `state.xₚ[i]`.
- **Shared Model Caution**: If the user-supplied model log-density evaluator contains mutable workspace variables, running in parallel will race. Users must ensure thread safety or supply an immutable model.

---

## Style Guidelines

Refer to the [Julia Standards](file:///home/gregbarnsley/.gemini/skills/julia-standards/SKILL.md) skill:
- **Naming**: Use `CamelCase` for types, `snake_case` for functions/variables, and append a bang `!` to functions mutating state or arguments (e.g., `proposal!`, `update_state!`).
- **Indentation**: Standard 4 spaces. No parentheses around conditions in `if` statements.
- **Type Piracy**: Avoid defining methods on external types not owned by this package.
