# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Run individual test files:** use the Julia MCP server (`mcp__julia__julia_eval`) with `env_path` set to `./test` (no `Project.toml` needed; deps resolve from the stacked global env). Mirror the `test/runtests.jl` preamble (`using` lines, `disable_logging(Logging.Info)`, `backwards_compat_rng`, the shared model structs), then `include` the file. Include `test_hmc.jl` before other `test_hmc_*.jl` files, since it defines `CorrelatedGaussianModel` and `hmc_adaptive_state`. Do not create temp environments or runner scripts.

**Formatting:** Runic (`pre-commit` hook and CI check). Run `runic --inplace src ext test` before committing.

**Docs:** `julia --project=docs docs/make.jl` (Documenter; pages: index, tutorial, hmc, custom).

**Benchmarks:** `benchmark/benchmarks.jl` (AirspeedVelocity runs on PRs).

## Architecture

DE-MC family samplers (deMC, deMCzs, DREAMz) built on `AbstractMCMC.jl`, with targets supplied as `LogDensityProblems` models wrapped in `AbstractMCMC.LogDensityModel`.

### AbstractMCMC integration (`src/chains.jl`)

- `step(rng, model, sampler; kwargs...)` initialises: builds temperature ladder, memory, adaptive state, per-chain model copies, and `DifferentialEvolutionState`.
- `step(rng, model, sampler, state)` is the post-warmup step, dispatched on a `DifferentialEvolutionAdaptiveStatic` state. It reseeds `state.rngs` from `rng`, then for each chain calls `proposal!(state, sampler, i)` and `update_chain!` (log density + tempered MH accept/reject).
- `step_warmup` also updates adaptive state; `fix_sampler` / `fix_sampler_state` freeze adapted parameters.
- `update_state` returns a new immutable state, swapping `x`/`xₚ` buffers and updating memory and ladder (`!!` functions return possibly-new objects).

### Core state (`src/DifferentialEvolutionMetropolis.jl`)

`DifferentialEvolutionState` holds current/proposed positions (`x`/`xₚ`), log densities (`ld`/`ldₚ`), per-chain `rngs`, `adaptive_state`, `temperature_ladder`, `memory`, `chain_models` (per-chain `deepcopy` of the model for threading), and `*_smpl_view` views that restrict output to cold chains when parallel tempering is on.

### Update types

Each implements `proposal!(state, sampler, i)` returning `(offset, ...)`; an offset of `-Inf` means auto-reject.
- `differential_evolution_update.jl`: classic DE-MC.
- `snooker_update.jl`: snooker update.
- `subspace_update.jl` / `subspace_adaptive_update.jl`: DREAM-style randomised subspace with adaptive crossover.
- `composite_sampler.jl`: `setup_sampler_scheme` weight-samples among updates each step; collapses to static adaptive state when all components are static.
- Chain picking uses `fast_sample_chains!` (`fast_sample.jl`) with preallocated index buffers (`n_preallocated_indices`).

### Memory (`src/memory.jl`)

`DifferentialEvolutionMemoryless` picks from live chains. `DifferentialEvolutionMemoryFill` / `DifferentialEvolutionMemoryRefill` pick from a history archive `mem_x`, with fill-every or thinned fill methods.

### Temperature (`src/temperature.jl`)

Null, static (parallel tempering via `n_hot_chains`), and annealing ladders. `get_temperature(ladder, i)` scales the MH ratio; hot chains are excluded from output via the `*_smpl_view` views.

### HMC (`src/hmc.jl` + `ext/AdvancedHMCExt.jl`)

`src/hmc.jl` only contains error stubs with docstrings for `setup_hmc_update`, `memory_metric`, `cluster_pooled_metric` and `per_cluster_metric`. The implementation is in `AdvancedHMCExt`. `DifferentialEvolutionHMCSampler` decomposes a `NUTS`/`HMC`/`HMCDA` into metric/integrator/kernel/adaptor so it can act as one update in a composite scheme. Metric strategies: stock Stan adaptor, or experimental archive-based strategies that estimate the mass matrix from the memory archive (these require memory and reject parallel tempering).

### Extensions

`MCMCChainsExt` / `FlexiChainsExt` handle output bundling to those chain types. `MCMCDiagnosticToolsExt` implements `r̂_stopping_criteria` (`src/convergence.jl` stubs it). Default output is `DifferentialEvolutionOutput` (`samples[iter, chain, param]`, `ld[iter, chain]`).

### Templates (`src/templates.jl`)

`deMC`, `deMCzs`, `DREAMz` assemble sampler schemes and call `AbstractMCMC.sample`, delegating to `_deMC` etc., which accept an `Int` iteration count or a stopping criterion.

### Testing conventions

- `test/runtests.jl` defines shared `IsotropicNormalModel` and `BendyBananaModel`; do not redefine them in individual files.
- `test_correct.jl` is a slow simulation-based rank-uniformity correctness test.
- `Aqua.test_all` runs ambiguity/piracy checks.
- Template/convergence tests run after `using MCMCDiagnosticTools, MCMCChains, FlexiChains`; earlier tests check behaviour without those extensions loaded.
