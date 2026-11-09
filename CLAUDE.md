# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Run one test group:** set `TEST_GROUP` to `core`, `hmc`, `mcmcdiagnostictools`, `mcmcchains`, `flexichains` or `pathfinder` (default `all`).

```bash
TEST_GROUP=mcmcchains julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/groups.jl` is the single list of test dependencies: `COMMON_DEPS` for every group, plus each group's `deps` and `files` in `GROUPS` (`all` takes the union and is the only group that runs `test_correct.jl`).
`activate_test_env(group)` generates and activates `test/environments/v<julia version>/<group>/` (gitignored), so only that group's packages constrain versions; delete it to re-resolve to the newest versions.
`test/Project.toml` only lists `Pkg`; add test dependencies to `test/groups.jl`, not to any `Project.toml`.
CI runs `all` on every Julia version, then `test/differing_groups.jl`, then each differing group in the same job: `core` when one of the package's direct dependencies resolves differently in `all`, an extension group when one of its own deps does.

**Downgrade test (local only, not in CI):** run a single group on the minimum Julia version with its `[compat]` lower bounds pinned.

```bash
TEST_DOWNGRADE=true TEST_GROUP=core julia +1.10 --project=. -e 'using Pkg; Pkg.test()'
```

`lower_bounds(group)` pins the non-stdlib direct dependencies for `core` and the trigger package for an extension group, each at its lowest registered version, in `test/environments/v<julia version>/<group>-downgrade/`; `all` is rejected.
Rerun it when a dependency is added or a lower bound changes.

Do not create temp environments or runner scripts.

**Formatting:** `pre-commit` (Runic for Julia, Panache for Markdown).
The hook formats on commit; do not run `pre-commit` or `runic` manually.

**Docs:** `julia --project=docs docs/make.jl` (Documenter; pages: index, tutorial, hmc, custom, pathfinder).

**Benchmarks:** `benchmark/benchmarks.jl` (AirspeedVelocity runs on PRs).

## Architecture

DE-MC family samplers (deMC, deMCzs, DREAMz) built on `AbstractMCMC.jl`, with targets supplied as `LogDensityProblems` models wrapped in `AbstractMCMC.LogDensityModel`.

### AbstractMCMC integration (`src/chains.jl`)

- `step(rng, model, sampler; kwargs...)` initialises: builds temperature ladder, memory, adaptive state, per-chain model copies, and `DifferentialEvolutionState`.
- `step(rng, model, sampler, state)` is the post-warmup step, dispatched on a `DifferentialEvolutionAdaptiveStatic` state.
  It reseeds `state.rngs` from `rng`, then for each chain calls `proposal!(state, sampler, i)` and `update_chain!` (log density + tempered MH accept/reject).
- `step_warmup` also updates adaptive state; `fix_sampler` / `fix_sampler_state` freeze adapted parameters.
- `update_state` returns a new immutable state, swapping `x`/`xₚ` buffers and updating memory and ladder (`!!` functions return possibly-new objects).

### Core state (`src/DifferentialEvolutionMetropolis.jl`)

`DifferentialEvolutionState` holds current/proposed positions (`x`/`xₚ`), log densities (`ld`/`ldₚ`), per-chain `rngs`, `adaptive_state`, `temperature_ladder`, `memory`, `chain_models` (per-chain `deepcopy` of the model for threading), `order_state` (fixed order cycle positions, `nothing` without a fixed order scheme), and `*_smpl_view` views that restrict output to cold chains when parallel tempering is on.

### Update types

Each implements `proposal!(state, sampler, i)` returning `(offset, ...)`; an offset of `-Inf` means auto-reject.

- `differential_evolution_update.jl`: classic DE-MC.
- `snooker_update.jl`: snooker update.
- `subspace_update.jl` / `subspace_adaptive_update.jl`: DREAM-style randomised subspace with adaptive crossover.
- `composite_sampler.jl`: `setup_random_order_scheme` weight-samples among updates each step, `setup_fixed_order_scheme` runs its first update `n` times then its second once (cycle position held in `state.order_state`); `setup_sampler_scheme` picks between them from `w`.
  Schemes collapse to static adaptive state when all components are static.
- Chain picking uses `fast_sample_chains!` (`fast_sample.jl`) with preallocated index buffers (`n_preallocated_indices`).

### Memory (`src/memory.jl`)

`DifferentialEvolutionMemoryless` picks from live chains.
`DifferentialEvolutionMemoryFill` / `DifferentialEvolutionMemoryRefill` pick from a history archive `mem_x`, with fill-every or thinned fill methods.

### Temperature (`src/temperature.jl`)

Null, static (parallel tempering via `n_hot_chains`), and annealing ladders.
`get_temperature(ladder, i)` scales the MH ratio; hot chains are excluded from output via the `*_smpl_view` views.

### HMC (`src/hmc.jl` + `ext/AdvancedHMCExt.jl`)

`src/hmc.jl` only contains error stubs with docstrings for `setup_hmc_update`, `memory_metric`, `cluster_pooled_metric` and `per_cluster_metric`.
The implementation is in `AdvancedHMCExt`.
`DifferentialEvolutionHMCSampler` decomposes a `NUTS`/`HMC`/`HMCDA` into metric/integrator/kernel/adaptor so it can act as one update in a composite scheme.
Metric strategies: stock Stan adaptor, or experimental archive-based strategies that estimate the mass matrix from the memory archive (these require memory and reject parallel tempering).

### Extensions

`MCMCChainsExt` / `FlexiChainsExt` handle output bundling to those chain types.
`MCMCDiagnosticToolsExt` implements `r̂_stopping_criteria` (`src/convergence.jl` stubs it).
Default output is `DifferentialEvolutionOutput` (`samples[iter, chain, param]`, `ld[iter, chain]`).

### Templates (`src/templates.jl`)

`deMC`, `deMCzs`, `DREAMz` assemble sampler schemes and call `AbstractMCMC.sample`, delegating to `_deMC` etc., which accept an `Int` iteration count or a stopping criterion.

### Testing conventions

- `test/setup.jl` defines the shared helpers and `IsotropicNormalModel` and `BendyBananaModel`; `test/hmc_setup.jl` defines `CorrelatedGaussianModel` and `hmc_adaptive_state`.
  Do not redefine them in individual files.
- Keep the test log free of warnings: wrap setups that deliberately trigger a warning not under test in `suppress_warnings(f, patterns...)`, passing only the expected message patterns (e.g. `NO_SWAP_WARNING`).
- `test_correct.jl` is a slow simulation-based correctness test (sequential rank-uniformity, one-step covariance, short-run Uniform stationarity and long-run Normal moments, looped over updates); it only runs in the `all` group.
- `Aqua.test_all` runs ambiguity/piracy checks.
- Extension test files load their trigger packages themselves; the `core` environment has no weakdeps installed, so core tests check behaviour without extensions.
- A test that needs an extension goes in that extension's group, not in a core file.
