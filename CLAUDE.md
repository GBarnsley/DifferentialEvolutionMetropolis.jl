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

`test/groups.jl` is the single list of test dependencies: `COMMON_DEPS` for every group, plus each group's `deps` and `files` in `GROUPS` (`all` takes the union).
`ALL_ONLY_FILES` in `test/runtests.jl` (`test_correct.jl`, `test_correct_invalid.jl`, `test_correct_hmc.jl`) only run in the `all` group.
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

**Docs:** `julia --project=docs docs/make.jl` (Documenter; pages: index, tutorial, hmc, custom, pathfinder, turing).

**Benchmarks:** `benchmark/benchmarks.jl` (AirspeedVelocity runs on PRs); `benchmark/individual.jl` for local runs.

## Architecture

DE-MC family samplers (deMC, deMCzs, DREAMz) built on `AbstractMCMC.jl`, with targets supplied as `LogDensityProblems` models wrapped in `AbstractMCMC.LogDensityModel`.

### AbstractMCMC integration (`src/chains.jl`)

- `step(rng, model, sampler; kwargs...)` initialises: resolves `initial_position`, builds temperature ladder, memory, adaptive state, per-chain model copies, and `DEMState`.
- `step(rng, model, sampler, state)` is the post-warmup step, dispatched on a `DEMAdaptiveStatic` state; on an adaptive state it calls `fix_sampler_state` and delegates.
  It reseeds `state.rngs` from `rng`, calls `prepare_scratch!`, then `update_chains!`, which for each chain calls `proposal!` and `update_chain!` (log density + tempered MH accept/reject).
- `update_chains!` threads over chains when `parallel = true`, except a memoryless state with a `DEMSamplerSerial` update, which updates chains in turn.
- `step_warmup` also updates adaptive state; `fix_sampler` / `fix_sampler_state` (unexported, documented with the rest of the custom sampler API in `docs/src/custom.md`) freeze adapted parameters.
- `update_state` returns a new immutable state, swapping `x`/`xₚ` buffers and updating memory and ladder (`!!` functions return possibly-new objects).
- `pick_chains(state, i, Val(N))` draws `N` distinct other chains (memoryless) or archive positions (memory) for chain `i`, using `fast_sample_chains!` (`fast_sample.jl`) with preallocated index buffers (`n_preallocated_indices`).

### Core types (`src/DifferentialEvolutionMetropolis.jl`)

`DEMState` holds current/proposed positions (`x`/`xₚ`), log densities (`ld`/`ldₚ`), per-chain `rngs`, `adaptive_state`, `temperature_ladder`, `memory`, `chain_models` (`n_chains * n_tries` `deepcopy`s of the model for threading), `multi_try` (trial buffers, `nothing` when `n_tries == 1`), `order_state` (fixed order cycle positions, `nothing` without a fixed order scheme), and `*_smpl_view` views that restrict output to cold chains when parallel tempering is on.

An update is an `AbstractDEMSampler{E, T}` with an execution mode `E` (`DEMSamplerSerial`: memoryless chains must update in turn; `DEMSamplerMultiThread`) and a try mode `T` (`DEMSamplerSingleTry` / `DEMSamplerMultipleTry`).
`chains_required` gives the minimum memoryless chain count (fallback 3) and `memory_picks` the number of archive draws.

### Update types

Each implements `proposal!(xₚ, x, state, sampler, i)`, writing the proposal into `xₚ` and returning a named tuple with at least `offset`, the log Hastings correction; `-Inf` means auto-reject.

- `differential_evolution_update.jl`: classic DE-MC.
- `snooker_update.jl`: snooker update (asymmetric, returns a non-zero offset).
- `subspace_update.jl` / `subspace_adaptive_update.jl`: DREAM-style randomised subspace with adaptive crossover.
- DE, snooker and subspace are `DEMSamplerSerial` and `DEMSamplerMultipleTry`.
- `swap_update.jl`: `setup_swap_update` swaps positions between chains at adjacent temperatures with its own `step` (no `proposal!`); the templates append it via `use_parallel_tempering_swaps` (default on when `n_hot_chains > 0`).
  `check_swap_setup` warns for hot chains without swaps and swaps without hot chains.
- `multi_try.jl`: multiple-try Metropolis for `n_tries > 1` with `DEMSamplerMultipleTry` updates, as `update_chains!` methods that propose and evaluate forward and reference trials.
- `composite_sampler.jl`: `setup_random_order_scheme` weight-samples among updates each step, `setup_fixed_order_scheme` runs its first update `n` times then its second once (cycle position held in `state.order_state`); `setup_sampler_scheme` picks between them from `w`.
  Schemes nest, and collapse to static adaptive state when all components are static.
- `scratch.jl`: `DEMProposalScratch` per-chain proposal buffers, sized by `prepare_scratch!` and read with `chain_buffers`; also defines `==`/`hash` for samplers, ignoring scratch.

### Memory (`src/memory.jl`)

`DEMMemoryless` picks from live chains.
`DEMMemoryFill` picks from a history archive `mem_x`, filled every step (`DEMMemoryFillEvery`) or thinned (`DEMMemoryFillThin`); when full it grows by `memory_size`, or with `memory_refill` becomes a `DEMMemoryRefill` that overwrites from the start.
`check_memory_size!` warns when the archive will hold fewer positions than `target_memory_size` before sampling.

### Temperature (`src/temperature.jl`)

Null, static (parallel tempering via `n_hot_chains`), and annealing ladders.
`get_temperature(ladder, i)` scales the MH ratio; hot chains are excluded from output via the `*_smpl_view` views.

### HMC (`src/hmc.jl` + `ext/AdvancedHMCExt.jl`)

`src/hmc.jl` only contains error stubs with docstrings for `setup_hmc_update`, `memory_metric`, `cluster_pooled_metric` and `per_cluster_metric`.
The implementation is in `AdvancedHMCExt`.
`DEMHMCSampler` (`DEMSamplerMultiThread`, `DEMSamplerSingleTry`) decomposes a `NUTS`/`HMC`/`HMCDA` into metric/integrator/kernel/adaptor so it can act as one update in a composite scheme.
Without warmup it runs the pieces as built, which needs an integrator with an explicit step size.
Metric strategies: stock Stan adaptor, or experimental archive-based strategies (`memory_metric`, `cluster_pooled_metric`, `per_cluster_metric`) that estimate the mass matrix from the memory archive (these require memory and reject parallel tempering).

### Extensions

`MCMCChainsExt` / `FlexiChainsExt` handle output bundling to those chain types.
`MCMCDiagnosticToolsExt` implements `r̂_stopping_criteria` (`src/convergence.jl` stubs it) and the epoch-based template methods `deMC(model, epoch_size, epoch_limit)` etc. `PathfinderExt` adds `resolve_initial_position` methods so Pathfinder results supply chain starting positions and initial memory.
Default output is `DEMOutput` (`samples[iter, chain, param]`, `ld[iter, chain]`), assembled by `bundle_samples` / `chainsstack` in `src/utilities.jl`.

### Templates (`src/templates.jl`)

`deMC`, `deMCzs`, `DREAMz` assemble sampler schemes and call `AbstractMCMC.sample`, delegating to `_deMC` etc., which accept an `Int` iteration count or a stopping criterion.
`src/docs.jl` holds docstring fragments shared by the templates and `step` (e.g. `generic_de_kwargs`).

### Other source files

`src/show.jl` defines `Base.show` methods for samplers, states and output (tested in `test_show.jl` / `test_show_hmc.jl`).

### Testing conventions

- `test/setup.jl` defines the shared helpers (`suppress_warnings`, warning patterns such as `NO_SWAP_WARNING`, `backwards_compat_rng`) and models (`IsotropicNormalModel`, `BendyBananaModel`, `UnitIntervalModel`, `CountingNormalModel`, `CountingNaNAtModel`, `IntervalModel`); `test/hmc_setup.jl` defines `CorrelatedGaussianModel` and `hmc_adaptive_state`.
  Do not redefine them in individual files.
- Keep the test log free of warnings: wrap setups that deliberately trigger a warning not under test in `suppress_warnings(f, patterns...)`, passing only the expected message patterns.
  `runtests.jl` already suppresses `MEMORY_SIZE_WARNING` and `MEMORY_REFILL_WARNING` around every file.
- `test_correct.jl` (DE-MC updates: without memory, with memory, multi-try, and parallel tempering with swaps) and `test_correct_hmc.jl` (HMC updates, memoryless, no parallel tempering) are slow simulation-based correctness tests (sequential rank-uniformity, one-step covariance, short-run Uniform stationarity and long-run Normal moments, looped over updates).
  `test_correct_invalid.jl` is a negative control: a deliberately invalid parallel memoryless DE that the checks must reject.
  Their shared checks live in `test/correctness_setup.jl`, which `runtests.jl` includes once for the `all` group.
  Every test runs at `CORRECTNESS_TEST_LEVEL`, a Bonferroni correction over all tests in both valid-update files; update `N_CORRECTNESS_UPDATES` (`(; demc, hmc)`, excluding the negative control) when adding or removing an update.
- `Aqua.test_all` runs ambiguity/piracy checks.
- Extension test files load their trigger packages themselves; the `core` environment has no weakdeps installed, so core tests check behaviour without extensions (`test_no_extensions.jl`).
- A test that needs an extension goes in that extension's group, not in a core file.
