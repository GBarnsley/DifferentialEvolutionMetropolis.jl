# Diagnostic checks and chain rescue

## Recommendation

Periodic diagnostics are useful; automatically copying a poorly performing chain to a better one should **not** be restored as a default sampling operation.
For the usual memory-based `deMCzs` and `DREAMz` workflows, an **opt-in, warmup-only rescue policy** is worth investigating for persistently stuck chains, but its benefit has not yet been established by benchmarks.
This page assesses the proposal in issue #40; it does not introduce a rescue API or change current sampling behavior.

Start with reporting rather than relocation.
Use acceptance/movement and log-density summaries to identify problems, then compare longer warmup, better initialization, proposal tuning, and tempering before adding automatic intervention.
Existing `r̂_stopping_criteria` already provides periodic convergence checking, not chain rescue.

## What is already available

- `MCMCDiagnosticTools` supplies R̂ and ESS; reuse it rather than implementing new convergence estimators.
  The extension currently checks R̂ on the last half of the collected samples.
  That is not a post-rescue window and must not be used unchanged if rescues are permitted during adaptive stopping.
- `AbstractMCMC` separates warmup through `step_warmup` from ordinary `step`.
  The templates pass `num_warmup` and normally discard those iterations; setting `save_burnt = true` deliberately retains warmup, which must not then be presented as stationary posterior draws.
- `update_chain!` returns a Metropolis acceptance flag.
  Ordinary steps discard it; adaptive subspace updates consume it locally.
  HMC uses a different transition path with its own acceptance statistics, so recording only `update_chain!` is insufficient.
- The old `ld_check` and `acceptance_check` compared chains and cloned selected positions/log densities.
  The remaining `test/test_diagnostics.jl` is disabled and targets the removed API; it is not evidence that a new implementation works.

## Would rescue help memory-based sampling?

An archive gives a stuck chain access to differences drawn from past positions even when other live chains are stuck.
That reduces dependence on the current population compared with memoryless DE.
Rescue can still shorten initialization transients when a chain starts far from useful regions, but cannot repair a bad target, an unsuitable proposal scale, or an archive that never represented an important mode.

Archive effects cut both ways: rescued chains can contribute better positions later, but earlier poor positions remain until overwritten (or indefinitely in a growing archive).
Replacing a chain does not repair the archive.
Repeated cloning can also reduce population diversity and bias the archive toward the most frequently visited mode.
Consequently, a speedup in acceptance rate alone is not enough to demonstrate benefit; compare effective samples per log-density evaluation and mode coverage.

There are important limitations to the proposed diagnostics:

- Low acceptance can mean poor scaling rather than a bad starting position.
  High acceptance can mean negligible movement.
  Measure both and require persistence across windows, rather than reacting to one noisy check.
- Low mean log density need not imply a defective chain.
  Legitimate low-density modes and high-dimensional typical sets must not be eliminated in favor of the maximum log-density point.
  The old rule of copying the best chain is particularly risky for multimodal targets.
- R̂ is an across-chain diagnostic, not a reliable ranking of which chain to replace.
  DE chains interact through their population/archive, so nominal chain count does not supply the same independence as separately initialized ensemble runs.
  Cloning can make agreement look better without improving exploration.
  Check independent runs and ESS/mode occupancy as well as within-ensemble R̂.
- Hot chains deliberately target different temperatures.
  Do not compare raw log densities or acceptance rates across temperatures, or replace them from cold chains under the same rule.

## Correctness boundary

Uncorrected selection and copying generally do not preserve the joint target.
Restrict interventions to a clearly marked initialization/adaptation phase, stop them before retained sampling, and allow a settling interval after the final intervention.
Discarding the intervention draw alone does **not** make the next draw stationary; normal convergence checks are still required.
Warmup-only rescue is a heuristic for initialization, not a proof of convergence or of archive-adaptation validity.

Allowing rescue while running until converged is substantially more intrusive.
The sampling driver would need to track the last rescue, discard every earlier draw from the returned posterior, reset diagnostic windows and counters, enforce a minimum settling interval, and eventually stop rescuing.
A moving last-half R̂ window does not provide this guarantee.
The current stopping callback returns a Boolean and receives already collected samples; it is not a clean place to implement output truncation and restart semantics.
Do not add that behavior to `r̂_stopping_criteria` as a side effect.

## Implementation difficulty

<!-- panache-ignore-start -->

| Scope | Difficulty | Main work |
| --- | --- | --- |
| Periodic warmup reporting | Low to moderate | Bounded per-chain windows/counters, validation, cold-chain filtering, acceptance/movement recording across update types |
| Warmup-only rescue for DE/subspace updates | Moderate | Explicit policy state, post-step intervention point, buffer/log-density consistency, archive and adaptation policy, deterministic threaded behavior |
| Rescue with HMC/composite/tempered samplers | Moderate to high | Component-specific statistics and cache/adaptor handling, temperature restrictions, compatibility validation |
| Rescue during convergence-controlled retained sampling | High | Restart/retention lifecycle, settling intervals, diagnostic reset, final-state and chain-type consistency |

<!-- panache-ignore-end -->

These are code-inspection estimates, not measured development timings.
An opt-in warmup policy should live outside proposal adaptation: even static samplers have warmup, and composite samplers should apply a policy once per ensemble iteration, not once per component.
A wrapper or explicit policy field can carry diagnostics; a callback that mutates the state must not bypass sample/buffer consistency.

A safe first implementation would:

1. Default to disabled and support reporting without intervention.
   Validate positive check intervals, sufficiently long windows, and a rescue cutoff strictly before the end of warmup; reserve a configurable settling interval.
2. Record actual MH decisions and separate movement statistics.
   For HMC, define what its trajectory acceptance statistic means rather than treating it as a binary `update_chain!` decision.
   Keep bounded history instead of saving all warmup draws.
3. Finish all threaded transitions before assessing/rescuing.
   Select donors from a snapshot using the supplied RNG, copy values without aliasing, and keep position buffers, log densities, views, and emitted samples consistent.
   Never clone RNGs.
4. Explicitly decide whether the archive is preserved or rebuilt.
   Preserving it is simpler and retains modes, but cannot immediately remove poor proposals; rebuilding must respect initialized capacity, minimum donor counts, refill/thinning counters, and diversity.
   Neither choice should be implicit.
5. Clear the rescued chain's diagnostic window and audit adaptive caches, subspace statistics, HMC metric assignments, and adaptation schedules.
   Initially reject unsupported HMC/temperature combinations rather than silently invalidating them.
6. Ensure ordinary `step`, resumption after warmup, and convergence stopping never relocate chains.
   Document retained warmup as non-posterior output.

## Evidence required before enabling intervention

Use reproducible synthetic targets generated in benchmark/test code, not committed data.
Compare rescue-disabled and rescue-enabled runs over multiple seeds under the same log-density evaluation budget, with both memory-based and memoryless samplers.
Include badly initialized unimodal targets, correlated targets, separated modes with known unequal weights, and correctly initialized controls.
Compare against longer warmup and improved initialization (including the existing Pathfinder integration).

Measure warmup recovery time, ESS per evaluation, posterior moment/quantile accuracy, mode weights, rescue frequency, and archive diversity.
Include threaded/sequential reproducibility, multiple position types, refill/growing/thinned archives, and proof that no interventions occur in retained sampling.
A rescue benefit on a unimodal target must not come at the cost of silently deleting a valid low-probability mode.

**Decision:** retain the existing convergence check and prioritize opt-in warmup reporting.
Prototype warmup-only rescue only if these comparisons show a material benefit for the default memory-based workflow.
Defer post-warmup rescue and automatic "copy the best chain" behavior.
