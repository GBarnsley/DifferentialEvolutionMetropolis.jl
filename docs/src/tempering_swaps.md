# Position swaps with parallel tempering

## Answer

Explicit replica exchange can help even though differential-evolution proposals already use information from other temperatures.
They are different mechanisms: DE transfers a *difference vector*, whereas exchange offers a whole explored position to a colder chain.
In the controlled experiment below, exchange improved cold-chain mode-occupancy error on separated mixtures.
This is evidence for an optional exchange kernel, not for enabling swaps by default on every model.

The investigation is deliberately outside the production sampler.
Run it with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmark/tempering_swaps.jl
```

No generated sample files are needed or written.

## What the current code does

With `memory=false`, `pick_chains` in `src/chains.jl` draws donor positions from all live chains, regardless of temperature.
`proposal!` in `src/differential_evolution_update.jl` adds their scaled difference to the current position.
Hot chains can therefore influence a cold proposal, but are not moved into the cold rung.
Acceptance uses the destination rung's temperature.

With memory enabled, donors come from the archive instead.
The warning about memory-based samplers with hot chains is important: the archive is another information path, not replica exchange.
This experiment uses no archive and no adaptation; it does not establish a benefit for deMCzs, DREAMz or HMC.

The existing `test/test_swapping.jl` checks swaps of current/proposed *buffers*.
That is unrelated to exchanging positions between temperature rungs.

## Exchange kernel

For the package's convention, the target at temperature `T` is proportional to `exp(ld(x)/T)`.
Swapping positions `xᵢ` and `xⱼ` has log acceptance ratio

```math
\log r = (1/T_i - 1/T_j)\,[\operatorname{ld}(x_j)-\operatorname{ld}(x_i)].
```

Accept if `log(u) < min(0, log r)`.
A symmetric, position-independent selection of pairs makes this a detailed-balance kernel for the product of tempered targets.
This statement concerns the exchange kernel, not a new proof of correctness for the existing ensemble update.

The harness alternates disjoint odd/even neighbouring-rung matchings after each DE sweep.
It chooses the cold participant uniformly among all cold chains, so no single cold chain monopolises the boundary.
Only positions and their cached log densities move.
Temperatures, chain models and RNGs remain attached to rungs.
No additional model evaluations are needed.
Cold statistics are read from the post-exchange live state, not the sample returned before the exchange.

## Experiment

Targets are an isotropic standard Gaussian control and an equal-weight mixture of unit-covariance Gaussians centred at `±6` in coordinate one.
The other coordinates are independent standard Gaussians.
The exact positive-mode probability is `0.5`, and the exact first-coordinate second moment is `1 + 6²` (`1` for the control).

Each treatment has 32 seeds, 2,000 discarded sweeps and 6,000 retained sweeps.
There are `max(4, 2d)` cold chains and either 4 or 12 hot chains, geometrically spaced from 1.5 to 16.
Every replica starts near the negative mode, rather than starting with both modes already populated.
Initial positions and local-update RNG streams are matched between treatments; exchanges use a separate RNG.
DE uses the fixed scale `2.38 / sqrt(2d)` and the default small proposal noise.
Serial execution avoids conflating exchange with threading.

Occupancy RMSE is calculated across the 32 independent runs, using each run's pooled cold-chain mean.
It includes both finite-run bias and variability.
Cold chains are interacting, so they are **not** treated as independent replicates.
Second-moment RMSE provides a separate check; transition rates alone can increase merely because positions are relabelled and are not ESS estimates.

Measured on Julia 1.13.1 (serial), 2026-10-05.
Lower RMSE is better.

```@raw html
<table>
<thead><tr><th>Target</th><th>Hot chains</th><th>Swaps</th><th>Occupancy RMSE</th><th>Second-moment RMSE</th><th>Transitions / cold chain / sweep</th><th>Overall swap acceptance</th><th>Cold-boundary acceptance</th></tr></thead>
<tbody>
<tr><td>Gaussian, d=2</td><td>4</td><td>No</td><td>.01052</td><td>.03343</td><td>.07853</td><td>—</td><td>—</td></tr>
<tr><td>Gaussian, d=2</td><td>4</td><td>Yes</td><td>.01081</td><td>.02644</td><td>.12074</td><td>.669</td><td>.799</td></tr>
<tr><td>Mixture, d=2</td><td>4</td><td>No</td><td>.02075</td><td>.26335</td><td>.03500</td><td>—</td><td>—</td></tr>
<tr><td>Mixture, d=2</td><td>4</td><td>Yes</td><td>.01589</td><td>.29587</td><td>.08092</td><td>.678</td><td>.799</td></tr>
<tr><td>Mixture, d=10</td><td>4</td><td>No</td><td>.10217</td><td>.30845</td><td>.00090</td><td>—</td><td>—</td></tr>
<tr><td>Mixture, d=10</td><td>4</td><td>Yes</td><td>.06486</td><td>.23932</td><td>.00731</td><td>.309</td><td>.528</td></tr>
<tr><td>Mixture, d=10</td><td>12</td><td>No</td><td>.07592</td><td>.29365</td><td>.00084</td><td>—</td><td>—</td></tr>
<tr><td>Mixture, d=10</td><td>12</td><td>Yes</td><td>.03810</td><td>.30436</td><td>.00749</td><td>.726</td><td>.534</td></tr>
</tbody>
</table>
```

In these finite runs, occupancy RMSE fell by approximately 23%, 37% and 50% for the three mixture configurations.
The Gaussian control had essentially unchanged occupancy error.
Second-moment error did **not** improve uniformly.
These are descriptive comparisons, not significance tests or universal speedups.
The 12-hot-chain treatment also costs more local model evaluations than the 4-hot-chain treatment, so compare swaps on/off *within* a row pair.

After compilation, mean wall time per run was 0.022–0.025 seconds for the 2-dimensional mixtures and 0.068–0.089 seconds for the 10-dimensional mixtures; the exchange treatment added roughly 5–14% within configurations on this machine.
These very short timings include diagnostic allocations and are not reliable production performance benchmarks.
The exchange itself uses cached densities.

## Recommendation and limits

- Investigate an **opt-in** exchange stage: the benefit is not redundant with DE donor sharing on these separated mixtures.
- Tune the ladder using **edge-specific** acceptance and mode transport.
  More hot rungs improve overlap between hot neighbours, but do not change the cold-to-1.5 boundary.
  A high overall acceptance rate can hide a bottleneck.
- Do not choose a default from this small target family.
  Compare further targets, longer runs, multiple swap frequencies, memory-based schemes and observables beyond mode occupancy.
  Measure ESS per model evaluation and per second using diagnostics that account for interacting chains.
- A production implementation must exchange before creating the returned sample and updating the archive.
  If threading is enabled, exchange only after all local proposals finish.
  Preserve current/proposed buffer views, cached log densities and rung-specific adaptation.
  Model copies must represent the same target; mutable position-specific model caches need invalidation or recomputation.
- Annealing is a changing-target schedule, not stationary parallel tempering; this experiment uses a fixed ladder and does not validate annealing exchange.

The script's nine deterministic checks cover the acceptance-ratio sign, equal-temperature exchange, cached densities, cold-view aliases and a subsequent DE step.
The experimental helper is not exported and does not change existing APIs.
