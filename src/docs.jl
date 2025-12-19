_generic_de_kwargs_1 = """
## Generic DE Arguments
- `n_chains`: Number of parallel chains. Defaults to `max(2 * dimension, 3)` for adequate mixing.
- `adapt`: Whether to enable adaptive behavior during warm-up (if the sampler supports it).
  Defaults to `true`.
- `initial_position`: Starting positions for chains. Can be `nothing` (random initialization),
  or a vector of parameter vectors. If the provided vector is smaller than `n_chains + n_hot_chains`,
  it will be expanded; if larger and `memory=true`, excess positions become initial memory. Defaults to `nothing`.
- `parallel`: Whether to evaluate initial log-densities in parallel. Useful for expensive models.
  Defaults to `false`.
- `n_preallocated_indices`: This package provides fast sampling-without-replacement by pre-allocating indices, defaults to 3 (which the most asked for by the implemented samplers). Consider increasing it if you implement your own proposal that calls `pick_chains` with `n_chains > 3`.
- `silent`: Suppress informational logging during initialization (e.g., initial position adjustments and
    memory setup) when `true`. Defaults to `false`.
## Memory-based Sampling Arguments
"""
_generic_de_kwargs_2 = """
- `N₀`: Initial memory size for memory-based samplers. Should be ≥ `n_chains + n_hot_chains`.
  Defaults to `2 * n_chains + n_hot_chains`.
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers). Defaults to `true`. Overwrites memory options given at initialization, generally should only be of use if calling `step` directly.
- `memory_refill`: Whether to refill memory when full instead of extending the memory, will replace from the start. Defaults to `false`.
- `memory_size`: Maximum number of positions preallocated per chain in memory. The effective number stored positions is `memory_size * (n_chains + n_hot_chains)`. Defaults to `1001` or `2*num_warmup` if that is provided here or via `sample`. If `memory_refill = true` this is the maximum number stored before refilling, if  `memory_refill = false` once the memory is full, the array is extended by another `memory_size` worth of positions. Set with consideration of available RAM and expected run length.
- `memory_thin_interval`: Thinning interval for memory updates. If > 0, only every `memory_thin_interval`-th
  position is stored in memory.
## Parallel Tempering and Simulated Annealing Arguments
- `n_hot_chains`: Number of hot chains for parallel tempering. Defaults to 0 (no parallel tempering).
- `max_temp_pt`: Maximum temperature for parallel tempering. Defaults to 2*sqrt(dimension).
- `max_temp_sa`: Maximum temperature for simulated annealing. Defaults to `max_temp_pt`.
- `α`: Temperature ladder spacing parameter. Controls the geometric spacing between temperatures.
  Defaults to 1.0.
- `annealing`: Whether to use simulated annealing (temperature decreases over time). Defaults to `false`.
- `annealing_steps`: Number of annealing steps. Defaults to `annealing ? num_warmup : 0`.
- `temperature_ladder`: Pre-defined temperature ladder as a vector of vectors. If provided,
  overrides automatic temperature ladder creation. Defaults to `create_temperature_ladder(n_chains, n_hot_chains, α, max_temp_pt, max_temp_sa, annealing_steps)`.
"""
generic_de_kwargs = """
$(_generic_de_kwargs_1)
- `memory`: Whether to use memory-based sampling that stores past positions. Memory-based
  samplers can be more efficient for high-dimensional problems. Defaults to `true`.
$(_generic_de_kwargs_2)
"""
generic_de_kwargs_no_mem = """
$(_generic_de_kwargs_1)
- `memory`: Whether to use memory-based sampling that stores past positions. Memory-based
  samplers can be more efficient for high-dimensional problems. Defaults to `false`.
$(_generic_de_kwargs_2)
"""
abstract_mcmc_kwargs = """
- `kwargs...`: Additional keyword arguments passed to `AbstractMCMC.sample` (e.g., `memory_refill`, `memory_thin_interval`, `silent`). See [AbstractMCMC documentation](https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments).
"""

generic_notes = """
- For non-memory samplers, `n_chains` should typically be ≥ dimension for good mixing
- Memory-based samplers can work effectively with fewer chains than the problem dimension
- Initial log-densities are computed automatically for all starting positions
- When using parallel tempering (`n_hot_chains > 0`), only the cold chains (first `n_chains`)
  are returned in the sample, but all chains participate in the sampling process
- Memory-based samplers with parallel tempering will issue warnings since hot chains typically
  aren't necessary when using memory
"""

template_chains_kwargs = """
- `chain_type`: Type of chain to return (e.g., `Any`, `DifferentialEvolutionOutput`, `MCMCChains.Chains`, or `FlexiChains.VNChain`). Defaults to `DifferentialEvolutionOutput`.
- `save_final_state`: Whether to return the final state along with samples, if true the output will be (samples::chain_type, final_state). Defaults to `false`.
"""

deMC_description = """
Run the Differential Evolution Markov Chain (DE-MC) sampler proposed by ter Braak (2006).

This sampler uses differential evolution updates with optional switching between two
scaling factors (`γ₁` and `γ₂`) to enable mode switching.

This implementation varies slightly from the original: updates within a population
occur based on the previous positions to enable easy parallelization.

See doi.org/10.1007/s11222-006-8769-1 for more information.
"""
deMC_kwargs = """
- `save_burnt`: Save burn-in samples in output. Defaults to `false`.
- `γ₁`: Primary scaling factor. Defaults to `2.38 / sqrt(2 * dim)`.
- `γ₂`: Secondary scaling factor for mode switching. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.1.
"""

deMCzs_description = """
Run the Differential Evolution Markov Chain with snooker update and historic sampling (DE-MCzs) sampler.

It combines DE updates with optional snooker moves and uses memory-based
sampling to efficiently handle high-dimensional problems with fewer chains.

Proposed by ter Braak and Vrugt (2008), see doi.org/10.1007/s11222-008-9104-9.
"""
deMCzs_kwargs = """
- `γ`: Scaling factor for DE updates. Defaults to `2.38 / sqrt(2 * dim)`.
- `γₛ`: Scaling factor for snooker updates. Defaults to `2.38 / sqrt(2)`.
- `p_snooker`: Probability of snooker moves. Defaults to 0.1.
- `β`: Noise distribution for DE updates. Defaults to `Uniform(-1e-4, 1e-4)`.
"""

DREAMz_description = """
Run the Differential Evolution Adaptive Metropolis (DREAMz) sampler.

This advanced adaptive sampler uses subspace sampling with adaptive crossover probabilities. It can switch between scaling factors and includes
outlier chain detection/replacement. The algorithm adapts during warm-up and can use
memory-based sampling for efficiency.

Based on Vrugt et al. (2009), see doi.org/10.1515/IJNSNS.2009.10.3.273.
"""
DREAMz_kwargs = """
- `γ₁`: Primary scaling factor for subspace updates. Defaults to adaptive.
- `γ₂`: Secondary scaling factor. Defaults to 1.0.
- `p_γ₂`: Probability of using `γ₂`. Defaults to 0.2.
- `n_cr`: Number of crossover probabilities for adaptation. Defaults to 3.
- `cr₁`: Crossover probability for `γ₁`. Defaults to adaptive.
- `cr₂`: Crossover probability for `γ₂`. Defaults to adaptive.
- `ϵ`: Additive noise distribution. Defaults to `Uniform(-1e-4, 1e-4)`.
- `e`: Multiplicative noise distribution. Defaults to `Normal(0.0, 1e-2)`.
"""
