# DifferentialEvolutionMetropolis Documentation

Tools for sampling from log-densities using differential evolution algorithms.

See [Sampling from multimodal distributions](@ref) and [Customizing your sampler](@ref) to get started.

This package is built upon [AbstractMCMC.jl](https://turinglang.org/AbstractMCMC.jl) so log-densities should be constructed using that package, and can be used with [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) or [Bijectors.jl](https://turinglang.org/Bijectors.jl) to control the parameter space.

The other key dependency is [Distributions.jl](https://juliastats.org/Distributions.jl).
Almost every parameter in proposals given here is defined via customizable univariate distributions.
Values that are fixed are specified via a [Dirac distribution](https://en.wikipedia.org/wiki/Dirac_delta_function), though in the API these can be specified with any real value.
There are some checks on the given distributions, but in the interest of flexibility it is up to the user to ensure that they are suitable for the given parameter.
You can disable any checking of your provided distributions with `; check_args = false` if you really want to ruin your sampler efficiency.

Alternative Julia packages for differential evolution and ensemble MCMC include [DifferentialEvolutionMCMC.jl](https://github.com/itsdfish/DifferentialEvolutionMCMC.jl) and [EnsembleMCMC.jl](https://github.com/JuliaBayes/EnsembleMCMC.jl).

## Main features

- Original differential evolution, snooker, and adaptive subspace sampling (i.e. from DREAM) updates
- Optional parallel tempering (with swaps) and annealing
- Composite samplers, can combine any of the implemented updates
- Easy to implement your own updates!
- Can output in `MCMCChains` or `FlexiChains` format, though if you use multiple sampling chains (i.e. chains of the DE-chains) these will all be appended together
- Accepts `Pathfinder` results to determine initial positions, and, if passing a `multipathfinder` result, the package will ensure that each mode is represented in the initial chain positions or memory (given sufficient chains or initial memory positions)
- Experimental support for `AdvancedHMC` samplers

## Supported position types

This package currently supports log-densities using `Float64`, `Float32`, and `BigFloat`.
Pass `T = Float32` or `T = BigFloat` to `step`, `sample`, `deMC`, `deMCzs`, or `DREAMz` to choose the position and output type.
Any supplied positions and initial memory draws are converted to type `T`.
If `T` is not provided, the type is determined by `initial_position` or defaults to `Float64` if no initial positions are supplied.
By default update parameters and distributions remain `Float64` and are converted when used with typed positions.
If using a type other than `Float64`, you are encouraged to create custom updates that use the correct types, for example `setup_de_update(γ = 0.5f0, β = Normal(0.0f0, 1.0f-6))`.
The precision and draw-type behavior of all distributions depend on Distributions.jl and are not directly supported in this package.

## Next Steps

A few plans for this package, feel free to suggest features or improvements via [issues](https://github.com/GBarnsley/DifferentialEvolutionMetropolis/issues):

- Diagnostics accounting for inter-chain dependence
- Double ensemble mode, an alternative for multi-try that allows parallel evaluations when `memory = false`
- Implement delayed rejection DREAM
- Kalman filtering for warmup (requires additions to the LD interface)
- Investigate implementations with Gibbs/particle-filtering to support the sampling of discrete parameters
- Investigate implementations alongside particle-gibbs/rj-MCMC with changing problem dimensions

## Contents

```@contents
```

## Functions

### Implemented Sampling Schemes

```@docs
deMC
deMCzs
DREAMz
```

### Updates

```@docs
setup_de_update
setup_snooker_update
setup_subspace_sampling
setup_swap_update
```

### Composite updates

```@docs
setup_sampler_scheme
setup_random_order_scheme
setup_fixed_order_scheme
```

### Core Sampling Functions

```@docs
DifferentialEvolutionMetropolis.step
DifferentialEvolutionMetropolis.step_warmup
```

### Convergence and Stopping Criteria

```@docs
r̂_stopping_criteria
```

### Output

The output format can be modified with `chain_type`; the supported options are `Chains` from [MCMCChains](https://turinglang.org/MCMCChains.jl/stable/), `VNChain` from [FlexiChains](https://github.com/penelopeysm/FlexiChains.jl), `Any` which returns the basic `DifferentialEvolutionMetropolis.DEMSample`, and the default option `DEMOutput`.
If `save_final_state = true` the format will be `(sample::requested format, final_state)`.
If run in parallel using `sample(model, sampler, parallel_option, n_its, n_meta_chains; n_chains = n_chains)` the meta chains and DE chains will be merged into one dimension for the `Chains`, `VNChain`, and `DEMOutput`.
If the final state is saved it will be a vector of length `n_meta_chains` containing the final state for each.

Note that support for `FlexiChains` is a bit underutilized as all samplers currently require all parameters to have the same type.

```@docs
DEMOutput
```

## Index

```@index
```
