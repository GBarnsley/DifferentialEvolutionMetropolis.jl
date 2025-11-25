# DifferentialEvolutionMetropolis Documentation

Tools for sampling from log-densities using differential evolution algorithms.

See [Sampling from multimodal distributions](@ref) and [Customizing your sampler](@ref) to get started.

This package is built upon [AbstractMCMC.jl](https://turinglang.org/AbstractMCMC.jl) so log-densities should be constructed using that package, and can be used with [TransformVariables.jl](https://github.com/tpapp/TransformVariables.jl) or [Bijectors.jl](https://turinglang.org/Bijectors.jl) to control the parameter space.

The other key dependency is [Distributions.jl](https://juliastats.org/Distributions.jl). Almost every parameter in proposals given here are defined via customizable univariate distributions. Values that are fixed are specified via a [Dirac distribution](https://en.wikipedia.org/wiki/Dirac_delta_function), though in the API these can be specified with any real value. As a *warning* there are some checks on the given distributions, but in the interest of flexibility it is up to the user to ensure that they are suitable for the given parameter. You can disable any checking of your provided distributions with `; check_args = false` if you really want to ruin your sampler efficiency.
Distributions can optionally be used to define your log-density, as in the examples given here. 

As far as I am aware, there is one other package that implements differential evolution MCMC in Julia, [DifferentialEvolutionMCMC.jl](https://github.com/itsdfish/DifferentialEvolutionMCMC.jl/tree/master).
I opted to implement my own version as I wanted a more flexible API and the subsampling scheme from DREAM. That's not to discredit DifferentialEvolutionMCMC.jl, it has many features this package does not, such as being able to work on optimization problems and parameter blocking.

## Main features

- Original differential evolution, snooker, and adaptive subspace sampling (i.e. from DREAM) updates
- Optional parallel tempering (no swaps yet, information is shared by the DE updates!) and annealing
- Composite samplers, can combine any of the implemented updates (in future I'll wrap other abstractMCMC based samplers)
- Easy to implement your own updates!
- Can output in `MCMCChains` format, though you use multiple sampling chains (i.e. chains of the DE-chains) these will all be appended together

## Next Steps

A few plans for this package, feel free to suggest features or improvements via [issues](https://github.com/GBarnsley/DifferentialEvolutionMetropolis/issues):
- Implement multi-try and delayed rejection DREAM, I avoided these so far since I have been using these samplers for costly log-densities with relatively few parameters, such as one that solve an ODE.
- Additional diagnostic checks and adaptive schemes.


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

### Setup Functions

```@docs
setup_sampler_scheme
setup_de_update
setup_snooker_update
setup_subspace_sampling
```

### Core Sampling Functions

```@docs
DifferentialEvolutionMetropolis.step
DifferentialEvolutionMetropolis.step_warmup
DifferentialEvolutionMetropolis.fix_sampler
DifferentialEvolutionMetropolis.fix_sampler_state
```

### Convergence and Stopping Criteria

```@docs
r̂_stopping_criteria
```

### Output

The output format can be modified with `chain_type`, the supported options are `Chains` from [MCMCChains](https://turinglang.org/MCMCChains.jl/stable/), `VNChain` from [FlexiChains](https://github.com/penelopeysm/FlexiChains.jl), `Any` which returns the basic `DifferentialEvolutionMetropolis.DifferentialEvolutionSample`, and the default option `DifferentialEvolutionOutput`. If `save_final_state = true` the format will be `(sample::requested format, final_state)`. If run in parallel using `step(model, sampler, parallel_option, n_its, n_meta_chains; n_chains = n_chains)` the meta chains and DE chains will be merged into one dimension for both `Chains` and `DifferentialEvolutionOutput`, if the final state is saved it will be a vector of length `n_meta_chains` containing the final state for each.

Note that support for `FlexiChains` is a bit underutilized as the all samplers currently require all of your parameters to have one type.

```@docs
DifferentialEvolutionOutput
```

## Index

```@index
```
