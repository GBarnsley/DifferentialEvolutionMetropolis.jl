# Combining HMC with differential evolution

**Warning:** HMC integration is experimental and not recommended for routine use.

Hamiltonian Monte Carlo (HMC, and its self-tuning variant NUTS) uses gradients to move through the posterior, but a Hamiltonian trajectory will not cross a deep low-density barrier, so a lone HMC chain tends to stay in whichever mode it started in.
The differential evolution updates move each chain using differences between members of the population, which lets them hop between separated modes cheaply.
However, in high dimensions their *within*-mode mixing suffers.
The hard case for either method is therefore a high-dimensional posterior with separated modes.

In this situation it might help to combine the two: an HMC kernel on each chain for efficient gradient-guided local exploration, and a differential evolution update for moving between modes.
When `AdvancedHMC.jl` is loaded this package exposes [`setup_hmc_update`](@ref), which wraps an AdvancedHMC sampler so it can act as an update inside a [`setup_sampler_scheme`](@ref) scheme.

As discussed below, it's not that simple to combine both methods.
HMC requires a well-conditioned mass matrix to perform well, but there's no guarantee a mass matrix that works on one mode will work on another.

## High-dimensional bimodal target

We use a `D`-dimensional mixture of two separated Gaussians with **opposite correlation structure**.
Mode `A` has an AR(1) covariance with `ρ = +0.9` (neighbouring coordinates positively correlated).
Mode `B` has an AR(1) covariance with `ρ = −0.9` (neighbouring coordinates negatively correlated).
In a 2D example, the two modes are ellipses tilted in opposite directions, so a single mass matrix will fit neither well.
With a high `D`, the differential evolution proposals will mix slowly *within* a mode.

As with `Pathfinder`, HMC methods require gradients of the log-density.
Here we use `ForwardDiff` to calculate them.

```@example HMC
using DifferentialEvolutionMetropolis, AbstractMCMC
using AdvancedHMC, ForwardDiff, LogDensityProblemsAD, LogDensityProblems
using Distributions, Random, LinearAlgebra

struct CorrMixture{T <: Real, M <: AbstractMatrix{T}}
    w::T
    μ_A::Vector{T}
    μ_B::Vector{T}
    P_A::M
    P_B::M
    logdet_A::T
    logdet_B::T
end

# Modes A and B are centred at ±sep with AR(1) correlations +ρ and -ρ, mode A has weight w
function CorrMixture(D::Int; sep = 4.0, ρ = 0.9, w = 0.6)
    ar1(r) = Symmetric([r^abs(i - j) for i in 1:D, j in 1:D])
    Σ_A = ar1(ρ)
    Σ_B = ar1(-ρ)
    return CorrMixture(w, fill(sep, D), fill(-sep, D), inv(Σ_A), inv(Σ_B), logdet(Σ_A), logdet(Σ_B))
end

LogDensityProblems.dimension(target::CorrMixture) = length(target.μ_A)
LogDensityProblems.capabilities(::Type{<:CorrMixture}) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(target::CorrMixture, x)
    dp = x .- target.μ_A; dm = x .- target.μ_B
    lpP = log(target.w)     - 0.5 * dot(dp, target.P_A, dp) - 0.5 * target.logdet_A
    lpM = log(1 - target.w) - 0.5 * dot(dm, target.P_B, dm) - 0.5 * target.logdet_B
    c = max(lpP, lpM) #logsumexp
    return c + log(exp(lpP - c) + exp(lpM - c))      # mixture of the two Gaussians
end

D = 20                           # high enough that within-mode mixing needs HMC
target = CorrMixture(D)
model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, target))
```

## HMC updates

[`setup_hmc_update`](@ref) takes any AdvancedHMC sampler and decomposes it into the modular pieces (metric, integrator, kernel, adaptor) that the harness drives directly.
The simplest way to construct one is the `NUTS` wrapper.
`NUTS`/`HMC`/`HMCDA` store their metric as a *symbol* and only size it once the model is known, so here you must tell `setup_hmc_update` the number of parameters via `n_dims`:

```@example HMC
hmc_update = setup_hmc_update(NUTS(0.8); n_dims = D)
```

If you would rather assemble the kernel yourself, `setup_hmc_update` also accepts a fully-built `HMCSampler`.
Because that already carries a correctly-sized metric, `n_dims` is not needed:

```@example HMC
metric = DiagEuclideanMetric(D)
integrator = Leapfrog(0.1)
κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

custom_hmc_update = setup_hmc_update(HMCSampler(κ, metric, adaptor))
```

During warm-up `NUTS` tunes the step size and mass matrix, `HMCDA` tunes only the step size, and `HMC` tunes nothing.
A symbol integrator (the default) has its step size found on the first warm-up step, while an explicit step size is kept by `HMC` and is where `NUTS` and `HMCDA` start dual averaging, as in AdvancedHMC.
Without warm-up (`adapt = false` or `num_warmup = 0`) the update runs the sampler as built, so give it an explicit step size:

```@example HMC
fixed_hmc_update = setup_hmc_update(HMC(0.1, 10); n_dims = D)
fixed_nuts_update = setup_hmc_update(NUTS(0.8; integrator = Leapfrog(0.1)); n_dims = D)
```

**Warning:** The archive-based metric strategies described here, [`memory_metric`](@ref), [`cluster_pooled_metric`](@ref) and [`per_cluster_metric`](@ref), are **experimental**.
Their behaviour and interface may change in future releases, and their effectiveness depends on the target's mode structure (see the example below).

### Estimating the metric from the memory archive

By default the HMC update adapts its mass matrix across the whole population during warm-up.
Because this package also keeps a historical archive when `memory = true`, the metric can instead be estimated from that much larger pool.
Passing [`memory_metric`](@ref) as the `metric_strategy` replaces the adaptor with a covariance computed from the memory archive every `every` HMC steps rather than accumulated online.
As the archive can keep growing, the metric is recomputed throughout both warm-up and sampling.

Because it reads the archive, [`memory_metric`](@ref) requires `memory = true` and is incompatible with parallel tempering (because the archive includes hot-chain positions).

```@example HMC
mem_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = memory_metric()
)
```

The mass matrix maintains the shape it was initialized with, i.e. a diagonal mass matrix remains diagonal.
We advise using a dense mass matrix to take advantage of the large size of the memory archive.

```@example HMC
dense_mem_update = setup_hmc_update(
    NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = memory_metric()
)
```

### Multimodal targets: the cluster-pooled metric

[`memory_metric`](@ref) might struggle when the posterior has well-separated modes.
The archive spans every mode, so, depending on the log-density, its covariance can be dominated by the *between-mode* covariance rather than the *within-mode* covariance.

[`cluster_pooled_metric`](@ref) clusters the archive (into at most `kmax` clusters) and attempts to remove the between-mode variation while keeping the (shared) within-mode variation.

```@example HMC
cluster_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = cluster_pooled_metric()
)
```

Whilst clustering `every` few iterations is slow, the clustering process tests for multimodality first, and acts like the [`memory_metric`](@ref) if there is none.

### Per-mode metrics for modes of different shape

[`cluster_pooled_metric`](@ref) still pools the clusters into one mass matrix; if the modes also differ in covariance, HMC performance will still suffer within modes.
[`per_cluster_metric`](@ref) uses the separate cluster covariances, ideally one mass matrix per mode, and assigns each chain to a cluster based on its current position.

```@example HMC
per_cluster_hmc_update = setup_hmc_update(
    NUTS(0.8); n_dims = D, metric_strategy = per_cluster_metric()
)
```

Each chain is relabelled to its nearest cluster before every HMC update, using its current position.
So a chain that has moved modes will use the correct mass matrix.
If modes are **well-separated**, this should only occur after differential evolution updates.
In **close or connected** modes, HMC trajectories can move between modes, which breaks detailed balance since the new mode won't have an identical mass matrix and this difference is **currently** not accounted for.

## Seeding the memory and sampling

```@example HMC
# 3 starting chains followed by a memory archive seeded 50/50 across the two modes.
function seeded_positions(rng, target; n_mem = 60)
    Lp = cholesky(inv(target.P_A)).L
    Lm = cholesky(inv(target.P_B)).L
    plus  = [target.μ_A .+ Lp * randn(rng, D) for _ in 1:(n_mem ÷ 2)]
    minus = [target.μ_B .+ Lm * randn(rng, D) for _ in 1:(n_mem ÷ 2)]
    chains = [plus[1], minus[1], plus[2]]               # the 3 chains span both modes
    return vcat(chains, plus[3:end], minus[2:end])      # the rest seed the memory
end

scheme = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = D, metric_strategy = per_cluster_metric()),
    setup_subspace_sampling(),                  # local population moves
    setup_subspace_sampling(γ = 1.0, cr = 1.0); # full-dimensional mode jump
    w = [0.5, 0.3, 0.2],
)

out = sample(
    Random.Xoshiro(1), model, scheme, 5000;
    n_chains = 3, num_warmup = 2000,
    initial_position = seeded_positions(Random.Xoshiro(11), target), N₀ = 60,
    memory = true, progress = false, chain_type = DEMOutput,
)
```

## Posterior exploration

Both modes are recovered, and each mode's *distinct* correlation is recovered:

```@example HMC
using Statistics

draws = reshape(out.samples, :, D)
in_plus = vec(sum(draws; dims = 2)) .> 0

println("P(mode A)            = ", round(mean(in_plus); digits = 2), "   (true ", target.w, ")")
println("corr(x1, x2)  mode A = ", round(cor(draws[in_plus, 1],  draws[in_plus, 2]);  digits = 2), "   (true +0.9)")
println("corr(x1, x2)  mode B = ", round(cor(draws[.!in_plus, 1], draws[.!in_plus, 2]); digits = 2), "   (true −0.9)")
```

A scatter of the first two coordinates shows the two modes recovered with their opposite tilts:

```@example HMC
using Plots

scatter(draws[in_plus, 1], draws[in_plus, 2];
    label = "mode A", ms = 2, alpha = 0.3, color = :steelblue)
scatter!(draws[.!in_plus, 1], draws[.!in_plus, 2];
    label = "mode B", ms = 2, alpha = 0.3, color = :firebrick)
plot!(xlabel = "x₁", ylabel = "x₂", title = "Two $(D)-D modes with opposite correlation")
```

Recovering the mode *weights* is still hard: the differential evolution update (in this example) has to update all dimensions at once since both modes are completely separated (hence `cr = 1.0`).

In general, if you find yourself with a posterior that is multimodal, high-dimensional and has pathological geometry, it's probably best to reparameterize your model.

### HMC Setup Functions

```@docs
setup_hmc_update
memory_metric
cluster_pooled_metric
per_cluster_metric
```
