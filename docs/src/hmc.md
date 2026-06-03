# Combining HMC with differential evolution

Hamiltonian Monte Carlo (HMC, and its self-tuning variant NUTS) uses gradients to glide
through awkward local geometry, but a single Hamiltonian trajectory will not cross a deep
low-density barrier, so a lone HMC chain tends to stay in whichever mode it started in.

The differential evolution updates in this package have a complementary strength. They move
each chain using *differences* between members of the population, which lets them hop between
modes cheaply — but only when those modes look alike up to a shift. If two modes live at very
different *scales*, a difference vector drawn from one will be the wrong size for the other,
and the population cannot move between them correctly. A funnel is the textbook example: its
"mouth" is broad and its "neck" is extremely tight, so no single step size fits both.

This is exactly the situation where it pays to combine the two: run an HMC kernel on each
chain for gradient-guided local exploration, and a [`setup_subspace_sampling`](@ref)
(DREAM-z) update for population moves. When `AdvancedHMC.jl` is loaded this package exposes
[`setup_hmc_update`](@ref), which wraps an AdvancedHMC sampler so it can act as one update
inside a [`setup_sampler_scheme`](@ref) composite, sharing a single, population-pooled metric
across all of the chains.

## A bimodal-scale funnel

We use Neal's funnel with a twist: the log-scale parameter `v` is drawn from a
**two-component mixture**, so the funnel has two mouths of very different width, and the
remaining coordinates are Gaussian with standard deviation `exp(v/2)`. Because we know the mixture exactly, the marginal posterior of `v` is a known
quantity we can hold each sampler up against.

HMC needs gradients, so the log density must expose them. We write it against the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface and wrap it
with [LogDensityProblemsAD](https://github.com/tpapp/LogDensityProblemsAD.jl) so that
`ForwardDiff` supplies the gradient.

```@example HMC
using DifferentialEvolutionMetropolis, AbstractMCMC
using AdvancedHMC, ForwardDiff, LogDensityProblemsAD, LogDensityProblems
using Distributions, Random

const D = 21          # one log-scale parameter v plus D-1 funnel coordinates
const A = 3.0         # the two scale modes sit at v ≈ ±A
const SV = 1.0        # width of each scale mode
const P_WIDE = 0.65   # weight of the wide mode (v ≈ +A)

struct BimodalFunnel end
LogDensityProblems.dimension(::BimodalFunnel) = D
LogDensityProblems.capabilities(::Type{BimodalFunnel}) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(::BimodalFunnel, x)
    v = x[1]
    lpR = log(P_WIDE)     + logpdf(Normal(A, SV), v)   # wide mouth: scale exp(A/2)
    lpL = log(1 - P_WIDE) + logpdf(Normal(-A, SV), v)  # narrow neck: scale exp(-A/2)
    m = max(lpR, lpL)
    lp = m + log(exp(lpR - m) + exp(lpL - m))          # bimodal prior on v
    s = exp(v / 2)
    @inbounds for i in 2:D
        lp += -log(s) - 0.5 * (x[i] / s)^2             # Normal(0, s) on each funnel coordinate
    end
    return lp
end

model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, BimodalFunnel()))
```

The two scale modes differ by a factor of `exp(A) ≈ 20` in width, so moving a chain from one
to the other means contracting (or expanding) all `D-1` funnel coordinates at once — the move
that difference-based proposals cannot make on their own.

## Building the HMC update

[`setup_hmc_update`](@ref) takes any AdvancedHMC sampler and decomposes it into the modular
pieces (metric, integrator, kernel, adaptor) that the harness drives directly. The simplest
way to construct one is the `NUTS` wrapper. `NUTS`/`HMC`/`HMCDA` store their metric as a
*symbol* and only size it once the model is known, so here you must tell `setup_hmc_update`
the number of parameters via `n_dims`:

```@example HMC
hmc_update = setup_hmc_update(NUTS(0.8); n_dims = D)
nothing # hide
```

If you would rather assemble the kernel yourself, `setup_hmc_update` also accepts a
fully-built `HMCSampler`. Because that already carries a correctly-sized metric, `n_dims` is
not needed:

```@example HMC
metric = DiagEuclideanMetric(D)
integrator = Leapfrog(0.1)
κ = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn(10, 1000.0)))
adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))

custom_hmc_update = setup_hmc_update(HMCSampler(κ, metric, adaptor))
nothing # hide
```

Both styles produce the same kind of update; the comparison below uses the `NUTS()` form.

## The three samplers

We compare the HMC update on its own, the subspace update on its own, and the composite that
weights the two equally. All three are ordinary [`setup_sampler_scheme`](@ref) objects.

```@example HMC
nuts_only = setup_sampler_scheme(setup_hmc_update(NUTS(0.8); n_dims = D))
subspace_only = setup_sampler_scheme(setup_subspace_sampling())
composite = setup_sampler_scheme(
    setup_hmc_update(NUTS(0.8); n_dims = D),
    setup_subspace_sampling();
    w = [0.5, 0.5]
)
nothing # hide
```

We seed half of the chains in each scale mode so that every sampler starts with both modes
represented; the question is whether it can then *maintain* the correct balance between them.

```@example HMC
const N_CHAINS = 24

function init_positions(rng)
    map(1:N_CHAINS) do i
        s = iseven(i) ? 1.0 : -1.0           # half the chains in each scale mode
        v = s * A + SV * randn(rng)
        vcat(v, exp(v / 2) .* randn(rng, D - 1))
    end
end
nothing # hide
```

## Recovering the scale parameter

For each scheme we report the posterior mean of `v`, the recovered weight of the wide mode
`P(v > 0)` (whose true value is `0.65`), the effective sample size of `v` from
[MCMCDiagnosticTools](https://turinglang.org/MCMCDiagnosticTools.jl), and the wall-clock time.
A short warm-up call is made first so the timing reflects sampling rather than compilation.

```@example HMC
using MCMCDiagnosticTools, Statistics

function evaluate(scheme; iters = 2500, warm = 800, seed = 11)
    init = init_positions(Random.Xoshiro(seed + 1))
    runit(n) = sample(
        Random.Xoshiro(seed), model, scheme, n;
        n_chains = N_CHAINS, num_warmup = warm, initial_position = init,
        memory = true, progress = false, chain_type = DifferentialEvolutionOutput
    )
    runit(20)                       # trigger compilation, untimed
    time = @elapsed out = runit(iters)
    v = vec(out.samples[:, :, 1])
    return (; v, time, mean_v = mean(v),
        p_wide = mean(v .> 0),                  # truth is P_WIDE = 0.65
        ess_v = ess(out.samples[:, :, 1:1])[1])
end

results = (
    NUTS = evaluate(nuts_only),
    subspace = evaluate(subspace_only),
    composite = evaluate(composite),
)

true_mean = P_WIDE * A + (1 - P_WIDE) * (-A)
println("true E[v] = ", round(true_mean; digits = 2), ",  true P(v>0) = ", P_WIDE, "\n")
println(rpad("sampler", 12), rpad("E[v]", 9), rpad("P(v>0)", 10), rpad("ESS(v)", 9), "time (s)")
for name in (:NUTS, :subspace, :composite)
    r = results[name]
    println(
        rpad(string(name), 12),
        rpad(round(r.mean_v; digits = 2), 9),
        rpad(round(r.p_wide; digits = 3), 10),
        rpad(round(Int, r.ess_v), 9),
        round(r.time; digits = 1),
    )
end
```

Because the target is constructed we can judge each sampler against the known marginal rather
than leaning on a convergence diagnostic. **Subspace alone** gets the scale weights wrong: it
over-populates the narrow neck and reports `P(v > 0)` well below the true `0.65`, because once
a chain's coordinates contract into the neck the difference proposals are too small to climb
back out — the population cannot rebalance the modes. **NUTS alone** has the opposite problem:
it concentrates on the wide mouth and barely samples the neck (a single global step size
cannot serve both), so its mean is biased high, and it is also by far the slowest because the
funnel forces very deep trajectories. The **composite** recovers both the mean and the mode
weight: the population moves carry chains between the two mouths while the HMC kernel follows
the funnel's curvature within each, at a fraction of NUTS's cost.

## Comparing the scale-parameter posteriors

Plotting the recovered marginal of `v` against the truth makes the difference plain. The
composite tracks the true bimodal density; subspace misplaces the mass between the modes; and
NUTS struggles to populate the narrow neck at all.

```@example HMC
using Plots

vgrid = range(-9, 8; length = 400)
true_density = @. P_WIDE * pdf(Normal(A, SV), vgrid) + (1 - P_WIDE) * pdf(Normal(-A, SV), vgrid)

plt = plot(
    vgrid, true_density;
    lw = 3, color = :black, label = "truth",
    xlabel = "v  (log scale)", ylabel = "density",
    title = "Posterior of the scale parameter",
)
for (name, c) in zip((:NUTS, :subspace, :composite), (:orange, :red, :green))
    stephist!(plt, results[name].v; normalize = :pdf, lw = 2, color = c, label = string(name))
end
plt
```

On this funnel the composite is the clear winner, but the broader lesson is about matching the
sampler to the geometry. Differential evolution moves excel when modes are alike up to a
shift, and gradient information is what lets a sampler follow a scale that changes across the
space; the HMC update lets you bring both to bear in a single scheme. As always, benchmark on
your own posterior — gradients are not free, and on smoother targets the population moves may
already be enough.
