using BenchmarkTools
using AbstractMCMC, DifferentialEvolutionMetropolis, Distributions, LogDensityProblems, Random
using AdvancedHMC, ForwardDiff, LogDensityProblemsAD
const SUITE = BenchmarkGroup()

#simple ld
struct IsotropicNormalModel{M <: AbstractVector{<:Real}}
    "mean of the isotropic Gaussian"
    mean::M
end
function LogDensityProblems.dimension(model::IsotropicNormalModel{<:AbstractVector{<:Real}})
    return length(model.mean)
end
function LogDensityProblems.logdensity(model::IsotropicNormalModel, x::AbstractVector{<:Real})
    return - sum(abs2, x .- model.mean) / 2
end
function LogDensityProblems.capabilities(model::IsotropicNormalModel)
    return LogDensityProblems.LogDensityOrder{0}()
end
am_model = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(5)))

#initial
initial_position = [zeros(5), ones(5), 2ones(5), 3ones(5)]
n_chains = size(initial_position, 1)
initial_position_with_memory = vcat(initial_position, initial_position)
N₀ = size(initial_position_with_memory, 1) - n_chains

#define updates
de_update = setup_de_update()
snooker_update = setup_snooker_update()
subspace_update = setup_subspace_sampling()
updates = (de_update, snooker_update, subspace_update)
names = ("de_update", "snooker_update", "subspace_update")

rng = Xoshiro(1234)

#initial steps
__,
    initial_state = AbstractMCMC.step(
    rng, am_model, de_update; memory = false,
    initial_position = initial_position, n_chains = n_chains
)
__,
    initial_state_memory = AbstractMCMC.step(
    rng, am_model, de_update; memory = true,
    initial_position = initial_position_with_memory, n_chains = n_chains, N₀ = N₀
)
__,
    initial_state_adaptive = AbstractMCMC.step(
    rng, am_model, subspace_update; memory = false,
    initial_position = initial_position, n_chains = n_chains, adapt = true
)
__,
    initial_state_pt_and_annealing = AbstractMCMC.step(
    rng, am_model, de_update; memory = false, initial_position = initial_position,
    n_chains = n_chains, n_hot_chains = 10, annealing_steps = 5
)

#HMC update: needs a gradient, so wrap the same density with ForwardDiff
hmc_model = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, IsotropicNormalModel(zeros(5))))
hmc_update = setup_hmc_update(NUTS(0.8); n_dims = 5)
__,
    hmc_initial_state = AbstractMCMC.step(
    rng, hmc_model, hmc_update; memory = false,
    initial_position = initial_position, n_chains = n_chains
)
__,
    hmc_initial_state_memory = AbstractMCMC.step(
    rng, hmc_model, hmc_update; memory = true,
    initial_position = initial_position_with_memory, n_chains = n_chains, N₀ = N₀
)

#HMC with the population/memory covariance metric: same kernel, but the mass matrix is
#estimated from the sampler's own population instead of AdvancedHMC's windowed adaptor.
#The metric is frozen after warm-up, so the post-warmup `step` cost matches `hmc_update`;
#the extra work (a covariance pass over the population) lives in `step_warmup` below.
hmc_population_update = setup_hmc_update(NUTS(0.8); n_dims = 5, metric_strategy = population_metric())
__,
    hmc_population_initial_state = AbstractMCMC.step(
    rng, hmc_model, hmc_population_update; memory = false,
    initial_position = initial_position, n_chains = n_chains
)
__,
    hmc_population_initial_state_memory = AbstractMCMC.step(
    rng, hmc_model, hmc_population_update; memory = true,
    initial_position = initial_position_with_memory, n_chains = n_chains, N₀ = N₀
)

#Dense population metric: estimates the full D×D covariance (correlations) from the archive,
#the costliest warm-up path (the covariance pass is O(n·D²)).
hmc_dense_population_update = setup_hmc_update(
    NUTS(0.8; metric = :dense); n_dims = 5, metric_strategy = population_metric()
)
__,
    hmc_dense_population_initial_state_memory = AbstractMCMC.step(
    rng, hmc_model, hmc_dense_population_update; memory = true,
    initial_position = initial_position_with_memory, n_chains = n_chains, N₀ = N₀
)

SUITE["MemoryLess"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["MemoryLess"][name] = @benchmarkable(
        AbstractMCMC.step(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state))
    )
end
SUITE["MemoryLess"]["hmc_update"] = @benchmarkable(
    AbstractMCMC.step(rng, $hmc_model, $hmc_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_initial_state))
)
SUITE["MemoryLess"]["hmc_population_update"] = @benchmarkable(
    AbstractMCMC.step(rng, $hmc_model, $hmc_population_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_population_initial_state))
)

SUITE["Memory"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["Memory"][name] = @benchmarkable(
        AbstractMCMC.step(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_memory))
    )
end
SUITE["Memory"]["hmc_update"] = @benchmarkable(
    AbstractMCMC.step(rng, $hmc_model, $hmc_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_initial_state_memory))
)
SUITE["Memory"]["hmc_population_update"] = @benchmarkable(
    AbstractMCMC.step(rng, $hmc_model, $hmc_population_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_population_initial_state_memory))
)

SUITE["Adaptive"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates[3:3], names[3:3])
    SUITE["Adaptive"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_adaptive))
    )
end
SUITE["Adaptive"]["hmc_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $hmc_model, $hmc_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_initial_state))
)
# The population strategy's extra warm-up cost is the covariance pass over the memory archive,
# so benchmark its adaptive step against the memory-backed state — diagonal and dense variants.
SUITE["Adaptive"]["hmc_population_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $hmc_model, $hmc_population_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_population_initial_state_memory))
)
SUITE["Adaptive"]["hmc_dense_population_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $hmc_model, $hmc_dense_population_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_dense_population_initial_state_memory))
)

SUITE["pt"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["pt"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_pt_and_annealing))
    )
end

SUITE["annealing"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["annealing"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_pt_and_annealing))
    )
end

tune!(SUITE)
results = run(SUITE, verbose = true, seconds = 10)

for (group_name, group) in results
    for (benchmark_name, benchmark_result) in group
        println("$group_name, $benchmark_name:")
        display(mean(benchmark_result))
        println()
    end
end
