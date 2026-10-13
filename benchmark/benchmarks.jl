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
swap_update = setup_swap_update()
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
# All chains share the first annealing temperature, so swaps are benchmarked one step in.
__,
    initial_state_mid_annealing = AbstractMCMC.step_warmup(
    rng, am_model, de_update, initial_state_pt_and_annealing
)
__,
    initial_state_pt = AbstractMCMC.step(
    rng, am_model, de_update; memory = false, initial_position = initial_position,
    n_chains = n_chains, n_hot_chains = 10
)

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
hmc_memory_update = setup_hmc_update(NUTS(0.8); n_dims = 5, metric_strategy = memory_metric())
__,
    hmc_memory_initial_state_memory = AbstractMCMC.step(
    rng, hmc_model, hmc_memory_update; memory = true,
    initial_position = initial_position_with_memory, n_chains = n_chains, N₀ = N₀
)

hmc_dense_memory_update = setup_hmc_update(
    NUTS(0.8; metric = :dense); n_dims = 5, metric_strategy = memory_metric()
)
__,
    hmc_dense_memory_initial_state_memory = AbstractMCMC.step(
    rng, hmc_model, hmc_dense_memory_update; memory = true,
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
SUITE["Memory"]["hmc_memory_update"] = @benchmarkable(
    AbstractMCMC.step(rng, $hmc_model, $hmc_memory_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_memory_initial_state_memory))
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
# The memory strategy's extra warm-up cost is the covariance pass over the memory archive,
# so benchmark its adaptive step against the memory-backed state — diagonal and dense variants.
SUITE["Adaptive"]["hmc_memory_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $hmc_model, $hmc_memory_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_memory_initial_state_memory))
)
SUITE["Adaptive"]["hmc_dense_memory_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $hmc_model, $hmc_dense_memory_update, state),
    setup = (rng = copy($rng); state = deepcopy($hmc_dense_memory_initial_state_memory))
)

SUITE["pt"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["pt"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_pt))
    )
end
SUITE["pt"]["swap_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $am_model, $swap_update, state),
    setup = (rng = copy($rng); state = deepcopy($initial_state_pt))
)

SUITE["annealing"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["annealing"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_pt_and_annealing))
    )
end
SUITE["annealing"]["swap_update"] = @benchmarkable(
    AbstractMCMC.step_warmup(rng, $am_model, $swap_update, state),
    setup = (rng = copy($rng); state = deepcopy($initial_state_mid_annealing))
)

# Multiple-try versions of the symmetric updates, for every memory option, adaptation, parallel tempering and annealing
n_tries = 4
multi_try_updates = (de_update, subspace_update)
multi_try_names = ("de_update", "subspace_update")
multi_try_step(update; kwargs...) = last(
    AbstractMCMC.step(
        rng, am_model, update; n_chains = n_chains, n_tries = n_tries, silent = true, kwargs...
    )
)
multi_try_states = Dict(
    "MemoryLess" => multi_try_step(de_update; memory = false, initial_position = initial_position),
    "Memory" => multi_try_step(de_update; memory = true, initial_position = initial_position_with_memory, N₀ = N₀),
    "MemoryRefill" => multi_try_step(
        de_update; memory = true, initial_position = initial_position_with_memory, N₀ = N₀,
        memory_refill = true, memory_size = 4
    ),
    "MemoryThinned" => multi_try_step(
        de_update; memory = true, initial_position = initial_position_with_memory, N₀ = N₀,
        memory_thin_interval = 2
    ),
    "pt" => multi_try_step(de_update; memory = false, initial_position = initial_position, n_hot_chains = 10),
    "annealing" => multi_try_step(
        de_update; memory = false, initial_position = initial_position, n_hot_chains = 10, annealing_steps = 5
    ),
)
multi_try_adaptive_states = Dict(
    "MemoryLess" => multi_try_step(subspace_update; memory = false, initial_position = initial_position, adapt = true),
    "Memory" => multi_try_step(
        subspace_update; memory = true, initial_position = initial_position_with_memory, N₀ = N₀, adapt = true
    ),
)

SUITE["MultiTry"] = BenchmarkGroup(["string"])
for (state_name, multi_try_state) in multi_try_states, (update, name) in zip(multi_try_updates, multi_try_names)
    SUITE["MultiTry"]["$(state_name)/$(name)"] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($multi_try_state))
    )
end
for (state_name, multi_try_state) in multi_try_adaptive_states
    SUITE["MultiTry"]["Adaptive/$(state_name)/subspace_update"] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $subspace_update, state),
        setup = (rng = copy($rng); state = deepcopy($multi_try_state))
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
