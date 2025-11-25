using BenchmarkTools
using AbstractMCMC, DifferentialEvolutionMetropolis, Distributions, LogDensityProblems, Random
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

SUITE["MemoryLess"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["MemoryLess"][name] = @benchmarkable(
        AbstractMCMC.step(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state))
    )
end

SUITE["Memory"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates, names)
    SUITE["Memory"][name] = @benchmarkable(
        AbstractMCMC.step(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_memory))
    )
end

SUITE["Adaptive"] = BenchmarkGroup(["string"])
for (update, name) in zip(updates[3:3], names[3:3])
    SUITE["Adaptive"][name] = @benchmarkable(
        AbstractMCMC.step_warmup(rng, $am_model, $update, state),
        setup = (rng = copy($rng); state = deepcopy($initial_state_adaptive))
    )
end

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
