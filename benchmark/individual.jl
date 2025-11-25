using AbstractMCMC, .DifferentialEvolutionMetropolis, Distributions, LogDensityProblems, Random
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

#settings
#update = setup_de_update()
#update = setup_subspace_sampling()
#update = setup_sampler_scheme(setup_subspace_sampling(), setup_subspace_sampling())
update = setup_subspace_sampling()
memory = false
rng = Xoshiro(1234)

#initial step
__,
    initial_state = AbstractMCMC.step(
    rng, am_model, update; memory = memory,
    initial_position = initial_position, n_chains = n_chains
);

function run_steps!(n, rng, am_model, update, initial_state)
    state = deepcopy(initial_state)
    for q in 1:n
        __, state = AbstractMCMC.step_warmup(rng, am_model, update, state)
    end
    return
end

@profview run_steps!(500, rng, am_model, update, initial_state)
@profview_allocs run_steps!(1000, rng, am_model, update, initial_state) sample_rate = 0.1

@benchmark AbstractMCMC.step_warmup(rng, $am_model, $update, state) setup = (
    rng = copy($rng); state = deepcopy($initial_state)
)
