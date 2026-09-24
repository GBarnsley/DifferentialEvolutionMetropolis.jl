module PathfinderExt
# use Pathfinder results as initial positions and initial memory

import DifferentialEvolutionMetropolis: resolve_initial_position
import Pathfinder: PathfinderResult, MultiPathfinderResult
import Distributions: MixtureModel, components
import Random: randperm

const PathfinderResults = Union{Tuple{Vararg{PathfinderResult}}, AbstractVector{<:PathfinderResult}}

columns(draws::AbstractMatrix) = [Vector(c) for c in eachcol(draws)]

function chain_starts(rng, mixture::MixtureModel, n_chains::Int, stratify::Bool)
    if !stratify
        return columns(rand(rng, mixture, n_chains))
    end
    comps = components(mixture)
    return [rand(rng, comps[mod1(i, length(comps))]) for i in 1:n_chains]
end

function resolve_initial_position(
        result::PathfinderResult, rng, n_chains, n_memory; stratify = true
    )
    return columns(rand(rng, result.fit_distribution, n_chains + n_memory))
end

function resolve_initial_position(
        result::MultiPathfinderResult, rng, n_chains, n_memory; stratify = true
    )
    chains = chain_starts(rng, result.fit_distribution, n_chains, stratify)
    n_draws = size(result.draws, 2)
    n_reused = min(n_memory, n_draws)
    memory = columns(result.draws[:, randperm(rng, n_draws)[1:n_reused]])
    append!(memory, columns(rand(rng, result.fit_distribution, n_memory - n_reused)))
    return vcat(chains, memory)
end

function resolve_initial_position(
        results::PathfinderResults, rng, n_chains, n_memory; stratify = true
    )
    isempty(results) && throw(ArgumentError("`initial_position` must contain at least one PathfinderResult."))
    mixture = MixtureModel([result.fit_distribution for result in results])
    return vcat(
        chain_starts(rng, mixture, n_chains, stratify),
        columns(rand(rng, mixture, n_memory))
    )
end

end
