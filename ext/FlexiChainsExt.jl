module FlexiChainsExt

import DifferentialEvolutionMetropolis
import FlexiChains: VNChain, FlexiChain, Parameter, Extra
import AbstractMCMC

function DifferentialEvolutionMetropolis.convert(
        ::Type{VNChain},
        samples::Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionSample{V, VV}}
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}

    n_its = length(samples)
    n_chains = length(samples[1].x)

    param_names = vcat(
        Parameter.(
            Symbol.(DifferentialEvolutionMetropolis.generate_names(length(samples[1].x[1]))),
        ), Extra("ld")
    )
    sample_dicts = Array{Dict{Any, T}, 2}(undef, n_its, n_chains)

    for i in axes(samples, 1)
        for j in axes(samples[i].x, 1)
            sample_dicts[i, j] = Dict{Any, T}(zip(param_names, vcat(samples[i].x[j], samples[i].ld[j])))
        end
    end

    return FlexiChain{Symbol}(n_its, n_chains, sample_dicts)
end

end
