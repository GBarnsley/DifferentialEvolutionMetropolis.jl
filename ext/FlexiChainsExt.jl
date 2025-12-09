module FlexiChainsExt

import DifferentialEvolutionMetropolis as DME
import FlexiChains: VNChain, FlexiChain, Parameter, Extra
import AbstractMCMC

function AbstractMCMC.bundle_samples(
        samples::Vector{DME.DifferentialEvolutionSample{V, VV}},
        model_wrapper::AbstractMCMC.LogDensityModel,
        sampler::DME.AbstractDifferentialEvolutionSampler,
        state::DME.DifferentialEvolutionState,
        ::Type{VNChain};
        save_final_state::Bool = false,
        kwargs...
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    samples_ = DME.convert(VNChain, samples)
    if save_final_state
        return (
            samples_,
            state,
        )
    else
        return samples_
    end
end

function DME.convert(
        ::Type{VNChain},
        samples::Vector{DME.DifferentialEvolutionSample{V, VV}}
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}

    n_its = length(samples)
    n_chains = length(samples[1].x)

    param_names = vcat(
        Parameter.(
            Symbol.(DME.generate_names(length(samples[1].x[1]))),
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
