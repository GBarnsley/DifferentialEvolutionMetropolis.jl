#code for sample() outputs into something useful, see MCMCChains

function samples_to_array(
        samples::Vector{
            DEMSample{
                V, VV,
            },
        }
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    n_draws = length(samples)
    n_chains = length(samples[1].x)
    n_params = length(samples[1].x[1])

    # Pre-allocate the 3D array
    result = Array{T, 3}(undef, n_draws, n_chains, n_params)

    for (i, sample) in enumerate(samples)
        for (j, chain) in enumerate(sample.x)
            for (k, param) in enumerate(chain)
                result[i, j, k] = param
            end
        end
    end

    return result
end

function ld_to_array(
        samples::Vector{
            DEMSample{
                V, VV,
            },
        }
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    n_draws = length(samples)
    n_chains = length(samples[1].ld)

    # Pre-allocate the 3D array
    result = Array{T, 2}(undef, n_draws, n_chains)

    for (i, sample) in enumerate(samples)
        for (j, ld) in enumerate(sample.ld)
            result[i, j] = ld
        end
    end

    return result
end

function process_outputs(
        samples::Vector{
            DEMSample{
                V, VV,
            },
        }
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return DEMOutput{T}(
        samples_to_array(samples),
        ld_to_array(samples)
    )
end

function process_outputs(samples::Vector{<:DEMSample}, n_ensembles::Int)
    output = process_outputs(ensemble_major(samples, n_ensembles))
    n_ensembles == 1 && return output
    n_iter, n_chains, n_params = size(output.samples)
    return DEMOutput(
        reshape(output.samples, n_iter, n_chains ÷ n_ensembles, n_ensembles, n_params),
        reshape(output.ld, n_iter, n_chains ÷ n_ensembles, n_ensembles)
    )
end

function ensemble_major(samples::Vector{<:DEMSample}, n_ensembles::Int)
    n_ensembles == 1 && return samples
    n_chains = length(samples[1].x)
    order = reduce(vcat, (e:n_ensembles:n_chains for e in 1:n_ensembles))
    return [DEMSample(s.x[order], s.ld[order]) for s in samples]
end

#code for post-processing outputs from sample()
function bundle_samples(
        samples::Vector{DEMSample{V, VV}},
        model_wrapper::LogDensityModel,
        sampler::AbstractDEMMetaSampler,
        state::DEMState,
        ::Type{T};
        save_final_state::Bool = false,
        kwargs...
    ) where {T, T2 <: Real, V <: AbstractVector{T2}, VV <: AbstractVector{V}}
    samples_ = convert(T, samples, n_ensembles(state.memory))
    if save_final_state
        return (
            samples_,
            state,
        )
    else
        return samples_
    end
end

function chainsstack(chns::Vector{<:DEMOutput{T}}) where {
        T <: Real,
    }
    return DEMOutput{T}(
        cat([c.samples for c in chns]...; dims = 2),
        cat((c.ld for c in chns)...; dims = 2)
    )
end

function chainsstack(
        chns::Vector{
            Tuple{
                T, E,
            },
        }
    ) where {T, E <: DEMState}
    return (
        chainsstack([c[1] for c in chns]),
        [c[2] for c in chns],
    )
end

function convert(
        ::Type{T},
        samples::Vector{DEMSample{V, VV}}
    ) where {T, T2 <: Real, V <: AbstractVector{T2}, VV <: AbstractVector{V}}
    return samples
end

function convert(
        ::Type{DEMOutput},
        samples::Vector{DEMSample{V, VV}}
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return process_outputs(samples)
end

# Chain types that bundle chains from every ensemble put them in the chain dimension in ensemble-major order
function convert(
        ::Type{T},
        samples::Vector{DEMSample{V, VV}},
        n_ensembles::Int
    ) where {T, T2 <: Real, V <: AbstractVector{T2}, VV <: AbstractVector{V}}
    return convert(T, samples)
end

function convert(
        ::Type{DEMOutput},
        samples::Vector{DEMSample{V, VV}},
        n_ensembles::Int
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    return process_outputs(samples, n_ensembles)
end

function generate_names(n_params::Int)
    return [Symbol("param_$(i)") for i in 1:n_params]
end

function foreach_threaded(f, indices, parallel::Bool)
    if parallel
        Threads.@threads for i in indices
            f(i)
        end
    else
        foreach(f, indices)
    end
    return nothing
end
