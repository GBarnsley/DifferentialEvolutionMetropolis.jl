# Backend for `parallel = MCMCDistributed()`, its `CachingPool` keeps a copy of the model on each worker for the run
struct DistributedBackend{P <: CachingPool}
    pool::P
end

# `CachingPool`s cannot be serialised, so a state sent between processes (or saved) gets a fresh pool on the receiving side
function serialize(s::AbstractSerializer, ::DistributedBackend)
    writetag(s.io, OBJECT_TAG)
    return serialize(s, DistributedBackend)
end
deserialize(::AbstractSerializer, ::Type{<:DistributedBackend}) = parallel_backend(MCMCDistributed())

"""
    parallel_backend(parallel)
    parallel_backend(parallel, current)

Normalise the `parallel` keyword into the backend used to evaluate log-densities within a step.

`false`/`MCMCSerial()` evaluate chains in turn, `true`/`MCMCThreads()` use `Threads.@threads` over
chains, and `MCMCDistributed()` evaluates the proposals on the Distributed workers. With `current`
(the backend stored in the state) an `MCMCDistributed()` request reuses the existing worker pool.
"""
parallel_backend(parallel::Bool) = parallel ? MCMCThreads() : MCMCSerial()
parallel_backend(parallel::Union{MCMCSerial, MCMCThreads, DistributedBackend}) = parallel
parallel_backend(::MCMCDistributed) = DistributedBackend(CachingPool(workers()))
function parallel_backend(parallel)
    throw(
        ArgumentError(
            "Unsupported `parallel = $(repr(parallel))`. Expected a `Bool`, `MCMCSerial()`, " *
                "`MCMCThreads()` or `MCMCDistributed()`."
        )
    )
end
parallel_backend(parallel, current) = parallel_backend(parallel)
parallel_backend(::MCMCDistributed, current::DistributedBackend) = current

is_rejected(offset) = isinf(offset) & (sign(offset) == -1.0)

# Set `ld[i] = logdensity(model, x[i])` for `i` in `indices`, threads use the per-chain `chain_models` copies
function evaluate_logdensities!(ld, ::MCMCSerial, model, chain_models, x, indices)
    for i in indices
        ld[i] = logdensity(model, x[i])
    end
    return ld
end
function evaluate_logdensities!(ld, ::MCMCThreads, model, chain_models, x, indices)
    Threads.@threads for i in indices
        ld[i] = logdensity(chain_models[i], x[i])
    end
    return ld
end
function evaluate_logdensities!(ld, backend::DistributedBackend, model, chain_models, x, indices)
    isempty(indices) && return ld
    ld[indices] = pmap(Base.Fix1(logdensity, model), backend.pool, x[indices])
    return ld
end

# Update every chain and call `record(i, proposal, accepted)`, Distributed only sends the log-density evaluations to workers
function update_chains!(record, ::MCMCSerial, model, state, sampler)
    for i in eachindex(state.x)
        prop = proposal!(state, sampler, i)
        record(i, prop, update_chain!(model, state, first(prop), i))
    end
    return nothing
end
function update_chains!(record, ::MCMCThreads, model, state, sampler)
    Threads.@threads for i in eachindex(state.x)
        prop = proposal!(state, sampler, i)
        record(i, prop, update_chain!(state.chain_models[i], state, first(prop), i))
    end
    return nothing
end
function update_chains!(record, backend::DistributedBackend, model, state, sampler)
    props = [proposal!(state, sampler, i) for i in eachindex(state.x)]
    to_evaluate = [i for i in eachindex(props) if !is_rejected(first(props[i]))]
    evaluate_logdensities!(state.ldₚ, backend, model, state.chain_models, state.xₚ, to_evaluate)
    for i in eachindex(props)
        record(i, props[i], accept_chain!(state, first(props[i]), i))
    end
    return nothing
end

no_record(i, prop, accepted) = nothing
