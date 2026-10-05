# Adapt ordinary vector -> scalar callables without transforming parameters or
# claiming gradient support. The dimension cannot be inferred from a callable.
struct FunctionLogDensity{F}
    f::F
    n_dims::Int
end

logdensity(model::FunctionLogDensity, x) = model.f(x)
dimension(model::FunctionLogDensity) = model.n_dims
capabilities(::Type{<:FunctionLogDensity}) = LogDensityOrder{0}()

function as_logdensity_model(model; n_dims = nothing)
    if !isnothing(n_dims) && !(n_dims isa Integer && n_dims > 0)
        throw(ArgumentError("n_dims must be a positive integer"))
    end
    if model isa LogDensityModel
        wrapped = model
    elseif !isnothing(capabilities(model))
        wrapped = LogDensityModel(model)
    else
        isnothing(n_dims) && throw(ArgumentError("Supply n_dims when sampling a raw log-density function"))
        return LogDensityModel(FunctionLogDensity(model, Int(n_dims)))
    end
    if !isnothing(n_dims) && n_dims != dimension(wrapped.logdensity)
        throw(ArgumentError("n_dims does not match the log-density dimension"))
    end
    return wrapped
end

# AbstractMCMC already wraps bare LogDensityProblems targets. Restrict this
# additional convenience method to functions and our samplers to avoid piracy.
"""
    sample([rng], logtarget::Function, sampler, N_or_isdone; n_dims, kwargs...)

Sample a vector-to-scalar log-density function with a differential evolution
sampler. Supply the positive integer `n_dims` because a raw function has no
LogDensityProblems dimension. The function is wrapped in an order-zero
LogDensityProblems target and `AbstractMCMC.LogDensityModel` internally.

No parameter transformation or Jacobian adjustment is performed. For HMC,
supply a gradient-capable LogDensityProblems target instead. Bare targets
implementing LogDensityProblems are already supported by AbstractMCMC.
"""
function sample(
        rng::AbstractRNG, f::Function,
        sampler::AbstractDifferentialEvolutionSampler, N_or_isdone;
        n_dims = nothing, kwargs...
    )
    return sample(rng, as_logdensity_model(f; n_dims = n_dims), sampler, N_or_isdone; kwargs...)
end
