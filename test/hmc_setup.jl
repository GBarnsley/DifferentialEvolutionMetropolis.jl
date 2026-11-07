using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD

# Correlated Gaussian with known moments, so the HMC metric should approach its marginal variances.
struct CorrelatedGaussianModel{D}
    dist::D
end
function LogDensityProblems.dimension(model::CorrelatedGaussianModel)
    return length(model.dist)
end
function LogDensityProblems.logdensity(model::CorrelatedGaussianModel, x::AbstractVector{<:Real})
    return logpdf(model.dist, x)
end
LogDensityProblems.capabilities(::Type{<:CorrelatedGaussianModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.capabilities(::CorrelatedGaussianModel) = LogDensityProblems.LogDensityOrder{0}()

# Pull the (single, shared) HMC adaptive state out of a composite state.
hmc_adaptive_state(state) = state.adaptive_state.adaptive_states[1]
