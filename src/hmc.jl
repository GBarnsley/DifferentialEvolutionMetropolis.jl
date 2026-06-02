"""
    setup_hmc_update(sampler; metric_strategy = ...)

Wrap an AdvancedHMC sampler (`NUTS`, `HMC`, `HMCDA`, or a hand-built `HMCSampler`) so it
can participate as one update inside a [`setup_sampler_scheme`](@ref) composite, running
the same HMC kernel on each chain in turn.

The implementation lives in the `AdvancedHMCExt` package extension; calling this requires
`AdvancedHMC.jl` to be loaded.
"""
function setup_hmc_update(args...; kwargs...)
    return error("`setup_hmc_update` requires AdvancedHMC.jl; run `using AdvancedHMC`.")
end
