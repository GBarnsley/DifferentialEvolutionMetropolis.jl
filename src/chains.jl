struct DifferentialEvolutionAdaptiveStatic{T} <:
    AbstractDifferentialEvolutionAdaptiveState{T} end

struct DifferentialEvolutionSample{V <: AbstractVector{<:Real}, VV <: AbstractVector{V}}
    "current position"
    x::VV
    "log density at current position"
    ld::V
end

function create_sample(
        state::DifferentialEvolutionState{
            T,
            A,
            L,
        }
    ) where {
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
    }
    return DifferentialEvolutionSample(
        copy.(state.xₚ[state.temperature_ladder.cold_chains]),
        copy(state.ldₚ[state.temperature_ladder.cold_chains])
    )
end

function create_sample(
        state::DifferentialEvolutionState{
            T, A,
            DifferentialEvolutionNullTemperatureLadder{T},
        }
    ) where {
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
    }
    return DifferentialEvolutionSample(
        copy.(state.xₚ), copy(state.ldₚ)
    )
end

function pick_chains(
        state::DifferentialEvolutionState{T, A, L, DifferentialEvolutionMemoryless{T}},
        current_chain::Int,
        n_chains::Int
    ) where {
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
    }
    return fast_sample_chains!(
        state.rngs[current_chain],
        state.x,
        length(state.x),
        n_chains,
        state.memory.indices_INTERNAL[current_chain],
        state.memory.ordered_indices_INTERNAL[current_chain],
        current_chain
    )
end

function pick_chains(
        state::DifferentialEvolutionState{T, A, L, M, V, VV},
        current_chain::Int,
        n_chains::Int
    ) where {
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
        V <: AbstractVector{T}, VV <: AbstractVector{V},
        M <: AbstractDifferentialEvolutionMemoryFormat{T, VV},
    }
    return fast_sample_chains!(
        state.rngs[current_chain],
        state.memory.mem_x,
        length(state.memory.mem_x),
        n_chains,
        state.memory.indices_INTERNAL[current_chain],
        state.memory.ordered_indices_INTERNAL[current_chain]
    )
end

function pick_chains(
        state::DifferentialEvolutionState{
            T, A, L, DifferentialEvolutionMemoryFill{T, VV}, V, VV,
        },
        current_chain::Int,
        n_chains::Int
    ) where {
        T <: Real, A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
        V <: AbstractVector{T}, VV <: AbstractVector{V},
    }
    return fast_sample_chains!(
        state.rngs[current_chain],
        state.memory.mem_x,
        state.memory.fill.position,
        n_chains,
        state.memory.indices_INTERNAL[current_chain],
        state.memory.ordered_indices_INTERNAL[current_chain]
    )
end

function update_state(
        state::DifferentialEvolutionState{
            T, A, L, DifferentialEvolutionMemoryless{T}, V, VV,
        };
        memory::DifferentialEvolutionMemoryless{T} = state.memory,
        adaptive_state::AbstractDifferentialEvolutionAdaptiveState{T} = state.adaptive_state,
        temperature_ladder::AbstractDifferentialEvolutionTemperatureLadder{T} = update_ladder!!(state.temperature_ladder),
        rngs::Vector{<:AbstractRNG} = state.rngs,
        x::VV = state.x,
        ld::V = state.ld,
        xₚ::VV = state.xₚ,
        ldₚ::V = state.ldₚ,
        kwargs...
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
    }
    return DifferentialEvolutionState(
        x, ld, xₚ, ldₚ, rngs, adaptive_state, temperature_ladder, memory
    )
end

function update_state(
        state::DifferentialEvolutionState{T, A, L, M, V, VV};
        memory::M = state.memory,
        adaptive_state::AbstractDifferentialEvolutionAdaptiveState{T} = state.adaptive_state,
        temperature_ladder::AbstractDifferentialEvolutionTemperatureLadder{T} = update_ladder!!(state.temperature_ladder),
        rngs::Vector{<:AbstractRNG} = state.rngs,
        x::VV = state.x,
        ld::V = state.ld,
        xₚ::VV = state.xₚ,
        ldₚ::V = state.ldₚ,
        update_memory::Bool = false,
        kwargs...
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A <: AbstractDifferentialEvolutionAdaptiveState{T},
        L <: AbstractDifferentialEvolutionTemperatureLadder{T},
        M <: AbstractDifferentialEvolutionMemoryFormat{T, VV},
    }
    if update_memory
        memory = update_memory!!(memory, x)
    end
    return DifferentialEvolutionState(
        x, ld, xₚ, ldₚ, rngs, adaptive_state, temperature_ladder, memory
    )
end

function update_chain!(model, state, offset, i)
    if isinf(offset) & (sign(offset) == -1.0)
        copyto!(state.xₚ[i], state.x[i])
        state.ldₚ[i] = state.ld[i]
        return false
    else
        state.ldₚ[i] = logdensity(model, state.xₚ[i])
        if log(rand(state.rngs[i])) * get_temperature(state.temperature_ladder, i) >
                (state.ldₚ[i] - state.ld[i] + offset)
            copyto!(state.xₚ[i], state.x[i])
            state.ldₚ[i] = state.ld[i]
            return false
        else
            return true
        end
    end
end

# non-adaptive step
"""
    step(rng, model_wrapper, sampler, state; parallel=false, update_memory=true, kwargs...)

Perform a single MCMC step using differential evolution sampling.

This is the core sampling function that proposes new states for all chains and accepts
or rejects them according to the Metropolis criterion. For adaptive samplers, the
function automatically fixes adaptive parameters before sampling.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler (any AbstractDifferentialEvolutionSampler)
- `state`: Current state of all chains

# Keyword Arguments
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`. Advisable for slow models.
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers). Defaults to `true`. Over writes memory options given at initialization.
- `kwargs...`: Additional keyword arguments passed to update functions (see https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)

# Returns
- `sample`: DifferentialEvolutionSample containing new positions and log-densities
- `new_state`: Updated state for the next iteration

# Example
```
sample, new_state = step(rng, model, sampler, state; parallel=true)
```

See also [`step_warmup`](@ref), [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments).
"""
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSampler,
        state::DifferentialEvolutionState{
            T, DifferentialEvolutionAdaptiveStatic{T},
        };
        parallel::Bool = false,
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    # Derive per-chain RNGs deterministically from the provided rng for this step.
    # Keep this here so `step` depends only on `rng` and `state`, and can be called in isolation.
    for i in eachindex(state.rngs)
        Random.seed!(state.rngs[i], rand(rng, UInt))
    end
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current states
    x = state.x

    # loop through chains running the update
    if parallel
        Threads.@threads for i in eachindex(x)
            offset, = proposal!(state, sampler, i)
            update_chain!(model, state, offset, i)
        end
    else
        for i in eachindex(x)
            offset, = proposal!(state, sampler, i)
            update_chain!(model, state, offset, i)
        end
    end

    return create_sample(state),
        update_state(
            state; x = state.xₚ, xₚ = state.x,
            ld = state.ldₚ, ldₚ = state.ld, update_memory = update_memory
        )
end

#previously adapted step
"""
    fix_sampler(sampler::AbstractDifferentialEvolutionSampler, adaptive_state::AbstractDifferentialEvolutionAdaptiveState)

Fix adaptive parameters of a sampler to their current adapted values.

For non-adaptive samplers, returns the sampler unchanged. For adaptive samplers,
returns a new sampler with the adaptive parameters fixed to their current values
in the `adaptive_state`.

# Arguments
- `sampler`: The differential evolution sampler to fix
- `adaptive_state`: The adaptive state containing current parameter values

# Returns
- A sampler with fixed (non-adaptive) parameters

# Example
```@example fix_sampler
using DifferentialEvolutionMetropolis, Random, Distributions

# This function is typically used after warmup/adaptation phase
# fixed_sampler = fix_sampler(adaptive_sampler, state.adaptive_state)
```

See also [`fix_sampler_state`](@ref).
"""
function fix_sampler(
        sampler::AbstractDifferentialEvolutionSampler,
        adaptive_state::AbstractDifferentialEvolutionAdaptiveState
    )
    return sampler
end

"""
    fix_sampler_state(sampler::AbstractDifferentialEvolutionSampler, state::DifferentialEvolutionState)

Fix adaptive sampler parameters and return a corresponding non-adaptive state.

Takes an adaptive sampler and state, fixes the sampler's adaptive parameters to their
current values, and returns both the fixed sampler and a simplified state without
adaptive components.

# Arguments
- `sampler`: The differential evolution sampler (potentially adaptive)
- `state`: The current sampler state (DifferentialEvolutionState)

# Returns
- `fixed_sampler`: Sampler with adaptive parameters fixed to current values
- `fixed_state`: State without adaptive components

# Example
```@example fix_sampler_state
using DifferentialEvolutionMetropolis, Random, Distributions

# This function is typically used after warmup/adaptation phase
# fixed_sampler, fixed_state = fix_sampler_state(sampler, state)
```

See also [`fix_sampler`](@ref).
"""
function fix_sampler_state(
        sampler::AbstractDifferentialEvolutionSampler,
        state::DifferentialEvolutionState{T}
    ) where {T <: Real}
    return fix_sampler(sampler, state.adaptive_state),
        DifferentialEvolutionState(
            state.x, state.ld, state.xₚ, state.ldₚ, state.rngs,
            DifferentialEvolutionAdaptiveStatic{T}(),
            state.temperature_ladder, state.memory
        )
end

function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSampler,
        state::DifferentialEvolutionState;
        update_memory::Bool = true,
        kwargs...
    )
    fixed_sampler, fixed_state = fix_sampler_state(sampler, state)
    sample, new_state = step(rng, model_wrapper, fixed_sampler, fixed_state; kwargs...)

    return sample,
        update_state(
            state; x = new_state.x, ld = new_state.ld, xₚ = new_state.xₚ,
            ldₚ = new_state.ldₚ, update_memory = update_memory,
            temperature_ladder = new_state.temperature_ladder
        )
end

function initialize_adaptive_state(
        sampler::AbstractDifferentialEvolutionSampler,
        model_wrapper::LogDensityModel, n_chains::Int
    )
    return DifferentialEvolutionAdaptiveStatic{Float64}()
end

"""
    step(rng, model_wrapper, sampler; n_chains, memory=true, N₀, adapt=true, initial_position=nothing, parallel=false, kwargs...)

Initialize differential evolution sampling by setting up chains and computing initial state.

This function serves as the entry point for differential evolution MCMC sampling. It handles
chain initialization, memory setup for memory-based samplers, adaptive state initialization,
and returns the initial sample and state that can be used with `AbstractMCMC.sample`.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler to use

# Keyword Arguments
$(generic_de_kwargs)

# Returns
- `sample`: DifferentialEvolutionSample containing initial positions and log-densities
- `state`: Initial state (DifferentialEvolutionState)
  ready for sampling

# Examples
```@example step_function
using DifferentialEvolutionMetropolis, Random, Distributions

# Setup
rng = Random.default_rng()
model_wrapper(θ) = logpdf(MvNormal([0.0, 0.0], I), θ)
sampler = deMCzs()

# Basic initialization with default settings
sample, state = step(rng, model_wrapper, sampler)

# Custom number of chains with memory disabled
sample2, state2 = step(rng, model_wrapper, sampler; n_chains=10, memory=false)

# With custom initial positions
init_pos = [randn(2) for _ in 1:8]
sample3, state3 = step(rng, model_wrapper, sampler; initial_position=init_pos)
```

# Notes
$(generic_notes)

See also [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments), [`deMC`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDifferentialEvolutionSampler;
        n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
        n_hot_chains::Int = 0,
        memory::Bool = true,
        memory_refill::Bool = false,
        memory_thin_interval::Int = 0,
        N₀::Int = 2 * (n_chains + n_hot_chains),
        adapt::Bool = true,
        initial_position::Union{Nothing, AbstractVector{<:AbstractVector{T}}} = nothing,
        parallel::Bool = false,
        #parallel tempering and annealing parameters
        max_temp_pt::T = 2.0 * sqrt(dimension(model_wrapper.logdensity)),
        max_temp_sa::T = max_temp_pt,
        α::T = 1.0,
        annealing::Bool = false,
        num_warmup::Int = 0,
        memory_size::Int = (num_warmup == 0) ? 1001 : num_warmup * 2,
        annealing_steps::Int = annealing ? num_warmup : 0,
        silent::Bool = false,
        temperature_ladder::Vector{Vector{T}} = create_temperature_ladder(
            n_chains, n_hot_chains, α, max_temp_pt, max_temp_sa, annealing_steps
        ),
        n_preallocated_indices::Int = 3,
        kwargs...
    ) where {T <: Real}
    model = model_wrapper.logdensity

    log = Vector{String}()

    n_true_chains = n_chains + n_hot_chains

    if adapt
        adaptive_state = initialize_adaptive_state(sampler, model_wrapper, n_true_chains)
    else
        adaptive_state = DifferentialEvolutionAdaptiveStatic{Float64}()
    end

    extra_memory = nothing

    if isnothing(initial_position)
        x = [randn(rng, dimension(model)) for _ in 1:n_true_chains]
    else
        push!(log, "DifferentialEvolutionMetropolis: adjusting provided initial positions...")
        initial_position = copy.(initial_position)
        current_N = length(initial_position)
        current_pars = length(initial_position[1])
        if current_pars != dimension(model)
            print_log(log)
            error("   Number of parameters in initial position must be equal to the number of parameters in the log density")
        end
        if current_N == n_true_chains
            push!(log, "   Done!")
            x = initial_position
        elseif current_N < n_true_chains
            push!(
                log,
                "   Initial position is smaller than the requested (or required) n_chains (including hot chains). Expanding initial position."
            )
            x = cat(
                [
                    randn(rng, eltype(initial_position[1]), current_pars)
                        for _ in 1:(n_true_chains - current_N)
                ],
                initial_position,
                dims = 1
            )
        elseif memory
            push!(
                log,
                "   Initial position is larger than requested number of chains. Shrinking initial position appending the rest to initial memory."
            )
            #shrink initial position
            x = initial_position[1:n_true_chains]
            extra_memory = initial_position[(n_true_chains + 1):end]
        elseif n_hot_chains == 0
            push!(log, "   Assuming initial position size is n_chains. Ignoring extra positions.")
            #assume n_chains is wrong
            n_true_chains = current_N
            x = initial_position
        else
            print_log(log)
            error("   Initial position size greater than n_chains + n_hot_chains. Cannot resolve mismatch.")
        end
    end

    if length(x) < dimension(model) && !memory
        @warn "In a memoryless model the number of chains should be greater than or equal to the number of parameters"
    end

    if parallel
        ld = Vector{eltype(x[1])}(undef, length(x))
        Threads.@threads for i in eachindex(x)
            ld[i] = logdensity(model, x[i])
        end
    else
        ld = [logdensity(model, xi) for xi in x]
    end

    if memory && n_hot_chains > 0
        @warn "Memory-based samplers do not typically require hot chains. Consider setting n_hot_chains=0."
    end

    temperature_ladder_struct = setup_temperature_struct(temperature_ladder)

    # Initialize per-chain RNGs deterministically from the provided rng.
    rngs = [Random.seed!(copy(rng), rand(rng, UInt)) for _ in 1:length(x)]

    if memory
        push!(log, "DifferentialEvolutionMetropolis: setting up memory...")
        mem_x = copy.(x)
        if !isnothing(extra_memory)
            push!(log, "   Appending initial extra memory")
            append!(mem_x, extra_memory)
        end
        if rem(N₀, n_true_chains) != 0
            push!(
                log,
                "   Setting initial memory size N₀ to be a multiple of n_chains for sampling convenience."
            )
            N₀ = ceil(Int, N₀ / n_true_chains) * n_true_chains
        end

        if length(mem_x) < N₀
            push!(log, "   Adding $(N₀ - length(mem_x)) random positions to initial memory.")
            for _ in 1:(N₀ - length(mem_x))
                push!(mem_x, randn(rng, eltype(x[1]), dimension(model)))
            end
        elseif length(mem_x) > N₀
            push!(log, "   Initial memory size greater than N₀, truncating memory.")
            mem_x = mem_x[(end - N₀ + 1):end]
        end

        total_memory_size = memory_size * n_true_chains
        if memory_size == 1001
            push!(log, "   Using default memory size of 1001, storing a maximum of $total_memory_size chains.")
            push!(log, "   Consider setting memory_size keyword argument to control memory usage!")
        elseif memory_size == num_warmup * 2
            push!(log, "   Using memory size of $memory_size (2x num_warmup/n_burnin) to store a maximum of $total_memory_size chains.")
            push!(log, "   Consider setting memory_size keyword argument to control memory usage!")
        end
        true_memory = vcat(mem_x, [similar(mem_x[1]) for _ in 1:(total_memory_size - N₀)])

        #setup fill method
        if memory_thin_interval > 0
            push!(log, "   Using thinning interval of $memory_thin_interval for memory updates.")
            memory_method = DifferentialEvolutionMemoryFillThin(
                N₀, n_true_chains, memory_thin_interval, memory_thin_interval
            )
        else
            memory_method = DifferentialEvolutionMemoryFillEvery(N₀, n_true_chains)
        end

        memory = DifferentialEvolutionMemoryFill{T, typeof(mem_x)}(
            true_memory, memory_method, memory_refill, length(true_memory),
            [Vector{Int}(undef, n_preallocated_indices) for _ in 1:n_true_chains],
            [Vector{Int}(undef, n_preallocated_indices - 1) for _ in 1:n_true_chains]
        )
    else
        memory = DifferentialEvolutionMemoryless{T}(
            [Vector{Int}(undef, n_preallocated_indices) for _ in 1:n_true_chains],
            [Vector{Int}(undef, n_preallocated_indices) for _ in 1:n_true_chains]
        )
    end

    if !silent
        print_log(log)
    end

    state = DifferentialEvolutionState(
        x, ld, copy.(x), copy(ld), rngs, adaptive_state, temperature_ladder_struct, memory
    )

    return create_sample(state), state
end

function print_log(log::Vector{String})
    for l in log
        @info l
    end
    return
end
