struct DEMAdaptiveStatic{T} <:
    AbstractDEMAdaptiveState{T} end

struct DEMSample{V <: AbstractVector{<:Real}, VV <: AbstractVector{V}}
    "current position"
    x::VV
    "log density at current position"
    ld::V
end

function create_sample(
        state::DEMState
    )
    return DEMSample(
        copy.(state.xₚ_smpl_view),
        copy(state.ldₚ_smpl_view)
    )
end

# Chains are updated in order, so chains before `current_chain` have their new position in `xₚ`
@generated function pick_chains(
        state::DEMState{T, A, L, M},
        current_chain::Int,
        ::Val{N}
    ) where {T, A, L, M <: DEMMemoryless, N}
    exprs = [:(indices[$i] < current_chain ? xₚ[indices[$i]] : x[indices[$i]]) for i in 1:N]
    return quote
        rng = state.rngs[current_chain]
        x = state.x
        xₚ = state.xₚ
        memory = state.memory
        indices = memory.indices_INTERNAL[current_chain]
        ordered_indices = memory.ordered_indices_INTERNAL[current_chain]
        fast_sample_chains!(
            rng,
            x,
            length(x),
            $N,
            indices,
            ordered_indices,
            current_chain
        )
        return ($(exprs...),)
    end
end

@generated function pick_chains(
        state::DEMState{T, A, L, M, Model, B, V, VV},
        current_chain::Int,
        ::Val{N}
    ) where {T, A, L, V, VV, Model, B, M <: AbstractDEMMemoryFormat{T, VV}, N}
    exprs = [:(mem_x[indices[$i]]) for i in 1:N]
    limit_expr = M <: DEMMemoryFill ? :(memory.fill.position) : :(length(mem_x))
    return quote
        rng = state.rngs[current_chain]
        memory = state.memory
        mem_x = memory.mem_x
        indices = memory.indices_INTERNAL[current_chain]
        ordered_indices = memory.ordered_indices_INTERNAL[current_chain]
        fast_sample_chains!(
            rng,
            mem_x,
            $limit_expr,
            $N,
            indices,
            ordered_indices
        )
        return ($(exprs...),)
    end
end

is_enabled(option::Val{true}) = true
is_enabled(option::Val{false}) = false

function _update_state(
        state::DEMState{
            T, A1, L1, M1, Model, B, V, VV, R, V1, V2,
        },
        swap_positions,
        memory::AbstractDEMMemory{T},
        adaptive_state::AbstractDEMAdaptiveState{T},
        temperature_ladder::AbstractDEMTemperatureLadder{T},
        rngs::Vector{<:AbstractRNG},
        new_state::DEMState{
            T, A2, L2, M2, Model, B, V, VV, R, V1, V2,
        },
        order_state,
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A1 <: AbstractDEMAdaptiveState{T},
        L1 <: AbstractDEMTemperatureLadder{T},
        A2 <: AbstractDEMAdaptiveState{T},
        L2 <: AbstractDEMTemperatureLadder{T},
        M1 <: AbstractDEMMemory{T},
        M2 <: AbstractDEMMemory{T},
        V1 <: SubArray, V2 <: SubArray,
        R <: AbstractRNG, Model, B,
    }
    if is_enabled(swap_positions)
        #swap over x and xₚ for the next update
        return DEMState(
            new_state.xₚ, new_state.ldₚ, rngs, adaptive_state, temperature_ladder, memory, state.chain_models, state.multi_try, new_state.x, new_state.ld, new_state.xₚ_smpl_view, new_state.ldₚ_smpl_view, new_state.x_smpl_view, new_state.ld_smpl_view, order_state
        )
    else
        #already swapped elsewhere
        return DEMState(
            new_state.x, new_state.ld, rngs, adaptive_state, temperature_ladder, memory, state.chain_models, state.multi_try, new_state.xₚ, new_state.ldₚ, new_state.x_smpl_view, new_state.ld_smpl_view, new_state.xₚ_smpl_view, new_state.ldₚ_smpl_view, order_state
        )
    end
end

function update_state(
        state::DEMState{
            T, A1, L1, M1, Model, B, V, VV, R, V1, V2,
        };
        swap_positions = Val(false),
        memory::DEMMemoryless{T} = state.memory,
        adaptive_state::AbstractDEMAdaptiveState{T} = state.adaptive_state,
        temperature_ladder::AbstractDEMTemperatureLadder{T} = update_ladder!!(state.temperature_ladder),
        rngs::Vector{<:AbstractRNG} = state.rngs,
        order_state = state.order_state,
        new_state::DEMState{
            T, A2, L2, M2, Model, B, V, VV, R, V1, V2,
        } = state,
        kwargs...
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A1 <: AbstractDEMAdaptiveState{T},
        L1 <: AbstractDEMTemperatureLadder{T},
        A2 <: AbstractDEMAdaptiveState{T},
        L2 <: AbstractDEMTemperatureLadder{T},
        M1 <: DEMMemoryless{T},
        M2 <: AbstractDEMMemory{T},
        V1 <: SubArray, V2 <: SubArray,
        R <: AbstractRNG, Model, B,
    }
    return _update_state(state, swap_positions, memory, adaptive_state, temperature_ladder, rngs, new_state, order_state)
end

function update_state(
        state::DEMState{
            T, A1, L1, M1, Model, B, V, VV, R, V1, V2,
        };
        swap_positions = Val(false),
        memory::AbstractDEMMemory{T} = state.memory,
        adaptive_state::AbstractDEMAdaptiveState{T} = state.adaptive_state,
        temperature_ladder::AbstractDEMTemperatureLadder{T} = update_ladder!!(state.temperature_ladder),
        rngs::Vector{<:AbstractRNG} = state.rngs,
        order_state = state.order_state,
        new_state::DEMState{
            T, A2, L2, M2, Model, B, V, VV, R, V1, V2,
        } = state,
        update_memory::Bool = false
    ) where {
        T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V},
        A1 <: AbstractDEMAdaptiveState{T},
        L1 <: AbstractDEMTemperatureLadder{T},
        A2 <: AbstractDEMAdaptiveState{T},
        L2 <: AbstractDEMTemperatureLadder{T},
        M1 <: AbstractDEMMemoryFormat{T, VV},
        M2 <: AbstractDEMMemory{T},
        V1 <: SubArray, V2 <: SubArray,
        R <: AbstractRNG, Model, B,
    }
    if update_memory
        if is_enabled(swap_positions)
            memory = update_memory!!(memory, new_state.xₚ)
        else
            #already swapped
            memory = update_memory!!(memory, new_state.x)
        end
    end
    return _update_state(state, swap_positions, memory, adaptive_state, temperature_ladder, rngs, new_state, order_state)
end

# Sequential updates without multiple tries run one chain at a time even with `parallel = true`
cannot_use_parallel(sampler::AbstractDEMSampler{AbstractDEMSamplerSerial, AbstractDEMSamplerSingleTry}, n_tries::Int) = true
cannot_use_parallel(sampler::AbstractDEMSampler{AbstractDEMSamplerSerial, AbstractDEMSamplerMultipleTry}, n_tries::Int) = !(n_tries > 1)
cannot_use_parallel(sampler::AbstractDEMSampler{AbstractDEMSamplerMultiThread}, n_tries::Int) = false

function check_parallel(sampler::AbstractDEMMetaSampler, parallel::Bool, memory::Bool, n_tries::Int)
    if parallel && !memory && cannot_use_parallel(sampler, n_tries)
        @warn "None of the provided samplers can evaluate in parallel, yet `parallel = true`, consider `memory = true` or multi-try updates to use multithreading."
    end
    return nothing
end

function checked_logdensity(model, x)
    value = logdensity(model, x)
    isnan(value) && throw(DomainError(value, "Log density must not be NaN"))
    return value
end

function update_chain!(model, state, offset, i)
    if isinf(offset) & (sign(offset) == -1.0)
        copyto!(state.xₚ[i], state.x[i])
        state.ldₚ[i] = state.ld[i]
        return false
    else
        state.ldₚ[i] = checked_logdensity(model, state.xₚ[i])
        if log(rand(state.rngs[i])) <
                (state.ldₚ[i] - state.ld[i]) / get_temperature(state.temperature_ladder, i) + offset
            return true
        else
            copyto!(state.xₚ[i], state.x[i])
            state.ldₚ[i] = state.ld[i]
            return false
        end
    end
end

function propose_and_update_chain!(state, sampler, i)
    offset, = proposal!(state.xₚ[i], state.x[i], state, sampler, i)
    update_chain!(state.chain_models[i], state, offset, i)
    return nothing
end

function propose_and_update_chains!(state, sampler)
    for i in eachindex(state.x)
        propose_and_update_chain!(state, sampler, i)
    end
    return nothing
end

function update_chains!(state, sampler, parallel::Bool)
    if parallel
        Threads.@threads for i in eachindex(state.x)
            propose_and_update_chain!(state, sampler, i)
        end
    else
        propose_and_update_chains!(state, sampler)
    end

    return nothing
end

# if we are not using memory and not parallel compatible, we ignore parallel, the multi-try method here is more specific so takes precedence
function update_chains!(
        state::DEMState{<:Any, <:Any, <:Any, <:DEMMemoryless},
        sampler::AbstractDEMSampler{AbstractDEMSamplerSerial}, parallel::Bool
    )
    propose_and_update_chains!(state, sampler)

    return nothing
end

# multi-try methods are defined in src/multi_try.jl

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
- `sampler`: Differential evolution sampler from this package
- `state`: Current state of all chains

# Keyword Arguments
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`. Advisable for slow models.
  With a memoryless state, updates other than HMC update the chains in turn and only evaluate each chain's multiple-try proposals in parallel.
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers). Defaults to `true`. Over writes memory options given at initialization.
- `kwargs...`: Additional keyword arguments passed to update functions (see https://turinglang.org/AbstractMCMC.jl/stable/api/#Common-keyword-arguments)

# Returns
- `sample`: DEMSample containing new positions and log-densities
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
        sampler::AbstractDEMMetaSampler,
        state::DEMState{
            T, DEMAdaptiveStatic{T},
        };
        parallel::Bool = false,
        update_memory::Bool = true,
        kwargs...
    ) where {T <: Real}
    # Derive per-chain RNGs deterministically from the provided rng for this step.
    # Keep this here so `step` depends only on `rng` and `state`, and can be called in isolation.
    for i in eachindex(state.rngs)
        reseed!(state.rngs[i], rng)
    end
    # Extract the wrapped model which implements LogDensityProblems.jl.
    model = model_wrapper.logdensity
    # Extract the current states
    x = state.x

    prepare_scratch!(sampler, state)

    # loop through chains running the update
    update_chains!(state, sampler, parallel)

    return create_sample(state),
        update_state(
            state;
            swap_positions = Val(true),
            update_memory = update_memory
        )
end

#previously adapted step
"""
    fix_sampler(sampler::AbstractDEMMetaSampler, adaptive_state::AbstractDEMAdaptiveState)

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
        sampler::AbstractDEMMetaSampler,
        adaptive_state::AbstractDEMAdaptiveState
    )
    return sampler
end

"""
    fix_sampler_state(sampler::AbstractDEMMetaSampler, state::DEMState)

Fix adaptive sampler parameters and return a corresponding non-adaptive state.

Takes an adaptive sampler and state, fixes the sampler's adaptive parameters to their
current values, and returns both the fixed sampler and a simplified state without
adaptive components. Adaptive updates inside sampler schemes are fixed as well.

# Arguments
- `sampler`: The differential evolution sampler (potentially adaptive)
- `state`: The current sampler state (DEMState)

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
        sampler::AbstractDEMMetaSampler,
        state::DEMState{T}
    ) where {T <: Real}
    return fix_sampler(sampler, state.adaptive_state),
        update_state(
            state; update_memory = false,
            adaptive_state = DEMAdaptiveStatic{T}(),
            temperature_ladder = state.temperature_ladder
        )
end

"""
    step(rng, model_wrapper, sampler, state::DEMState; update_memory=true, kwargs...)

Evaluate an adaptive step. Delegates to the inner step after fixing the sampler state,
correctly forwarding the `update_memory` flag, and updating the outer state with the
resulting memory and temperature ladder without duplicating memory writes or temperature ladder advances.
"""
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDEMMetaSampler,
        state::DEMState;
        update_memory::Bool = true,
        kwargs...
    )

    fixed_sampler, fixed_state = fix_sampler_state(sampler, state)

    sample, new_state = step(
        rng, model_wrapper, fixed_sampler, fixed_state;
        update_memory = update_memory, kwargs...
    )

    return sample,
        update_state(
            state; new_state = new_state, update_memory = false,
            memory = new_state.memory,
            temperature_ladder = new_state.temperature_ladder
        )
end

function initialize_adaptive_state(
        sampler::AbstractDEMMetaSampler,
        model_wrapper::LogDensityModel, n_chains::Int
    )
    return initialize_adaptive_state(sampler, model_wrapper, n_chains, Float64)
end

function initialize_adaptive_state(
        sampler::AbstractDEMMetaSampler,
        model_wrapper::LogDensityModel, n_chains::Int, ::Type{T}
    ) where {T <: Real}
    return DEMAdaptiveStatic{T}()
end

"""
    resolve_initial_position(initial_position, rng, n_chains, n_memory; stratify = true)

Convert a user-supplied `initial_position` into `nothing` or a vector of parameter vectors.

Extensions add methods for other input types (e.g. Pathfinder results). These should return
`n_chains` chain starting positions followed by `n_memory` initial memory positions.
The default methods return the input unchanged and ignore `stratify`.
"""
resolve_initial_position(initial_position::Nothing, rng, n_chains, n_memory; stratify = true) = initial_position
resolve_initial_position(initial_position::AbstractVector{<:AbstractVector{<:Real}}, rng, n_chains, n_memory; stratify = true) = initial_position
function resolve_initial_position(initial_position, rng, n_chains, n_memory; stratify = true)
    throw(
        ArgumentError(
            "Unsupported `initial_position` of type $(nameof(typeof(initial_position))). " *
                "Expected `nothing` or a vector of parameter vectors. Pathfinder results require `using Pathfinder`."
        )
    )
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
- `sample`: DEMSample containing initial positions and log-densities
- `state`: Initial state (DEMState)
  ready for sampling

# Examples
```@example step_function
using DifferentialEvolutionMetropolis, AbstractMCMC, Random
using TransformedLogDensities, TransformVariables

# Setup
rng = Random.default_rng()
ld = TransformedLogDensity(as(Array, 2), θ -> -sum(abs2, θ) / 2)
model_wrapper = AbstractMCMC.LogDensityModel(ld)
sampler = setup_random_order_scheme(setup_de_update(n_dims = 2), setup_snooker_update(); w = [0.9, 0.1])

# Basic initialization with default settings
sample, state = AbstractMCMC.step(rng, model_wrapper, sampler)

# Custom number of chains with memory disabled
sample2, state2 = AbstractMCMC.step(rng, model_wrapper, sampler; n_chains=10, memory=false)

# With custom initial positions
init_pos = [randn(rng, 2) for _ in 1:8]
sample3, state3 = AbstractMCMC.step(rng, model_wrapper, sampler; initial_position=init_pos)
```

# Notes
$(generic_notes)

See also [`sample` from AbstractMCMC](https://turinglang.org/AbstractMCMC.jl/dev/api/#Common-keyword-arguments), [`deMC`](@ref), [`deMCzs`](@ref), [`DREAMz`](@ref).
"""
function step(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDEMMetaSampler;
        n_chains::Int = max(dimension(model_wrapper.logdensity) * 2, 3),
        n_hot_chains::Int = 0,
        memory::Bool = true,
        memory_refill::Bool = false,
        memory_thin_interval::Int = 0,
        N₀::Union{Nothing, Int} = nothing,
        adapt::Bool = true,
        initial_position = nothing,
        T::Union{Nothing, Type{<:AbstractFloat}} = nothing,
        stratify_initial_position::Bool = true,
        parallel::Bool = false,
        #parallel tempering and annealing parameters
        max_temp_pt::Real = 2.0 * sqrt(dimension(model_wrapper.logdensity)),
        max_temp_sa::Real = max_temp_pt,
        α::Real = 1.0,
        annealing::Bool = false,
        num_warmup::Int = 0,
        memory_size::Int = (num_warmup == 0) ? 1001 : num_warmup * 2,
        annealing_steps::Int = annealing ? num_warmup : 0,
        silent::Bool = false,
        temperature_ladder = nothing,
        n_preallocated_indices::Int = 3,
        n_tries::Int = 1,
        kwargs...
    )
    model = model_wrapper.logdensity
    n_tries = check_n_tries(sampler, n_tries)
    check_parallel(sampler, parallel, memory, n_tries)
    initial_position = resolve_initial_position(
        initial_position, rng, n_chains + n_hot_chains,
        memory ? something(N₀, n_chains + n_hot_chains) : 0;
        stratify = stratify_initial_position
    )
    if isnothing(T)
        T = isnothing(initial_position) ? Float64 : eltype(initial_position[1])
    end

    if isnothing(temperature_ladder)
        temperature_ladder = create_temperature_ladder(
            n_chains, n_hot_chains, T(α), T(max_temp_pt), T(max_temp_sa), annealing_steps
        )
    else
        temperature_ladder = [T.(ladder_rung) for ladder_rung in temperature_ladder]
    end

    log = Vector{String}()

    n_true_chains = n_chains + n_hot_chains

    if adapt
        adaptive_state = initialize_adaptive_state(sampler, model_wrapper, n_true_chains, T)
    else
        adaptive_state = DEMAdaptiveStatic{T}()
    end

    extra_memory = nothing

    if isnothing(initial_position)
        x = [randn(rng, T, dimension(model)) for _ in 1:n_true_chains]
    else
        push!(log, "DifferentialEvolutionMetropolis: adjusting provided initial positions...")
        initial_position = [Vector{T}(position) for position in initial_position]
        current_N = length(initial_position)
        current_pars = length(initial_position[1])
        if any(position -> length(position) != dimension(model), initial_position)
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
                initial_position,
                [
                    randn(rng, eltype(initial_position[1]), current_pars)
                        for _ in 1:(n_true_chains - current_N)
                ];
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

    if !memory && n_true_chains < chains_required(sampler)
        error("Memoryless scheme with sampler $(typeof(sampler)) requires at least $(chains_required(sampler)) chains, but only $n_true_chains were provided (including hot chains).")
    end

    if length(x) < dimension(model) && !memory
        @warn "In a memoryless model the number of chains should be greater than or equal to the number of parameters"
    end

    chain_models = [deepcopy(model) for _ in 1:(length(x) * n_tries)]
    if parallel
        ld = Vector{eltype(x[1])}(undef, length(x))
        Threads.@threads for i in eachindex(x)
            ld[i] = checked_logdensity(chain_models[i], x[i])
        end
    else
        ld = T[checked_logdensity(model, xi) for xi in x]
    end

    if memory && n_hot_chains > 0
        @warn "Memory-based samplers do not typically require hot chains. Consider setting n_hot_chains=0."
    end

    if memory
        push!(log, "DifferentialEvolutionMetropolis: setting up memory...")
        mem_x = copy.(x)
        if !isnothing(extra_memory)
            push!(log, "   Appending initial extra memory")
            append!(mem_x, extra_memory)
        end
        if isnothing(N₀)
            N₀ = max(2 * n_true_chains, length(mem_x))
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
        if total_memory_size < N₀ + n_true_chains
            error("memory_size (currently $memory_size) is too small for the initial memory N₀ (currently $N₀) and number of chains (currently $n_true_chains). The total memory size (memory_size * n_chains = $total_memory_size) must be at least N₀ + n_chains = $(N₀ + n_true_chains). Consider specifying a larger memory_size.")
        end
        if (memory_size == 1001) & memory_refill
            push!(log, "   Using memory refill with default memory size of 1001, storing a maximum of $total_memory_size chains.")
            push!(log, "   Consider tailoring memory_size keyword argument to control memory usage!")
        elseif (memory_size == (num_warmup * 2)) & memory_refill
            push!(log, "   Using memory refill with default memory size of $memory_size (2x num_warmup/n_burnin) to store a maximum of $total_memory_size chains.")
            push!(log, "   Consider tailoring memory_size keyword argument to control memory usage!")
        end
        true_memory = vcat(mem_x, [similar(mem_x[1]) for _ in 1:(total_memory_size - N₀)])

        #setup fill method
        if memory_thin_interval > 0
            push!(log, "   Using thinning interval of $memory_thin_interval for memory updates.")
            memory_method = DEMMemoryFillThin(
                N₀, n_true_chains, memory_thin_interval, memory_thin_interval
            )
        else
            memory_method = DEMMemoryFillEvery(N₀, n_true_chains)
        end

        memory = DEMMemoryFill{T, typeof(mem_x), typeof(memory_method)}(
            true_memory, memory_method, memory_refill, length(true_memory),
            [fill(1, n_preallocated_indices) for _ in 1:n_true_chains],
            [Vector{Int}(undef, n_preallocated_indices - 1) for _ in 1:n_true_chains]
        )
    else
        memory = DEMMemoryless{T}(
            [fill(1, n_preallocated_indices) for _ in 1:n_true_chains],
            [Vector{Int}(undef, n_preallocated_indices) for _ in 1:n_true_chains],
        )
    end

    temperature_ladder_struct = setup_temperature_struct(temperature_ladder)
    check_swap_setup(sampler, temperature_ladder_struct)

    if !silent
        print_log(log)
    end

    state = DEMState(
        reseed!(copy(rng), rng), x, ld, adaptive_state, temperature_ladder_struct, memory, chain_models, n_tries,
        initialize_order_state(sampler)
    )

    return create_sample(state), state
end

function print_log(log::Vector{String})
    for l in log
        @info l
    end
    return
end
