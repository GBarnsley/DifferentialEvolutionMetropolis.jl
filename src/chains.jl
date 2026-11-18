"""
    DEMAdaptiveStatic{T}

Adaptive state of an update that does not adapt, or of an adaptive update run with `adapt = false`.
"""
struct DEMAdaptiveStatic{T} <:
    AbstractDEMAdaptiveState{T} end

"""
    DEMSample

Sample returned by each step, with fields `x` (position of each output chain) and `ld` (log density at each position).

Hot chains are excluded when parallel tempering is used.
"""
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

"""
    pick_chains(state::DEMState, current_state::Int, ::Val{N})

Return a tuple of `N` distinct positions, drawn without replacement using `state.rngs[current_state]`.

Without memory these are other live chains, excluding the current chain; with memory they are drawn from the memory archive.
With `double_ensemble = true` they are chains of the other ensemble: an odd chain picks the current positions of even chains, an even chain picks the updated positions of odd chains.
If `N` exceeds `n_preallocated_indices` (default 3, see [`deMC`](@ref)) the index buffers are resized with a warning.
"""
function pick_chains end

# Chains are updated in order, so chains before the current chain have their new position in `xₚ`
@generated function pick_chains(
        state::DEMState{T, A, L, M},
        current_state::Int,
        ::Val{N}
    ) where {T, A, L, M <: DEMMemoryless, N}
    exprs = [:(indices[$i] < current_chain ? xₚ[indices[$i]] : x[indices[$i]]) for i in 1:N]
    return quote
        rng = state.rngs[current_state]
        x = state.x
        xₚ = state.xₚ
        memory = state.memory
        indices = memory.indices_INTERNAL[current_state]
        ordered_indices = memory.ordered_indices_INTERNAL[current_state]
        current_chain = mod1(current_state, length(x))
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

# Odd chains are updated before even chains, so odd chains have their new position in `xₚ`; trials share their chain's parity as the chain count is even
@generated function pick_chains(
        state::DEMState{T, A, L, M},
        current_state::Int,
        ::Val{N}
    ) where {T, A, L, M <: DEMDoubleEnsemble, N}
    exprs = [:(odd_chain ? x[2 * indices[$i]] : xₚ[2 * indices[$i] - 1]) for i in 1:N]
    return quote
        rng = state.rngs[current_state]
        x = state.x
        xₚ = state.xₚ
        memory = state.memory
        indices = memory.indices_INTERNAL[current_state]
        ordered_indices = memory.ordered_indices_INTERNAL[current_state]
        odd_chain = isodd(current_state)
        fast_sample_chains!(
            rng,
            x,
            length(x) ÷ 2,
            $N,
            indices,
            ordered_indices
        )
        return ($(exprs...),)
    end
end

@generated function pick_chains(
        state::DEMState{T, A, L, M, Model, B, V, VV},
        current_state::Int,
        ::Val{N}
    ) where {T, A, L, V, VV, Model, B, M <: DEMMemoryFill{T, VV}, N}
    exprs = [:(mem_x[indices[$i]]) for i in 1:N]
    return quote
        rng = state.rngs[current_state]
        memory = state.memory
        mem_x = memory.mem_x
        indices = memory.indices_INTERNAL[current_state]
        ordered_indices = memory.ordered_indices_INTERNAL[current_state]
        fast_sample_chains!(
            rng,
            mem_x,
            memory.fill.stored,
            $N,
            indices,
            ordered_indices
        )
        return ($(exprs...),)
    end
end

# Swaps x and xₚ in place and advances the ladder and, when `update_memory`, the memory
function update_state!(state::DEMState; update_memory::Bool = false)
    update_memory && update_memory!(state.memory, state.xₚ)
    for i in eachindex(state.x)
        state.x[i], state.xₚ[i] = state.xₚ[i], state.x[i]
        state.ld[i], state.ldₚ[i] = state.ldₚ[i], state.ld[i]
    end
    update_ladder!(state.temperature_ladder)
    return nothing
end

# Sequential updates without multiple tries run one chain at a time even with `parallel = true`
cannot_use_parallel(sampler::AbstractDEMSampler{DEMSamplerSerial, DEMSamplerSingleTry}, n_tries::Int) = true
cannot_use_parallel(sampler::AbstractDEMSampler{DEMSamplerSerial, DEMSamplerMultipleTry}, n_tries::Int) = !(n_tries > 1)
cannot_use_parallel(sampler::AbstractDEMSampler{DEMSamplerMultiThread}, n_tries::Int) = false

function check_parallel(sampler::AbstractDEMMetaSampler, parallel::Bool, memory::Bool, n_tries::Int, double_ensemble::Bool = false)
    if double_ensemble
        if memory
            error("`double_ensemble = true` requires `memory = false`, memory-based samplers already update chains in parallel.")
        elseif !parallel
            @warn "`double_ensemble = true` but `parallel = false`, the double ensemble only speeds up sampling when chains update in parallel."
        end
    elseif parallel && !memory && cannot_use_parallel(sampler, n_tries)
        @warn "None of the provided samplers can evaluate in parallel, yet `parallel = true`, consider `memory = true`, `double_ensemble = true` or multi-try updates to use multithreading."
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

function propose_and_update_chain!(state, sampler, adaptive_state, i)
    proposal = proposal!(state.xₚ[i], state.x[i], state, sampler, adaptive_state, i)
    record_proposal!(adaptive_state, proposal, i)
    update_chain!(state.chain_models[i], state, proposal.offset, i)
    return nothing
end

# Calls f on each chain, ensemble by ensemble, threading within an ensemble when parallel
function foreach_chain(f, state, sampler, parallel::Bool)
    for chains in chain_ensembles(state.memory)
        foreach_threaded(f, chains, parallel)
    end
    return nothing
end

# Memoryless chains with a sequential update pick from the updated positions of earlier chains, so they update in turn
function foreach_chain(
        f, state::DEMState{<:Any, <:Any, <:Any, <:DEMMemoryless},
        sampler::AbstractDEMSampler{DEMSamplerSerial}, parallel::Bool
    )
    foreach(f, eachindex(state.x))
    return nothing
end

function update_chains!(state, sampler, adaptive_state, parallel::Bool)
    foreach_chain(i -> propose_and_update_chain!(state, sampler, adaptive_state, i), state, sampler, parallel)
    return nothing
end

# multi-try methods are defined in src/multi_try.jl

"""
    step(rng, model_wrapper, sampler, state; parallel=false, update_memory=true, kwargs...)

Perform a single MCMC step using differential evolution sampling.

This is the core sampling function that proposes new states for all chains and accepts
or rejects them according to the Metropolis criterion. Adaptive updates propose with their
current adapted values but do not adapt them further, adaptation only happens in [`step_warmup`](@ref).

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler from this package
- `state`: Current state of all chains

# Keyword Arguments
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`. Advisable for slow models.
  With a memoryless state, updates other than HMC update the chains in turn and only evaluate each chain's multiple-try proposals in parallel, unless the state was set up with `double_ensemble = true`, which updates each ensemble's chains in parallel.
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
        state::DEMState;
        adaptive_state::AbstractDEMAdaptiveState = state.adaptive_state,
        parallel::Bool = false,
        update_memory::Bool = true,
        kwargs...
    )
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
    update_chains!(state, sampler, adaptive_state, parallel)

    sample = create_sample(state)
    update_state!(state; update_memory = update_memory)
    return sample, state
end

"""
    step_warmup(rng, model_wrapper, sampler, state; parallel=false, update_memory=true, kwargs...)

Perform a single MCMC step during the warm-up (adaptive) phase.

During warm-up, this function performs the same sampling as [`step`](@ref), then adapts the update's
adaptive state with [`DifferentialEvolutionMetropolis.adapt!`](@ref). For subspace samplers, it adapts
crossover probabilities based on the effectiveness of different parameter subsets. Only cold chains
contribute to the adaptation, and jumps are recorded once the running variance is available.

# Arguments
- `rng`: Random number generator
- `model_wrapper`: LogDensityModel containing the target log-density function
- `sampler`: Differential evolution sampler from this package
- `state`: Current state including adaptive parameters

# Keyword Arguments
- `update_memory`: Whether to update the memory with new positions (for memory-based samplers).
  Defaults to `true`. Useful if memory has grown too large.
- `parallel`: Whether to run chains in parallel using threading. Defaults to `false`.
- `kwargs...`: Additional keyword arguments passed to [`step`](@ref)

# Returns
- `sample`: DEMSample containing new positions and log-densities
- `new_state`: Updated state with adapted parameters for the next iteration

# Example
```@example step_warmup
using DifferentialEvolutionMetropolis, AbstractMCMC, Random
using TransformedLogDensities, TransformVariables

# Setup for warmup step example
rng = Random.default_rng()
ld = TransformedLogDensity(as(Array, 2), θ -> -sum(abs2, θ) / 2)
model_wrapper = AbstractMCMC.LogDensityModel(ld)
sampler = setup_subspace_sampling()

# Initialize state and perform a warmup step
_, state = AbstractMCMC.step(rng, model_wrapper, sampler)
sample, new_state = AbstractMCMC.step_warmup(rng, model_wrapper, sampler, state; parallel=false)
```

See also [`step`](@ref).
"""
function step_warmup(
        rng::AbstractRNG,
        model_wrapper::LogDensityModel,
        sampler::AbstractDEMMetaSampler,
        state::DEMState;
        adaptive_state::AbstractDEMAdaptiveState = state.adaptive_state,
        kwargs...
    )
    sample, new_state = step(rng, model_wrapper, sampler, state; adaptive_state, kwargs...)
    adapt!(adaptive_state, sampler, new_state)
    return sample, new_state
end

"""
    adapt!(adaptive_state::AbstractDEMAdaptiveState, sampler::AbstractDEMMetaSampler, state::DEMState)

Adapt `adaptive_state` in place after a warm-up step of `sampler`.

`state` is the state after the step, so `state.x` holds the positions after the step and `state.xₚ` the positions before it.
The fallback does nothing. An adaptive update defines a method for its own adaptive state.
"""
adapt!(adaptive_state::AbstractDEMAdaptiveState, sampler::AbstractDEMMetaSampler, state) = nothing

"""
    record_proposal!(adaptive_state::AbstractDEMAdaptiveState, proposal::NamedTuple, current_state::Int)

Record what [`DifferentialEvolutionMetropolis.adapt!`](@ref) needs from `proposal`, the named tuple returned by [`DifferentialEvolutionMetropolis.proposal!`](@ref) for `current_state`.

Called for every proposal a chain can move to, but not for the reference trials of multiple tries, and also after warm-up.
`current_state` runs up to `n_chains * n_tries`, so size any buffer indexed by it in [`DifferentialEvolutionMetropolis.initialize_adaptive_state`](@ref).
The fallback records nothing.
"""
record_proposal!(adaptive_state::AbstractDEMAdaptiveState, proposal, current_state::Int) = nothing

"""
    initialize_adaptive_state(sampler::AbstractDEMMetaSampler, model_wrapper::LogDensityModel, n_chains::Int, n_tries::Int, ::Type{T})

Create the adaptive state of `sampler` at the start of sampling, for `n_chains` chains each proposing `n_tries` trials, with `T` the position element type.

The fallback returns a static (non-adaptive) state.
An adaptive update defines a method returning its own [`DifferentialEvolutionMetropolis.AbstractDEMAdaptiveState`](@ref),
a [`DifferentialEvolutionMetropolis.proposal!`](@ref) method for that state, and a [`DifferentialEvolutionMetropolis.adapt!`](@ref) method that adapts it.
"""
function initialize_adaptive_state(
        sampler::AbstractDEMMetaSampler,
        model_wrapper::LogDensityModel, n_chains::Int, n_tries::Int, ::Type{T}
    ) where {T <: Real}
    return DEMAdaptiveStatic{T}()
end

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
        double_ensemble::Bool = false,
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
    check_parallel(sampler, parallel, memory, n_tries, double_ensemble)
    n_true_chains = (double_ensemble ? 2 : 1) * (n_chains + n_hot_chains)
    default_N₀ = cld(10 * dimension(model), n_chains + n_hot_chains) * (n_chains + n_hot_chains)
    initial_position = resolve_initial_position(
        initial_position, rng, n_true_chains,
        memory ? something(N₀, max(default_N₀ - n_chains - n_hot_chains, 0)) : 0;
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
    if double_ensemble
        temperature_ladder = [repeat(ladder_rung; inner = 2) for ladder_rung in temperature_ladder]
    end

    log = Vector{String}()

    if adapt
        adaptive_state = initialize_adaptive_state(sampler, model_wrapper, n_true_chains, n_tries, T)
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
        elseif n_hot_chains == 0 && (!double_ensemble || iseven(current_N))
            push!(log, "   Assuming initial position size is n_chains. Ignoring extra positions.")
            #assume n_chains is wrong
            n_true_chains = current_N
            x = initial_position
        else
            print_log(log)
            error("   Initial position size greater than the number of chains ($n_true_chains). Cannot resolve mismatch.")
        end
    end

    n_ensemble_chains = double_ensemble ? n_true_chains ÷ 2 : n_true_chains
    if !memory && n_ensemble_chains < chains_required(sampler)
        error("Memoryless scheme with sampler $(typeof(sampler)) requires at least $(chains_required(sampler)) chains, but only $n_ensemble_chains were provided (including hot chains).")
    end

    if n_ensemble_chains < dimension(model) && !memory
        @warn "In a memoryless model the number of chains should be greater than or equal to the number of parameters"
    end

    chain_models = [deepcopy(model) for _ in 1:(length(x) * n_tries)]
    ld = Vector{T}(undef, length(x))
    let x = x
        foreach_threaded(eachindex(x), parallel) do i
            ld[i] = checked_logdensity(chain_models[i], x[i])
        end
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
            N₀ = max(default_N₀, length(mem_x))
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

        memory_increment = memory_size * n_true_chains
        if memory_refill && memory_increment < N₀ + n_true_chains
            error("memory_size (currently $memory_size) is too small for the initial memory N₀ (currently $N₀) and number of chains (currently $n_true_chains). With memory_refill = true the total memory size (memory_size * n_chains = $memory_increment) must be at least N₀ + n_chains = $(N₀ + n_true_chains). Consider specifying a larger memory_size.")
        elseif memory_size < 1
            error("memory_size (currently $memory_size) must be at least 1.")
        end
        total_memory_size = memory_refill ? memory_increment : N₀ + memory_increment
        if (memory_size == 1001) & memory_refill
            push!(log, "   Using memory refill with default memory size of 1001, storing a maximum of $total_memory_size chains.")
            push!(log, "   Consider tailoring memory_size keyword argument to control memory usage!")
        elseif (memory_size == (num_warmup * 2)) & memory_refill
            push!(log, "   Using memory refill with default memory size of $memory_size (2x num_warmup/n_burnin) to store a maximum of $total_memory_size chains.")
            push!(log, "   Consider tailoring memory_size keyword argument to control memory usage!")
        end
        check_memory_size!(log, sampler, n_true_chains, N₀, num_warmup, memory_thin_interval, memory_refill, total_memory_size)
        true_memory = vcat(mem_x, [similar(mem_x[1]) for _ in 1:(total_memory_size - N₀)])

        #setup fill method
        if memory_thin_interval > 0
            push!(log, "   Using thinning interval of $memory_thin_interval for memory updates.")
            memory_method = DEMMemoryFillThin(N₀, n_true_chains, memory_thin_interval)
        else
            memory_method = DEMMemoryFillEvery(N₀, n_true_chains)
        end

        memory = DEMMemoryFill{T, typeof(mem_x), typeof(memory_method)}(
            (1:n_true_chains,), true_memory, memory_method, memory_refill, memory_increment,
            [fill(1, n_preallocated_indices) for _ in 1:(n_true_chains * n_tries)],
            [Vector{Int}(undef, n_preallocated_indices - 1) for _ in 1:(n_true_chains * n_tries)]
        )
    elseif double_ensemble
        memory = DEMDoubleEnsemble{T}(
            (1:2:n_true_chains, 2:2:n_true_chains),
            [fill(1, n_preallocated_indices) for _ in 1:(n_true_chains * n_tries)],
            [Vector{Int}(undef, n_preallocated_indices - 1) for _ in 1:(n_true_chains * n_tries)],
        )
    else
        memory = DEMMemoryless{T}(
            (1:n_true_chains,),
            [fill(1, n_preallocated_indices) for _ in 1:(n_true_chains * n_tries)],
            [Vector{Int}(undef, n_preallocated_indices) for _ in 1:(n_true_chains * n_tries)],
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
