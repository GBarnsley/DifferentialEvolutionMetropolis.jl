show_distribution(io::IO, d) = show(IOContext(io, :compact => true), d)
show_distribution(io::IO, d::Dirac{T}) where {T} = print(IOContext(io, :compact => true), d.value, "::", T)
function show_distribution(io::IO, d::Union{DiscreteNonParametric, CrossoverSampler})
    print(io, "DiscreteNonParametric(support = ")
    show(IOContext(io, :compact => true), Distributions.support(d))
    return print(io, ")")
end

function show_fields(io::IO, fields)
    width = maximum(length ∘ first, fields)
    for (name, value) in fields
        print(io, "\n  ", rpad(name * ":", width + 1), " ")
        value isa AbstractString ? print(io, value) : show_distribution(io, value)
    end
    return nothing
end

function show_multiline(io::IO, title::AbstractString, fields)
    print(io, title)
    get(io, :compact, false) || show_fields(io, fields)
    return nothing
end

# Samplers

function Base.show(io::IO, s::DifferentialEvolutionSampler)
    print(io, "DE update(γ = ")
    show_distribution(io, s.γ_spl)
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::DifferentialEvolutionSampler)
    return show_multiline(io, "Differential evolution update", ["γ" => s.γ_spl, "β" => s.β_spl])
end

function Base.show(io::IO, s::DifferentialEvolutionSnookerSampler)
    print(io, "Snooker update(γ = ")
    show_distribution(io, s.γ_spl)
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::DifferentialEvolutionSnookerSampler)
    return show_multiline(io, "Snooker update", ["γ" => s.γ_spl])
end

subspace_γ(::DifferentialEvolutionSubspaceSampler) = "2.38 / sqrt(2δd)"
subspace_γ(s::DifferentialEvolutionSubspaceSamplerFixedGamma) = string(s.γ)

is_adaptive(s::AbstractDifferentialEvolutionSubspaceSampler) = s.n_cr > 1

subspace_title(s::AbstractDifferentialEvolutionSubspaceSampler) =
    is_adaptive(s) ? "Adaptive subspace update" : "Subspace update"

subspace_cr(s::AbstractDifferentialEvolutionSubspaceSampler) =
    is_adaptive(s) ? "adaptive over $(s.n_cr) values" : sprint(show_distribution, s.cr_spl)

function Base.show(io::IO, s::AbstractDifferentialEvolutionSubspaceSampler)
    return print(io, subspace_title(s), "(γ = ", subspace_γ(s), ", cr = ", subspace_cr(s), ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::AbstractDifferentialEvolutionSubspaceSampler)
    fields = [
        "γ" => subspace_γ(s), "cr" => s.cr_spl, "n_cr" => string(s.n_cr),
        "δ" => s.δ_spl, "ϵ" => s.ϵ_spl, "e" => s.e_spl,
    ]
    if is_adaptive(s)
        push!(fields, "cr_uniform_weight" => string(s.cr_uniform_weight), "min_variance_count" => string(s.min_variance_count))
    end
    return show_multiline(io, subspace_title(s), fields)
end

function Base.show(io::IO, s::DifferentialEvolutionCompositeSampler)
    print(io, "Composite sampler(")
    join(io, (sprint(show, u; context = io) for u in s.updates), ", ")
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::DifferentialEvolutionCompositeSampler)
    n = length(s.updates)
    print(io, "Composite sampler with ", n, n == 1 ? " update" : " updates")
    get(io, :compact, false) && return nothing
    probabilities = s.update_weights ./ sum(s.update_weights)
    for (update, p) in zip(s.updates, probabilities)
        print(io, "\n  ", lpad(string(round(p; sigdigits = 3)), 6), " × ")
        show(io, update)
    end
    return nothing
end

# Internal components, shown as single-line summaries

Base.show(io::IO, ::DifferentialEvolutionAdaptiveStatic) = print(io, "static")

function Base.show(io::IO, a::DifferentialEvolutionAdaptiveSubspace)
    return print(io, "adaptive subspace (", length(a.L), " crossover probabilities)")
end

function Base.show(io::IO, a::DifferentialEvolutionAdaptiveComposite)
    print(io, "composite(")
    join(io, (sprint(show, s; context = io) for s in a.adaptive_states), ", ")
    return print(io, ")")
end

Base.show(io::IO, ::DifferentialEvolutionNullTemperatureLadder) = print(io, "none")

function Base.show(io::IO, l::DifferentialEvolutionStaticTemperatureLadder)
    n_hot = length(l.temperature) - length(l.cold_chains)
    return print(io, "parallel tempering (", n_hot, " hot chains)")
end

function Base.show(io::IO, l::DifferentialEvolutionAnnealingTemperatureLadder)
    return print(io, "annealing (", length(l.temperature_ladder) - 1, " steps remaining)")
end

Base.show(io::IO, ::DifferentialEvolutionMemoryless) = print(io, "none")

fill_method(::DifferentialEvolutionMemoryFillEvery) = "every iteration"
fill_method(f::DifferentialEvolutionMemoryFillThin) = "every $(f.max_count) iterations"

function Base.show(io::IO, m::DifferentialEvolutionMemoryFill)
    return print(
        io, m.fill.position, " stored positions (filled ", fill_method(m.fill),
        m.refill ? ", refills when full)" : ", grows when full)"
    )
end

function Base.show(io::IO, m::DifferentialEvolutionMemoryRefill)
    return print(
        io, length(m.mem_x), " stored positions (full, refilled ", fill_method(m.fill), ")"
    )
end

# Sampler state and output

function Base.show(io::IO, s::DifferentialEvolutionState{T}) where {T}
    return print(
        io, "DifferentialEvolutionState{", T, "}(", length(s.x), " chains, ",
        length(first(s.x)), " parameters)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", s::DifferentialEvolutionState{T}) where {T}
    n_chains = length(s.x)
    n_cold = length(s.x_smpl_view)
    chains = n_cold == n_chains ? string(n_chains) : "$n_chains ($n_cold cold)"
    fields = [
        "chains" => chains,
        "parameters" => string(length(first(s.x))),
        "adaptive state" => sprint(show, s.adaptive_state; context = io),
        "temperature" => sprint(show, s.temperature_ladder; context = io),
        "memory" => sprint(show, s.memory; context = io),
        "log density" => "max $(round(maximum(s.ld_smpl_view); sigdigits = 5))",
    ]
    return show_multiline(io, "DifferentialEvolutionState{$T}", fields)
end

function Base.show(io::IO, s::DifferentialEvolutionSample)
    return print(
        io, "DifferentialEvolutionSample(", length(s.x), " chains, ",
        length(first(s.x)), " parameters)"
    )
end

function Base.show(io::IO, o::DifferentialEvolutionOutput{T}) where {T}
    n_iter, n_chains, n_params = size(o.samples)
    return print(
        io, "DifferentialEvolutionOutput{", T, "}(", n_iter, " iterations, ",
        n_chains, " chains, ", n_params, " parameters)"
    )
end

function Base.show(io::IO, ::MIME"text/plain", o::DifferentialEvolutionOutput{T}) where {T}
    n_iter, n_chains, n_params = size(o.samples)
    fields = [
        "iterations" => string(n_iter),
        "chains" => string(n_chains),
        "parameters" => string(n_params),
    ]
    if !isempty(o.ld)
        push!(fields, "log density" => "mean $(round(StatsBase.mean(o.ld); sigdigits = 5)), range $(round.(extrema(o.ld); sigdigits = 5))")
    end
    return show_multiline(io, "DifferentialEvolutionOutput{$T}", fields)
end
