using Pkg

const COMMON_DEPS = ["AbstractMCMC", "Distributions", "LogDensityProblems", "Logging", "Random", "StatsBase", "Test"]

const GROUPS = (;
    core = (;
        deps = ["Aqua", "Distributed"],
        files = [
            "test_aqua.jl",
            "test_differential_evolution_update.jl",
            "test_snooker_update.jl",
            "test_subspace_update.jl",
            "test_subspace_adaptive_update.jl",
            "test_multi_try.jl",
            "test_scratch.jl",
            "test_composite.jl",
            "test_fixed_order.jl",
            "test_rng.jl",
            "test_threadsafe.jl",
            "test_parallel.jl",
            "test_swapping.jl",
            "test_temperature.jl",
            "test_swap_update.jl",
            "test_types.jl",
            "test_show.jl",
            "test_memory_accounting.jl",
            "test_correct.jl",
            "test_no_extensions.jl",
            "test_templates.jl",
            "test_logdensity.jl",
        ],
    ),
    hmc = (;
        deps = ["AdvancedHMC", "ForwardDiff", "LinearAlgebra", "LogDensityProblemsAD"],
        files = [
            "hmc_setup.jl",
            "test_hmc.jl",
            "test_hmc_memory_metric.jl",
            "test_hmc_cluster_pooled_metric.jl",
            "test_hmc_per_cluster_metric.jl",
            "test_types_hmc.jl",
            "test_show_hmc.jl",
        ],
    ),
    mcmcdiagnostictools = (;
        deps = ["MCMCDiagnosticTools"],
        files = ["test_convergence.jl", "test_templates_mcmcdiagnostictools.jl"],
    ),
    mcmcchains = (; deps = ["MCMCChains"], files = ["test_templates_mcmcchains.jl"]),
    flexichains = (; deps = ["FlexiChains"], files = ["test_templates_flexichains.jl"]),
    pathfinder = (; deps = ["Pathfinder", "ForwardDiff", "LinearAlgebra"], files = ["test_pathfinder.jl"]),
)

"""
    group_deps(group::Symbol)

Test dependencies of `group`: `COMMON_DEPS` plus the group's own, or every group's for `:all`.
"""
function group_deps(group::Symbol)
    group === :all && return unique(vcat(COMMON_DEPS, (g.deps for g in GROUPS)...))
    haskey(GROUPS, group) ||
        throw(ArgumentError("unknown test group $group, expected all or one of: $(join(keys(GROUPS), ", "))"))
    return vcat(COMMON_DEPS, GROUPS[group].deps)
end

const PROJECT = Pkg.Types.read_project(joinpath(dirname(@__DIR__), "Project.toml"))

"""
    lowest_version(name::String)

Lowest registered, non-yanked version of `name` allowed by the package's `[compat]`.
"""
function lowest_version(name::String)
    uuid = get(PROJECT.deps, name, get(PROJECT.weakdeps, name, nothing))
    spec = PROJECT.compat[name].val
    for reg in Pkg.Registry.reachable_registries()
        haskey(reg, uuid) || continue
        entry = reg[uuid]
        # Newer Pkg versions take the registry as well as the entry.
        info = applicable(Pkg.Registry.registry_info, reg, entry) ?
            Pkg.Registry.registry_info(reg, entry) : Pkg.Registry.registry_info(entry)
        versions = [v for (v, vinfo) in info.version_info if v in spec && !vinfo.yanked]
        isempty(versions) || return minimum(versions)
    end
    error("no registered version of $name satisfies its compat $(PROJECT.compat[name].str)")
end

"""
    lower_bounds(group::Symbol)

Versions a downgrade run of `group` pins: the non-stdlib direct dependencies for `:core`, the
extension trigger packages for an extension group, each at its `lowest_version`.
"""
function lower_bounds(group::Symbol)
    names = if group === :core
        [name for (name, uuid) in PROJECT.deps if !Pkg.Types.is_stdlib(uuid)]
    elseif haskey(GROUPS, group)
        filter(in(keys(PROJECT.weakdeps)), GROUPS[group].deps)
    else
        throw(ArgumentError("downgrade runs take a single group, got $group"))
    end
    return Dict(name => lowest_version(name) for name in names)
end

"""
    activate_test_env(group::Symbol; downgrade::Bool = false)

Activate the generated environment for `group` under `test/environments/v<julia version>/`, with the
local package developed and `group_deps(group)` added. With `downgrade`, the environment is
`<group>-downgrade` and pins `lower_bounds(group)`. The environment is rebuilt from scratch when its
dependency list or pins no longer match, and otherwise reused with its resolved versions.
"""
function activate_test_env(group::Symbol; downgrade::Bool = false)
    deps = group_deps(group)
    pins = downgrade ? lower_bounds(group) : Dict{String, VersionNumber}()
    env = joinpath(
        @__DIR__, "environments", "v$(VERSION.major).$(VERSION.minor)",
        downgrade ? "$group-downgrade" : string(group)
    )
    Pkg.activate(env)
    if Set(keys(Pkg.project().dependencies)) != Set(["DifferentialEvolutionMetropolis"; deps; collect(keys(pins))]) ||
            any(info -> get(pins, info.name, info.version) != info.version, values(Pkg.dependencies()))
        rm(env; recursive = true, force = true)
        Pkg.activate(env)
        Pkg.develop(PackageSpec(; path = dirname(@__DIR__)))
        Pkg.add(
            [
                haskey(pins, dep) ? PackageSpec(; name = dep, version = pins[dep]) : PackageSpec(; name = dep)
                    for dep in union(deps, keys(pins))
            ]
        )
    end
    Pkg.instantiate()
    return env
end
