include("groups.jl")

"""
    resolved_versions(group::Symbol)

Versions of every package resolved in the generated environment for `group`, keyed by name.
"""
function resolved_versions(group::Symbol)
    activate_test_env(group)
    return Dict(p.name => p.version for p in values(Pkg.dependencies()))
end

"""
    package_deps()

Names of the direct dependencies of DifferentialEvolutionMetropolis in the active environment.
"""
function package_deps()
    package = only(p for p in values(Pkg.dependencies()) if p.name == "DifferentialEvolutionMetropolis")
    return collect(keys(package.dependencies))
end

"""
    differing_groups()

Groups whose isolated tests cover versions the `all` group does not: `core` when one of the
package's direct dependencies resolves to a different version in `all`, and an extension group when
one of its own test dependencies does.
"""
function differing_groups()
    all_versions = resolved_versions(:all)
    return filter(collect(keys(GROUPS))) do group
        versions = resolved_versions(group)
        compared = group === :core ? package_deps() : GROUPS[group].deps
        return any(name -> versions[name] != get(all_versions, name, nothing), compared)
    end
end

groups = "[" * join(("\"$g\"" for g in differing_groups()), ",") * "]"
println("Differing groups: ", groups)
haskey(ENV, "GITHUB_OUTPUT") && open(io -> println(io, "groups=", groups), ENV["GITHUB_OUTPUT"], "a")
