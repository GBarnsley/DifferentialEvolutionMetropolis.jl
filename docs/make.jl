using Documenter, DifferentialEvolutionMetropolis, DocumenterInterLinks

ENV["GKSwstype"] = "100" #prevents plots from opening windows

links = InterLinks(
    "MCMCDiagnosticTools" => "https://turinglang.org/MCMCDiagnosticTools.jl/stable/objects.inv"
);

makedocs(
    sitename = "Differential Evolution Metropolis",
    plugins = [links],
    pages = [
        "index.md",
        "tutorial.md",
        "hmc.md",
        "custom.md",
        "pathfinder.md",
        "turing.md",
    ],
    modules = [DifferentialEvolutionMetropolis]
)

deploydocs(
    repo = "github.com/GBarnsley/DifferentialEvolutionMetropolis.jl.git",
)
