using Documenter, DifferentialEvolutionMetropolis, DocumenterInterLinks

links = InterLinks(
    "MCMCDiagnosticTools" => "https://turinglang.org/MCMCDiagnosticTools.jl/stable/objects.inv"
);

makedocs(
    sitename = "Differential Evolution Metropolis",
    plugins = [links],
    pages = [
        "index.md",
        "tutorial.md",
        "custom.md",
    ],
    modules = [DifferentialEvolutionMetropolis]
)

deploydocs(
    repo = "github.com/GBarnsley/DifferentialEvolutionMetropolis.jl.git",
)
