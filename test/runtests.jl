include("groups.jl")

# Slow tests run only in the all group, which CI runs on every Julia version.
const ALL_ONLY_FILES = ["test_correct.jl"]

const TEST_GROUP = Symbol(get(ENV, "TEST_GROUP", "all"))
const TEST_DOWNGRADE = parse(Bool, get(ENV, "TEST_DOWNGRADE", "false"))
activate_test_env(TEST_GROUP; downgrade = TEST_DOWNGRADE)
Pkg.status()

include("setup.jl")

@testset "DifferentialEvolutionMetropolis.jl" begin
    for group in (TEST_GROUP === :all ? keys(GROUPS) : (TEST_GROUP,))
        @testset "$group" begin
            for file in GROUPS[group].files
                TEST_GROUP !== :all && file in ALL_ONLY_FILES && continue
                suppress_warnings(MEMORY_SIZE_WARNING, MEMORY_REFILL_WARNING) do
                    include(file)
                end
            end
        end
    end
end
