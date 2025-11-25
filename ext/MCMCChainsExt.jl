module MCMCChainsExt

import DifferentialEvolutionMetropolis
import MCMCChains: Chains, replacenames
import AbstractMCMC

function DifferentialEvolutionMetropolis.convert(
        ::Type{Chains},
        samples::Vector{DifferentialEvolutionMetropolis.DifferentialEvolutionSample{V, VV}}
    ) where {T <: Real, V <: AbstractVector{T}, VV <: AbstractVector{V}}
    output = DifferentialEvolutionMetropolis.process_outputs(samples)

    new_ld = Array{T, 3}(undef, size(output.ld, 1), 1, size(output.ld, 2))
    #can replace with insertdims(output.ld, dims = 2) in julia 1.12+
    for i in 1:size(output.ld, 1)
        for j in 1:size(output.ld, 2)
            new_ld[i, 1, j] = output.ld[i, j]
        end
    end

    array_out = cat(
        permutedims(output.samples, (1, 3, 2)),
        new_ld, dims = 2
    )

    chns = Chains(array_out)
    chns = replacenames(
        chns, Dict(
            zip(
                ["param_$i" for i in axes(array_out, 2)], vcat(DifferentialEvolutionMetropolis.generate_names(size(output.samples, 3)), ["ld"])
            )
        )
    )

    return chns
end


end
