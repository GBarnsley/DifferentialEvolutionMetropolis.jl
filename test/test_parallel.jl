using Distributed

@testset "ensemble × in-step threading" begin
    new_workers = addprocs(2; exeflags = `--project=$(Base.active_project()) --threads=2`)
    @everywhere new_workers using DifferentialEvolutionMetropolis, LogDensityProblems
    @everywhere begin
        struct EnsembleNormalModel
            mean::Vector{Float64}
        end
        LogDensityProblems.dimension(m::EnsembleNormalModel) = length(m.mean)
        LogDensityProblems.logdensity(m::EnsembleNormalModel, x::AbstractVector{<:Real}) = -sum(abs2, x .- m.mean) / 2
        LogDensityProblems.capabilities(::EnsembleNormalModel) = LogDensityProblems.LogDensityOrder{0}()
    end
    try
        @test all(fetch(@spawnat worker Threads.nthreads()) >= 2 for worker in new_workers)
        ld = AbstractMCMC.LogDensityModel(EnsembleNormalModel([-5.0, 5.0, 1.0]))
        flat(chains) = [(s.x, s.ld) for chain in chains for s in chain]
        ensembles = (MCMCSerial(), MCMCThreads(), MCMCDistributed())
        @testset "$scheme" for scheme in (:de, :de_snooker, :adaptive_subspace)
            sampler = scheme === :de ? setup_sampler_scheme(setup_de_update()) :
                scheme === :de_snooker ? setup_sampler_scheme(setup_de_update(), setup_snooker_update()) :
                setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update())
            run(ensemble, parallel; kwargs...) = sample(
                backwards_compat_rng(5), ld, sampler, ensemble, 100, 2;
                parallel = parallel, n_chains = 8, num_warmup = 50, progress = false, silent = true, kwargs...
            )
            reference = flat(run(MCMCSerial(), false))
            @testset "$(nameof(typeof(ensemble))) × parallel=$parallel" for ensemble in ensembles, parallel in (false, true)
                @test flat(run(ensemble, parallel)) == reference
                @test run(ensemble, parallel; save_final_state = true) isa Tuple
            end
        end
    finally
        rmprocs(new_workers)
    end
end
