using Distributed, Serialization

@testset "parallel backends" begin
    DEM = DifferentialEvolutionMetropolis

    @testset "backend normalisation" begin
        @test DEM.parallel_backend(false) === MCMCSerial()
        @test DEM.parallel_backend(true) === MCMCThreads()
        @test DEM.parallel_backend(MCMCThreads()) === MCMCThreads()
        @test DEM.parallel_backend(MCMCDistributed()) isa DEM.DistributedBackend
        current = DEM.parallel_backend(MCMCDistributed())
        @test DEM.parallel_backend(MCMCDistributed(), current) === current
        @test DEM.parallel_backend(MCMCSerial(), current) === MCMCSerial()
        @test_throws ArgumentError DEM.parallel_backend(:threads)
    end

    @testset "initialization stores the backend" begin
        ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))
        _, state = AbstractMCMC.step(
            backwards_compat_rng(1), ld, setup_de_update(); parallel = true, silent = true
        )
        @test state.parallel_backend === MCMCThreads()
        _, state = AbstractMCMC.step(backwards_compat_rng(1), ld, setup_de_update(), state)
        @test state.parallel_backend === MCMCThreads()
    end

    # Workers need the package and the model type loaded
    new_workers = addprocs(2; exeflags = "--project=$(Base.active_project())")
    @everywhere new_workers begin
        using DifferentialEvolutionMetropolis, LogDensityProblems
    end
    @everywhere begin
        struct DistributedNormalModel
            mean::Vector{Float64}
        end
        LogDensityProblems.dimension(m::DistributedNormalModel) = length(m.mean)
        LogDensityProblems.logdensity(m::DistributedNormalModel, x::AbstractVector{<:Real}) = -sum(abs2, x .- m.mean) / 2
        LogDensityProblems.capabilities(::DistributedNormalModel) = LogDensityProblems.LogDensityOrder{0}()
    end

    try
        ld = AbstractMCMC.LogDensityModel(DistributedNormalModel([-5.0, 5.0, 1.0]))
        backends = (false, true, MCMCSerial(), MCMCThreads(), MCMCDistributed())

        @testset "$(nameof(template)) gives identical chains on every backend" for template in (deMC, deMCzs, DREAMz)
            outs = [
                template(
                    ld, 200; rng = backwards_compat_rng(11), n_chains = 8, parallel = parallel,
                    silent = true, progress = false
                ) for parallel in backends
            ]
            @test all(out.samples == outs[1].samples for out in outs)
            @test all(out.ld == outs[1].ld for out in outs)
        end

        @testset "parallel tempering gives identical chains on every backend" begin
            outs = [
                deMC(
                    ld, 200; rng = backwards_compat_rng(3), n_chains = 6, n_hot_chains = 2, parallel = parallel,
                    silent = true, progress = false
                ) for parallel in backends
            ]
            @test all(out.samples == outs[1].samples for out in outs)
        end

        @testset "distributed state keeps one worker pool" begin
            sampler = setup_sampler_scheme(setup_de_update())
            _, state = AbstractMCMC.step(
                backwards_compat_rng(1), ld, sampler; n_chains = 6, parallel = MCMCDistributed(), silent = true
            )
            pool = state.parallel_backend
            @test pool isa DEM.DistributedBackend
            _, state = AbstractMCMC.step(
                backwards_compat_rng(1), ld, sampler, state; parallel = MCMCDistributed()
            )
            @test state.parallel_backend === pool
        end

        flat(chains) = [(s.x, s.ld) for chain in chains for s in chain]
        run_ensemble(ensemble, parallel; kwargs...) = sample(
            backwards_compat_rng(5), ld, setup_sampler_scheme(setup_de_update()), ensemble, 50, 2;
            parallel = parallel, n_chains = 6, progress = false, silent = true, kwargs...
        )
        reference = flat(run_ensemble(MCMCSerial(), MCMCSerial()))
        ensembles = (MCMCSerial(), MCMCThreads(), MCMCDistributed())
        @testset "ensemble $(nameof(typeof(ensemble))) × in-step $(nameof(typeof(parallel))) matches serial" for ensemble in ensembles, parallel in ensembles
            @test flat(run_ensemble(ensemble, parallel)) == reference
        end

        @testset "distributed backend survives serialisation" begin
            _, state = AbstractMCMC.step(
                backwards_compat_rng(1), ld, setup_sampler_scheme(setup_de_update());
                n_chains = 6, parallel = MCMCDistributed(), silent = true
            )
            io = IOBuffer()
            serialize(io, state)
            new_state = deserialize(seekstart(io))
            @test new_state.parallel_backend isa DEM.DistributedBackend
            @test new_state.parallel_backend !== state.parallel_backend
            @test new_state.x == state.x
            @test run_ensemble(MCMCDistributed(), MCMCDistributed(); save_final_state = true) isa Tuple
        end
    finally
        rmprocs(new_workers)
    end
end
