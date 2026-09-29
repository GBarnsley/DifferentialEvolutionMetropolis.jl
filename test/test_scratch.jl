@testset "Proposal scratch buffers" begin
    DEM = DifferentialEvolutionMetropolis
    # Allocations are only checked without coverage, and the helpers use the full package name as calls through `DEM` allocate
    check_allocations = Base.JLOptions().code_coverage == 0
    n_dims = 5
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(n_dims)))
    initial_state(update; n_chains = 6, memory = false) = last(
        AbstractMCMC.step(
            backwards_compat_rng(1), model, update; n_chains = n_chains, memory = memory
        )
    )

    @testset "buffers are only reallocated when chains or dimensions change" begin
        scratch = DEM.ProposalScratch{2}()
        DEM.ensure_size!(scratch, Float64, 4, 5)
        @test length(scratch.buffers) == 4
        @test all(length(buffer) == 5 for buffers in scratch.buffers for buffer in buffers)
        first_buffer = scratch.buffers[1][1]
        DEM.ensure_size!(scratch, Float64, 4, 5)
        @test scratch.buffers[1][1] === first_buffer
        DEM.ensure_size!(scratch, Float64, 8, 7)
        @test length(scratch.buffers) == 8
        @test all(length(buffer) == 7 for buffers in scratch.buffers for buffer in buffers)
        @test scratch.buffers[1][1] === first_buffer
        if check_allocations
            ensure_bytes(s) = @allocated DifferentialEvolutionMetropolis.ensure_size!(s, Float64, 8, 7)
            ensure_bytes(scratch)
            @test ensure_bytes(scratch) == 0
        end
    end

    @testset "buffers follow the element type of the positions" begin
        scratch = DEM.ProposalScratch{2}()
        DEM.ensure_size!(scratch, Float64, 4, 5)
        DEM.ensure_size!(scratch, Float32, 4, 5)
        @test scratch.buffers isa Vector{NTuple{2, Vector{Float32}}}
        @test length(scratch.buffers) == 4
        @test all(length(buffer) == 5 for buffers in scratch.buffers for buffer in buffers)
        @test DEM.chain_buffers(scratch, Float32, 1) === scratch.buffers[1]
        if check_allocations
            typed_ensure_bytes(s) = @allocated DifferentialEvolutionMetropolis.ensure_size!(s, Float32, 4, 5)
            typed_ensure_bytes(scratch)
            @test typed_ensure_bytes(scratch) == 0
        end
    end

    updates = (
        ("DE", setup_de_update(n_dims = n_dims)),
        ("snooker", setup_snooker_update()),
        ("subspace", setup_subspace_sampling()),
        ("subspace fixed γ", setup_subspace_sampling(γ = 1.0)),
    )
    @testset "$name proposals do not allocate" for (name, update) in updates
        for memory in (false, true), T in (Float64, Float32)
            state = last(
                AbstractMCMC.step(
                    backwards_compat_rng(1), model, update;
                    n_chains = 6, memory = memory,
                    initial_position = [randn(backwards_compat_rng(i), T, n_dims) for i in 1:6]
                )
            )
            DEM.prepare_scratch!(update, state)
            @test length(update.scratch.buffers) == length(state.x)
            @test update.scratch.buffers isa Vector{<:NTuple{<:Any, Vector{T}}}
            if check_allocations
                proposal_bytes(s, u) = @allocated DifferentialEvolutionMetropolis.proposal!(s, u, 1)
                proposal_bytes(state, update)
                @test proposal_bytes(state, update) == 0
            end
        end
    end

    @testset "a reused update resizes its buffers for more chains and dimensions" begin
        update = setup_subspace_sampling()
        rng = backwards_compat_rng(2)
        _, state = AbstractMCMC.step(rng, model, update; n_chains = 4, memory = false)
        _, state = AbstractMCMC.step(rng, model, update, state)
        @test all(length(buffer) == n_dims for buffers in update.scratch.buffers for buffer in buffers)
        larger = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(8)))
        _, state = AbstractMCMC.step(rng, larger, update; n_chains = 10, memory = false)
        _, state = AbstractMCMC.step(rng, larger, update, state)
        @test length(update.scratch.buffers) == 10
        @test all(length(buffer) == 8 for buffers in update.scratch.buffers for buffer in buffers)
        @test all(length(x) == 8 for x in state.x)
    end

    @testset "subspace proposals only change the selected dimensions" begin
        for (cr, expected) in ((1.0e-9, 1), (1.0, n_dims))
            update = setup_subspace_sampling(cr = cr)
            state = initial_state(update)
            DEM.prepare_scratch!(update, state)
            for _ in 1:20, i in eachindex(state.x)
                DEM.proposal!(state, update, i)
                @test count(state.xₚ[i] .!= state.x[i]) == expected
            end
        end
    end

    @testset "equality and hashing ignore the buffers" begin
        a = setup_de_update(n_dims = n_dims)
        b = setup_de_update(n_dims = n_dims)
        DEM.prepare_scratch!(a, initial_state(a))
        @test a == b
        @test hash(a) == hash(b)
        @test a != setup_de_update(γ = 0.5, n_dims = n_dims)
        @test setup_subspace_sampling(cr = 0.5) == setup_subspace_sampling(cr = 0.5)
        empty_scratch = DEM.ProposalScratch{2}()
        @test empty_scratch == a.scratch
        @test hash(empty_scratch) == hash(a.scratch)
    end

    @testset "composite sampler needs no proposal buffers of its own" begin
        update = setup_de_update(n_dims = n_dims)
        composite = setup_sampler_scheme(update)
        state = initial_state(update)
        @test DEM.prepare_scratch!(composite, state) === nothing
        @test isempty(update.scratch.buffers)
    end

    @testset "reseed! is deterministic for Xoshiro and other rngs" begin
        for chain_rng in (Random.Xoshiro(0), Random.MersenneTwister(0))
            a = DEM.reseed!(copy(chain_rng), backwards_compat_rng(3))
            b = DEM.reseed!(copy(chain_rng), backwards_compat_rng(3))
            c = DEM.reseed!(copy(chain_rng), backwards_compat_rng(4))
            @test typeof(a) == typeof(chain_rng)
            draws = rand(a, 5)
            @test draws == rand(b, 5)
            @test draws != rand(c, 5)
        end
        if check_allocations
            reseed_bytes(chain_rng, rng) = @allocated DifferentialEvolutionMetropolis.reseed!(chain_rng, rng)
            chain_rng, rng = Random.Xoshiro(0), Random.Xoshiro(1)
            reseed_bytes(chain_rng, rng)
            @test reseed_bytes(chain_rng, rng) == 0
        end
    end
end
