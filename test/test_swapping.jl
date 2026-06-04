struct SwapCheckNormal end
LogDensityProblems.dimension(::SwapCheckNormal) = 3
LogDensityProblems.logdensity(::SwapCheckNormal, x) = -sum(abs2, x) / 2
LogDensityProblems.capabilities(::SwapCheckNormal) = LogDensityProblems.LogDensityOrder{0}()

@testset "x/xₚ swapping keeps the sample views consistent" begin
    model = AbstractMCMC.LogDensityModel(SwapCheckNormal())

    # After every warm-up and sampling step: the cold-chain sample views must still alias the
    # live x / xₚ arrays, and the returned sample must equal the live cold positions. A swap
    # that forgets the views shows up as a detached parent or a stale (frozen) sample.
    function check_swap(sampler; n_chains = 6, n_hot_chains = 0, warmups = 4, posts = 4, seed = 1)
        rng = Xoshiro(seed)
        init = [randn(Xoshiro(i), 3) for i in 1:(n_chains + n_hot_chains)]
        _, st = AbstractMCMC.step(
            rng, model, sampler; n_chains, n_hot_chains,
            num_warmup = warmups, memory = false, silent = true, initial_position = init
        )
        seen = Vector{NTuple{3, Float64}}()
        for (stepfn, nsteps) in ((AbstractMCMC.step_warmup, warmups), (AbstractMCMC.step, posts))
            for _ in 1:nsteps
                smpl, st = stepfn(rng, model, sampler, st; num_warmup = warmups)
                @test parent(st.x_smpl_view) === st.x
                @test parent(st.xₚ_smpl_view) === st.xₚ
                @test smpl.x == st.x_smpl_view
                push!(seen, Tuple(smpl.x[1]))
            end
        end
        @test length(unique(seen)) > 1   # positions actually evolve across the run
    end

    @testset "simple DE update (no tempering / tempering)" begin
        check_swap(setup_de_update())
        check_swap(setup_de_update(); n_hot_chains = 4)
    end
    @testset "composite scheme (no tempering / tempering)" begin
        check_swap(setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update()))
        check_swap(setup_sampler_scheme(setup_subspace_sampling(), setup_snooker_update()); n_hot_chains = 4)
    end
end
