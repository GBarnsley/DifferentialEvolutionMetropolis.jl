@testset "Untransformed LogDensityProblems target" begin
    target = IsotropicNormalModel([-5.0, 5.0])
    wrapped = AbstractMCMC.LogDensityModel(target)
    sampler = setup_de_update()
    options = (;
        n_chains = 4, num_warmup = 10, silent = true, progress = false,
        chain_type = DEMOutput,
    )
    reference = sample(backwards_compat_rng(123), wrapped, sampler, 10; options...)
    result = sample(backwards_compat_rng(123), target, sampler, 10; options...)
    @test result.samples == reference.samples
    @test result.ld == reference.ld
    @test all(isfinite, result.ld)
end

@testset "Starting position dimensions" begin
    model = AbstractMCMC.LogDensityModel(IsotropicNormalModel(zeros(2)))
    options = (; n_chains = 4, silent = true, progress = false, chain_type = DEMOutput)
    @test_throws ErrorException sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [[1.0, 2.0], [1.0], [2.0, 3.0], [4.0, 5.0]],
        memory = false, options...
    )
    @test_throws ErrorException sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [zeros(2), ones(2), -ones(2), [1.0, -1.0], [1.0]],
        memory = true, options...
    )
    expanded = sample(
        backwards_compat_rng(42), model, setup_de_update(), 1;
        initial_position = [zeros(2)], memory = false, options...
    )
    @test size(expanded.samples) == (1, 4, 2)
    @test all(isfinite, expanded.ld)
end

@testset "Non-finite log densities" begin
    options = (; n_chains = 4, memory = false, silent = true, progress = false, chain_type = DEMOutput)
    sampler = setup_de_update()

    @testset "initial" begin
        model = AbstractMCMC.LogDensityModel(CountingNaNAtModel(0, 1, 0.0))
        @test_throws DomainError sample(
            backwards_compat_rng(42), model, sampler, 2; options...
        )
    end

    @testset "$stage" for (stage, nan_at, n_tries) in (
            ("proposal", 1, 1), ("forward trial", 1, 2), ("reference trial", 3, 2),
        )
        rng = backwards_compat_rng(42)
        model = AbstractMCMC.LogDensityModel(CountingNaNAtModel(0, 0, 0.0))
        _, state = AbstractMCMC.step(rng, model, sampler; n_tries = n_tries, options...)
        for chain_model in state.chain_models
            chain_model.count = 0
            chain_model.nan_at = nan_at
        end
        @test_throws DomainError AbstractMCMC.step(rng, model, sampler, state)
    end

    @testset "undefined ratio at $value" for value in (-Inf, Inf)
        model = AbstractMCMC.LogDensityModel(CountingNaNAtModel(0, 0, value))
        output = sample(backwards_compat_rng(42), model, sampler, 2; options...)
        @test output.samples[2, :, :] == output.samples[1, :, :]
        @test all(==(value), output.ld)
    end

    @testset "zero-density support" begin
        model = AbstractMCMC.LogDensityModel(IntervalModel(-Inf))
        output = sample(
            backwards_compat_rng(42), model, sampler, 100;
            initial_position = [[0.1], [0.3], [0.7], [0.9]], options...
        )
        @test all(x -> 0 <= x <= 1, output.samples)
        @test all(iszero, output.ld)
        _, state = AbstractMCMC.step(
            backwards_compat_rng(42), model, sampler;
            initial_position = [[-1.0], [0.3], [0.7], [0.9]], options...
        )
        # A proposal inside the support must be accepted from a zero-density position.
        state.xₚ[1] .= 0.5
        @test DifferentialEvolutionMetropolis.update_chain!(state.chain_models[1], state, 0.0, 1)
        @test iszero(state.ldₚ[1])
    end

    @testset "infinite-density region" begin
        model = AbstractMCMC.LogDensityModel(IntervalModel(Inf))
        output = sample(
            backwards_compat_rng(42), model, sampler, 100;
            initial_position = [[0.1], [0.3], [0.7], [0.9]], options...
        )
        @test all(iszero, output.ld[1, :])
        @test all(==(Inf), output.ld[end, :])
        for densities in eachcol(output.ld)
            first_infinite = findfirst(isinf, densities)
            @test all(==(Inf), densities[first_infinite:end])
        end
    end
end
