@testset "test rng states" begin
    @testset "sequential" begin
        model = IsotropicNormalModel([-5.0, 5.0])

        # DE + Snooker composite
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_de_update(n_dims = LogDensityProblems.dimension(model)),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0),
            setup_subspace_sampling(cr = DiscreteNonParametric([0.5, 1.0], [0.5, 0.5]))
        )

        de_sampler_old = deepcopy(de_sampler)

        #should be equal after deepcopy
        @test de_sampler.update_weights == de_sampler_old.update_weights
        for i in eachindex(de_sampler.updates)
            if isa(de_sampler.updates[i], DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionSubspaceSampler)
                @test de_sampler.updates[i].cr_spl == de_sampler_old.updates[i].cr_spl
                @test de_sampler.updates[i].n_cr == de_sampler_old.updates[i].n_cr
                @test de_sampler.updates[i].δ_spl == de_sampler_old.updates[i].δ_spl
                @test de_sampler.updates[i].ϵ_spl == de_sampler_old.updates[i].ϵ_spl
                @test de_sampler.updates[i].e_spl == de_sampler_old.updates[i].e_spl
            else
                @test de_sampler.updates[i] == de_sampler_old.updates[i]
            end
        end

        #should give the same result
        n_its = 1000
        n_warmup = 1000
        seed = 112
        output1 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            n_its;
            num_warmup = n_warmup,
            parallel = false,
            progress = false
        )

        output2 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            n_its;
            num_warmup = n_warmup,
            parallel = true,
            progress = false
        )

        #should still be equal after deepcopy
        @test de_sampler.update_weights == de_sampler_old.update_weights
        for i in eachindex(de_sampler.updates)
            if isa(de_sampler.updates[i], DifferentialEvolutionMetropolis.AbstractDifferentialEvolutionSubspaceSampler)
                @test de_sampler.updates[i].cr_spl == de_sampler_old.updates[i].cr_spl
                @test de_sampler.updates[i].n_cr == de_sampler_old.updates[i].n_cr
                @test de_sampler.updates[i].δ_spl == de_sampler_old.updates[i].δ_spl
                @test de_sampler.updates[i].ϵ_spl == de_sampler_old.updates[i].ϵ_spl
                @test de_sampler.updates[i].e_spl == de_sampler_old.updates[i].e_spl
            else
                @test de_sampler.updates[i] == de_sampler_old.updates[i]
            end
        end

        equality_x = [isequal(output1[i].x, output2[i].x) for i in 1:length(output1)]
        equality_ld = [isequal(output1[i].ld, output2[i].ld) for i in 1:length(output1)]
        @test all(equality_x)
        @test all(equality_ld)
    end
    @testset "multithreaded" begin
        model = IsotropicNormalModel([-5.0, 5.0])

        # DE + Snooker composite
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_de_update(n_dims = LogDensityProblems.dimension(model)),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0),
            setup_subspace_sampling(cr = DiscreteNonParametric([0.5, 1.0], [0.5, 0.5]))
        )

        #should give the same result
        n_its = 1000
        n_chains = 3
        n_warmup = 1000
        seed = 112
        output1 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            MCMCThreads(),
            n_its,
            n_chains;
            num_warmup = n_warmup,
            parallel = false,
            progress = false
        )

        output2 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            MCMCThreads(),
            n_its,
            n_chains;
            num_warmup = n_warmup,
            parallel = true,
            progress = false
        )

        equality_x = vcat(
            [
                [isequal(output1[j][i].x, output2[j][i].x) for i in 1:length(output1)]
                    for j in 1:n_chains
            ]...
        )
        equality_ld = vcat(
            [
                [isequal(output1[j][i].ld, output2[j][i].ld) for i in 1:length(output1)]
                    for j in 1:n_chains
            ]...
        )
        @test all(equality_x)
        @test all(equality_ld)
    end
    @testset "multicore" begin
        model = IsotropicNormalModel([-5.0, 5.0])

        # DE + Snooker composite
        de_sampler = setup_sampler_scheme(
            setup_de_update(),
            setup_de_update(n_dims = LogDensityProblems.dimension(model)),
            setup_snooker_update(deterministic_γ = false),
            setup_snooker_update(deterministic_γ = true),
            setup_subspace_sampling(),
            setup_subspace_sampling(γ = 1.0),
            setup_subspace_sampling(cr = DiscreteNonParametric([0.5, 1.0], [0.5, 0.5]))
        )

        #should give the same result
        n_its = 1000
        n_chains = 3
        n_warmup = 1000
        seed = 112
        output1 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            MCMCDistributed(),
            n_its,
            n_chains;
            num_warmup = n_warmup,
            parallel = false,
            progress = false
        )

        output2 = sample(
            backwards_compat_rng(seed),
            AbstractMCMC.LogDensityModel(model),
            de_sampler,
            MCMCDistributed(),
            n_its,
            n_chains;
            num_warmup = n_warmup,
            parallel = true,
            progress = false
        )

        equality_x = vcat(
            [
                [isequal(output1[j][i].x, output2[j][i].x) for i in 1:length(output1)]
                    for j in 1:n_chains
            ]...
        )
        equality_ld = vcat(
            [
                [isequal(output1[j][i].ld, output2[j][i].ld) for i in 1:length(output1)]
                    for j in 1:n_chains
            ]...
        )
        @test all(equality_x)
        @test all(equality_ld)
    end
    @testset "Stress check fast sample" begin
        pop_size = 5
        rng = backwards_compat_rng(1234)
        x = [randn(rng, 5) for _ in 1:(pop_size * 2)]
        n_chains = pop_size
        max_length = pop_size
        N_tests = 1000

        @testset "no current_chain" begin
            indices = Vector{Int}(undef, n_chains)
            ordered_indices = Array{Int}(undef, n_chains - 1)

            res = [
                DifferentialEvolutionMetropolis.fast_sample_chains!(
                        rng,
                        x,
                        max_length,
                        n_chains,
                        indices,
                        ordered_indices
                    )
                    for _ in 1:N_tests
            ]

            @test all(length(unique(r)) == n_chains for r in res)
            @test all(all(findlast(r[i:i] .== x) ≤ max_length for i in 1:n_chains) for r in res)
        end
        @testset "current_chain" begin
            current_chain = 3
            n_chains_2 = n_chains - 1
            indices = Vector{Int}(undef, n_chains_2)
            ordered_indices = Array{Int}(undef, n_chains_2)

            res = [
                DifferentialEvolutionMetropolis.fast_sample_chains!(
                        rng,
                        x,
                        max_length,
                        n_chains_2,
                        indices,
                        ordered_indices,
                        current_chain
                    )
                    for _ in 1:N_tests
            ]

            @test all(length(unique(r)) == n_chains_2 for r in res)
            @test all(all(findlast(r[i:i] .== x) ≤ max_length for i in 1:n_chains_2) for r in res)
            @test all(all(r[i] != current_chain for i in 1:n_chains_2) for r in res)
        end

        max_length = round(Int, 1.5 * pop_size)
        @testset "no current_chain no prealloc" begin
            indices = Vector{Int}(undef, 2)
            ordered_indices = Array{Int}(undef, 1)

            disable_logging(Logging.Warn)
            res = [
                DifferentialEvolutionMetropolis.fast_sample_chains!(
                        rng,
                        x,
                        max_length,
                        n_chains,
                        indices,
                        ordered_indices
                    )
                    for _ in 1:N_tests
            ]
            disable_logging(Logging.Info)
            @test_logs (:warn, "Picking $n_chains chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains`.")  DifferentialEvolutionMetropolis.fast_sample_chains!(
                rng,
                x,
                max_length,
                n_chains,
                indices,
                ordered_indices
            )
            @test all(length(unique(r)) == n_chains for r in res)
            @test all(all(findlast(r[i:i] .== x) ≤ max_length for i in 1:n_chains) for r in res)
        end
        @testset "current_chain" begin
            current_chain = 3
            n_chains_2 = n_chains - 1
            indices = Vector{Int}(undef, 2)
            ordered_indices = Array{Int}(undef, 1)

            disable_logging(Logging.Warn)
            res = [
                DifferentialEvolutionMetropolis.fast_sample_chains!(
                        rng,
                        x,
                        max_length,
                        n_chains_2,
                        indices,
                        ordered_indices,
                        current_chain
                    )
                    for _ in 1:N_tests
            ]
            disable_logging(Logging.Info)


            @test_logs (:warn, "Picking $n_chains_2 chains but only $(length(indices)) preallocated, consider setting `n_preallocated_indices = $n_chains_2`.")  DifferentialEvolutionMetropolis.fast_sample_chains!(
                rng,
                x,
                max_length,
                n_chains_2,
                indices,
                ordered_indices,
                current_chain
            )
            @test all(length(unique(r)) == n_chains_2 for r in res)
            @test all(all(findlast(r[i:i] .== x) ≤ max_length for i in 1:n_chains_2) for r in res)
            @test all(all(r[i] != current_chain for i in 1:n_chains_2) for r in res)
        end
    end
end
