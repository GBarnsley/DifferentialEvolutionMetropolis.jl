using MCMCChains

@testset "templates (MCMCChains)" begin
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))

    @testset "chain_type outputs" begin
        # Test MCMCChains.Chains output
        result_chains = deMC(ld, 50; memory = false, chain_type = Chains)
        @test isa(result_chains, Chains)
        @test size(result_chains, 1) == 50  # iterations
        @test size(result_chains, 2) == 3   # parameters (2 + 1 ld)
        @test size(result_chains, 3) == 4   # chains (2 * n_dims)

        # Test with burn-in and Chains output
        result_chains_burnin = deMC(
            ld, 50; memory = false, chain_type = Chains,
            n_burnin = 25, save_burnt = false
        )
        @test isa(result_chains_burnin, Chains)
        # When save_burnt = false, burn-in is discarded, so only main iterations remain
        @test size(result_chains_burnin, 1) == 50  # Only the main 50 iterations

        # Test with save_burnt = true to include burn-in samples
        result_chains_with_burnin = deMC(
            ld, 30; memory = false, chain_type = Chains,
            n_burnin = 20, save_burnt = true
        )
        @test isa(result_chains_with_burnin, Chains)
        # When save_burnt = true, we get n_its + n_burnin total iterations
        @test size(result_chains_with_burnin, 1) == 50  # 30 + 20 iterations

        #test with meta chains
        n_meta_chains = 3
        n_samples = 10000
        n_burnin = 10000
        n_dims = LogDensityProblems.dimension(ld.logdensity)
        n_total_chains = n_meta_chains * (2 * n_dims)

        meta_chain = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples + n_burnin, n_meta_chains;
            chain_type = Chains, num_warmup = n_burnin, discard_initial = 0
        )
        @test size(meta_chain) == (n_samples + n_burnin, n_dims + 1, n_total_chains)

        meta_chain = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples, n_meta_chains;
            chain_type = Chains, num_warmup = n_burnin, save_final_state = true
        )
        @test size(meta_chain[1]) == (n_samples, n_dims + 1, n_total_chains)
        @test isa(meta_chain[2], Vector{<:DifferentialEvolutionMetropolis.DEMState})
    end
end
