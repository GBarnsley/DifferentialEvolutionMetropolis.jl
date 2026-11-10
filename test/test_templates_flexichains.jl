using FlexiChains

@testset "templates (FlexiChains)" begin
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))

    @testset "chain_type outputs" begin
        # Test FlexiChains output
        result_chains = deMC(ld, 50; memory = false, chain_type = VNChain)
        @test isa(result_chains, FlexiChains.FlexiChain)
        @test size(result_chains, 1) == 50  # iterations
        @test size(result_chains, 2) == 4   # chains (2 * n_dims)
        @test size(FlexiChains.parameters(result_chains), 1) == 2   # parameters
        @test size(FlexiChains.extras(result_chains), 1) == 1   # ld

        #test with meta chains
        n_meta_chains = 3
        n_samples = 10000
        n_burnin = 10000
        n_dims = LogDensityProblems.dimension(ld.logdensity)
        n_total_chains = n_meta_chains * (2 * n_dims)

        meta_chain = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples + n_burnin, n_meta_chains;
            chain_type = VNChain, num_warmup = n_burnin, discard_initial = 0
        )
        @test size(meta_chain) == (n_samples + n_burnin, n_total_chains)
        @test size(FlexiChains.parameters(meta_chain), 1) == n_dims   # parameters
        @test size(FlexiChains.extras(meta_chain), 1) == 1   # ld

        meta_chain = sample(
            ld, setup_subspace_sampling(), MCMCSerial(), n_samples, n_meta_chains;
            chain_type = VNChain, num_warmup = n_burnin, save_final_state = true
        )

        @test size(meta_chain[1]) == (n_samples, n_total_chains)
        @test size(FlexiChains.parameters(meta_chain[1]), 1) == n_dims   # parameters
        @test size(FlexiChains.extras(meta_chain[1]), 1) == 1   # ld
        @test isa(meta_chain[2], Vector{<:DifferentialEvolutionMetropolis.DEMState})
    end
end
