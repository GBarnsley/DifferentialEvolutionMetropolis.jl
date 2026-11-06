@testset "show methods (HMC)" begin
    de = setup_de_update(; γ = 1.0)
    hmc = setup_hmc_update(NUTS(0.8); n_dims = 2, metric_strategy = memory_metric())
    @test repr(hmc) == "HMC update(DiagEuclideanMetric, memory_metric(shrinkage = 0.0, every = 100))"
    @test startswith(plain(hmc), "HMC update\n  trajectory:")
    @test repr(cluster_pooled_metric()) == "cluster_pooled_metric(shrinkage = 0.0, every = 100, kmax = 10)"
    @test repr(per_cluster_metric(; kmax = 3)) == "per_cluster_metric(shrinkage = 0.0, every = 100, kmax = 3)"
    @test repr(setup_hmc_update(NUTS(0.8); n_dims = 2)) == "HMC update(DiagEuclideanMetric, stock adaptor)"

    model = AbstractMCMC.LogDensityModel(
        ADgradient(:ForwardDiff, CorrelatedGaussianModel(MvNormal(zeros(2), [1.0 0.0; 0.0 1.0])))
    )
    _, state = AbstractMCMC.step(
        backwards_compat_rng(1), model, setup_sampler_scheme(hmc, de); n_chains = 4
    )
    @test occursin("adaptive state: composite(HMC (not initialised, 0 HMC steps), static)", plain(state))
end
