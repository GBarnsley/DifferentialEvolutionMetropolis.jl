@testset "Test HMC correctness" begin
    rng = backwards_compat_rng(1234)
    α = 0.05

    hmc_updates = [
        "NUTS" => d -> setup_sampler_scheme(setup_hmc_update(NUTS(0.8; integrator = Leapfrog(0.5)); n_dims = d)),
        "NUTS dense" => d -> setup_sampler_scheme(
            setup_hmc_update(NUTS(0.8; integrator = Leapfrog(0.5), metric = :dense); n_dims = d)
        ),
        "HMC" => d -> setup_sampler_scheme(setup_hmc_update(HMC(0.5, 5); n_dims = d)),
        "HMCDA" => d -> setup_sampler_scheme(setup_hmc_update(HMCDA(0.8, 2.0; integrator = Leapfrog(0.5)); n_dims = d)),
        "NUTS + DE" => d -> setup_sampler_scheme(
            setup_hmc_update(NUTS(0.8; integrator = Leapfrog(0.5)); n_dims = d), setup_de_update(; n_dims = d)
        ),
    ]

    @testset "$name" for (name, update) in hmc_updates
        test_update_correctness(
            rng, update, α;
            memory = false, n_hot_chains = 0, n_tries = 1, gradient = true
        )
    end
end
