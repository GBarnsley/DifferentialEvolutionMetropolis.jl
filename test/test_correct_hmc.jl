@testset "Test HMC correctness" begin
    rng = backwards_compat_rng(1234)
    α = 0.05

    """
        warmed_up(rng, scheme, d; num_warmup = 200)

    `scheme` fixed after `num_warmup` warmup steps of 3 memoryless chains on a `d`-dimensional
    standard normal, since HMC updates only resolve their step size and chains during warmup.
    """
    function warmed_up(rng, scheme, d; num_warmup = 200)
        target = AbstractMCMC.LogDensityModel(ADgradient(:ForwardDiff, IsotropicNormalModel(zeros(d))))
        _, state = AbstractMCMC.step(rng, target, scheme; n_chains = 3, memory = false, num_warmup, silent = true)
        for _ in 1:num_warmup
            _, state = AbstractMCMC.step_warmup(rng, target, scheme, state; num_warmup)
        end
        return DifferentialEvolutionMetropolis.fix_sampler(scheme, state.adaptive_state)
    end

    hmc_updates = [
        "NUTS" => d -> setup_sampler_scheme(setup_hmc_update(NUTS(0.8); n_dims = d)),
        "NUTS dense" => d -> setup_sampler_scheme(setup_hmc_update(NUTS(0.8; metric = :dense); n_dims = d)),
        "NUTS + DE" => d -> setup_sampler_scheme(setup_hmc_update(NUTS(0.8); n_dims = d), setup_de_update(; n_dims = d)),
    ]

    @testset "$name" for (name, update) in hmc_updates
        test_update_correctness(
            rng, d -> warmed_up(rng, update(d), d), α;
            memory = false, n_hot_chains = 0, n_tries = 1, gradient = true
        )
    end
end
