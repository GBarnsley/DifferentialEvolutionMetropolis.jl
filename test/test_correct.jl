@testset "Test correctness" begin
    rng = backwards_compat_rng(1234)

    standard_updates = [
        "DE" => d -> setup_de_update(; n_dims = d),
        "snooker" => d -> setup_snooker_update(),
        "subspace" => d -> setup_subspace_sampling(),
        "composite" => d -> setup_sampler_scheme(setup_de_update(; n_dims = d), setup_snooker_update(), setup_subspace_sampling()),
    ]
    pt_updates = [
        "subspace" => d -> setup_subspace_sampling(),
        "subspace + swap" => d -> setup_sampler_scheme(setup_subspace_sampling(), setup_swap_update()),
    ]
    settings = [
        ("Without memory", standard_updates, (; memory = false, n_hot_chains = 0, n_tries = 1)),
        ("With memory", standard_updates, (; memory = true, n_hot_chains = 0, n_tries = 1)),
        ("Multi-try", standard_updates, (; memory = false, n_hot_chains = 0, n_tries = 3)),
        ("With pt", pt_updates, (; memory = false, n_hot_chains = 3, n_tries = 1)),
    ]
    @test sum(((_, updates, _),) -> length(updates), settings) == N_CORRECTNESS_UPDATES.demc

    @testset "$setting" for (setting, updates, setup) in settings
        @testset "$name" for (name, update) in updates
            test_update_correctness(rng, update, CORRECTNESS_TEST_LEVEL; setup...)
        end
    end
end
