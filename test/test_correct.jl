@testset "Test correctness" begin
    rng = backwards_compat_rng(1234)
    α = 0.05

    standard_updates = [
        "DE" => (d -> setup_de_update(; n_dims = d), 1),
        "snooker" => (d -> setup_snooker_update(), 1),
        "subspace" => (d -> setup_subspace_sampling(), 1),
        "composite" => (d -> setup_sampler_scheme(setup_de_update(; n_dims = d), setup_snooker_update(), setup_subspace_sampling()), 1),
        "DE multi-try" => (d -> setup_de_update(; n_dims = d), 3),
        "subspace multi-try" => (d -> setup_subspace_sampling(), 3),
    ]
    pt_updates = [
        "subspace" => (d -> setup_subspace_sampling(), 1),
        "subspace + swap" => (d -> setup_sampler_scheme(setup_subspace_sampling(), setup_swap_update()), 1),
    ]
    settings = [
        ("Without memory", standard_updates, (; memory = false, n_hot_chains = 0)),
        ("With memory", standard_updates, (; memory = true, n_hot_chains = 0)),
        ("With pt", pt_updates, (; memory = false, n_hot_chains = 3)),
    ]

    @testset "$setting" for (setting, updates, setup) in settings
        @testset "$name" for (name, (update, n_tries)) in updates
            test_update_correctness(rng, update, α; n_tries, setup...)
        end
    end
end
