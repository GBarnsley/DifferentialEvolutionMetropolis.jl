@testset "complex likelihood - diagnostic check" begin
    #banana, likelhood, will not sample well, so should identify outliers
    function ld_banana(x)
        logpdf(Normal(0, 1), x[1]) + logpdf(Normal(x[1], 0.5), x[2])
    end
    n_pars = 2
    ld = TransformedLogDensities.TransformedLogDensity(as(Array, n_pars), ld_banana)
    n_chains = 4
    rng = backwards_compat_rng(1234)
    initial_state = randn(rng, n_chains, n_pars)
    sampler_scheme = setup_sampler_scheme(
        setup_subspace_sampling()
    )

    n_its = 1000
    n_burnin = 10000
    check_epochs = 10

    @testset "ld_check" begin
        diagnostic_checks = [
            ld_check(), #strict so we definitely get some outliers
        ]
        output = composite_sampler(
            ld, n_its, n_chains, false, initial_state, sampler_scheme;
            save_burnt = false, rng = rng, n_burnin = n_burnin, parallel = false,
            check_epochs = check_epochs
        )
        #simple checks more interested in if it runs or not
        @test size(output.samples) == (n_its, n_chains, n_pars)
        @test size(output.ld) == (n_its, n_chains)
        @test eltype(output.samples) == eltype(initial_state)
        @test eltype(output.ld) == eltype(initial_state)
    end

    @testset "ld_check" begin
        diagnostic_checks = [
            acceptance_check(0.25, 0.5), #strict so we definitely get some outliers
        ]
        output = composite_sampler(
            ld, n_its, n_chains, false, initial_state, sampler_scheme;
            save_burnt = false, rng = rng, n_burnin = n_burnin, parallel = false,
            check_epochs = check_epochs
        )
        #simple checks more interested in if it runs or not
        @test size(output.samples) == (n_its, n_chains, n_pars)
        @test size(output.ld) == (n_its, n_chains)
        @test eltype(output.samples) == eltype(initial_state)
        @test eltype(output.ld) == eltype(initial_state)
    end
end
