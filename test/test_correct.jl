@testset "Test correctness with MCMCTesting.jl" begin
    #basic example from (https://arxiv.org/pdf/2001.06465) inspired by MCMCTesting.jl

    function sample_joint(rng::Random.AbstractRNG, model::IsotropicNormalModel)
        rand.(rng, Normal.(model.mean, 1))
    end

    function test_for_correctness!(
            rng, p_values, sum_of_rank_of_ranks, update, base_model, sample_kwargs, thinning, n, L,
            M_sampler, initial_positions, complete_chain, ordinal_ranks
        )
        d = LogDensityProblems.dimension(base_model)
        n_chains = sample_kwargs.n_chains
        for attempt in 1:n
            M = rand(rng, M_sampler)

            for i in eachindex(initial_positions)
                initial_positions[i] .= sample_joint(rng, base_model)
            end

            if M < L
                res = sample(
                    rng,
                    AbstractMCMC.LogDensityModel(base_model),
                    update,
                    L - M + 1;
                    initial_position = initial_positions,
                    thinning = thinning,
                    sample_kwargs...
                )
                complete_chain[M:L, :, 1:(end - 1)] .= res.samples
                complete_chain[M:L, :, end] .= res.ld
            else
                complete_chain[M, :, 1:(end - 1)] .= cat(initial_positions[1:n_chains]...; dims = 2)'
                complete_chain[M, :, end] .= [LogDensityProblems.logdensity(base_model, initial_positions[i]) for i in 1:n_chains]
            end

            if M > 1
                res = sample(
                    rng,
                    AbstractMCMC.LogDensityModel(base_model),
                    update,
                    M;
                    initial_position = initial_positions,
                    thinning = thinning,
                    sample_kwargs...
                )
                #append but ignore the value at M, its already on there
                if res.samples[1, :, :] != complete_chain[M, :, 1:(end - 1)]
                    error("Something went wrong with chaining the results together")
                end
                complete_chain[1:(M - 1), :, 1:(end - 1)] .= reverse(res.samples; dims = 1)[1:(end - 1), :, :]
                complete_chain[1:(M - 1), :, end] .= reverse(res.ld; dims = 1)[1:(end - 1), :]
            end

            for chain in 1:n_chains
                for parameter in 1:(d + 1)
                    ordinal_ranks[attempt, chain, parameter] = ordinalrank(complete_chain[:, chain, parameter])[M]
                end
            end

            for parameter in 1:(d + 1)
                sum_of_rank_of_ranks[1:n_chains, parameter] .+= ordinalrank(ordinal_ranks[attempt, :, parameter])
                sum_of_rank_of_ranks[end, parameter] += 1
            end
        end

        #across iterations, assuming they are all independent, we'll use sum_of_rank_of_ranks to test within iterations later
        p_values .= 0.0 #holds test statistic for now
        expected = n * (n_chains) / L
        n_v = zeros(Int, L, d + 1)
        values = zeros(Float64, L, d + 1)
        for v in 1:L
            for p in 1:(d + 1)
                n_v[v, p] = count(==(v), ordinal_ranks[1:n, :, p][:])
                values[v, p] = (n_v[v, p] - expected)^2 / expected
                p_values[p] += (n_v[v, p] - expected)^2 / expected
            end
        end

        #convert to p-values
        p_values .= ccdf.(Chisq(L - 1), p_values)

        return nothing
    end

    function friedman_statistic(sum_of_rank_of_ranks, attempts, n_chains)
        return ((12 / (attempts * n_chains * (n_chains + 1))) * sum(sum_of_rank_of_ranks .^ 2)) - 3 * attempts * (n_chains + 1)
    end

    function holm_procedure(p_values, α)
        sorted_p_values = sort(p_values)

        adjusted_min = minimum(sorted_p_values[sorted_p_values .> (α ./ (length(sorted_p_values) + 1 .- eachindex(sorted_p_values)))])

        any(sorted_p_values .< adjusted_min) #adjustment for multiple testing
    end

    function sequential_testing(
            rng, update, base_model, L, α_between, α_across, k, Δ, n, thinning, memory::Bool;
            d = LogDensityProblems.dimension(base_model),
            n_chains = memory ? 5 : d * 3,
            n_hot_chains = 0,
            N₀ = memory ? max(10 * d - n_chains, n_chains + n_hot_chains) : 0
        )
        #setup for model
        sample_kwargs = (
            memory = memory, N₀ = N₀, n_chains = n_chains, progress = false, adapt = false,
            chain_type = DifferentialEvolutionOutput, silent = true, n_hot_chains = n_hot_chains,
        )

        #setup for checks
        M_sampler = Distributions.sampler(DiscreteUniform(1, L))
        #rank based on values and ld so its less stuff to store
        n_positions = n_chains + N₀ + n_hot_chains
        initial_positions = [Vector{Float64}(undef, d) for i in 1:n_positions]
        complete_chain = Array{Float64, 3}(undef, L, n_chains, d + 1)
        ordinal_ranks = Array{Int, 3}(undef, n, n_chains, d + 1)
        sum_of_rank_of_ranks = zeros(Int, n_chains + 1, d + 1)

        β = α_across / k
        γ = β^(1 / k)

        p_values = Vector{Float64}(undef, d + 1)

        #sequential test for uniformity across iterations
        for i in 1:k
            test_for_correctness!(
                rng, p_values, sum_of_rank_of_ranks, update, base_model, sample_kwargs, thinning, n, L,
                M_sampler, initial_positions, complete_chain, ordinal_ranks
            )
            q = minimum(p_values) * length(p_values)
            if q ≤ β
                @warn "Failed rank-uniformity across all iterations test with p-value $(minimum(p_values))"
                return false
            elseif q > γ + β
                break
            else
                β = β / γ
                if i == 1
                    thinning *= Δ
                end
            end
        end

        #test for uniformity within iterations
        #since we can only rank 1:n_chains we can calculate the number of attempts as
        attempts = unique(sum_of_rank_of_ranks[end, :])[1]
        sum_of_rank_of_ranks = sum_of_rank_of_ranks[1:n_chains, :]

        statistics = [friedman_statistic(sum_of_rank_of_ranks[:, p], attempts, n_chains) for p in 1:(d + 1)]

        # good approx when n_chains > 4 and attempts > 15
        p_values_within = ccdf.(Chisq(n_chains - 1), statistics)

        if holm_procedure(p_values_within, α_between) #adjustment for multiple testing
            @warn "Failed rank-uniformity between DE-chains test with p-value $(minimum(p_values_within))"
            W = maximum(statistics) / (attempts * (n_chains - 1))
            if W > 0.01
                @warn "Large coefficient of concordance W = $W"
                return false
            else
                @warn "But small coefficient of concordance W = $W."
            end
        end

        return true
    end

    rng = backwards_compat_rng(1234)
    n = 1000
    Δ = 4
    L = 100
    base_model = IsotropicNormalModel(zeros(Int, 6))
    α_between = 10^-5
    α_across = 10^-5
    k = 7

    composite_update = setup_sampler_scheme(
        setup_de_update(n_dims = LogDensityProblems.dimension(base_model)),
        setup_snooker_update(),
        setup_subspace_sampling()
    )

    @testset "Without memory" begin
        thinning = 4
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α_between, α_across, k, Δ, n, thinning, false)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α_between, α_across, k, Δ, n, thinning, false)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α_between, α_across, k, Δ, n, thinning, false)
        @test sequential_testing(rng, composite_update, base_model, L, α_between, α_across, k, Δ, n, thinning, false)
    end
    @testset "With memory" begin
        thinning = 4
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α_between, α_across, k, Δ, n, 10, true)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α_between, α_across, k, Δ, n, thinning, true)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α_between, α_across, k, Δ, n, thinning, true)
        @test sequential_testing(rng, composite_update, base_model, L, α_between, α_across, k, Δ, n, thinning, true)
    end
    @testset "With pt" begin
        thinning = 2
        n_chains = 3
        n_hot_chains = LogDensityProblems.dimension(base_model) * 2 - n_chains
        @test sequential_testing(rng, setup_de_update(n_dims = LogDensityProblems.dimension(base_model)), base_model, L, α_between, α_across, k, Δ, n, 5, false; n_hot_chains = n_hot_chains, n_chains = n_chains)
        @test sequential_testing(rng, setup_snooker_update(), base_model, L, α_between, α_across, k, Δ, n, thinning, false; n_hot_chains = n_hot_chains, n_chains = n_chains)
        @test sequential_testing(rng, setup_subspace_sampling(), base_model, L, α_between, α_across, k, Δ, n, thinning, false; n_hot_chains = n_hot_chains, n_chains = n_chains)
        @test sequential_testing(rng, composite_update, base_model, L, α_between, α_across, k, Δ, n, thinning, false; n_hot_chains = n_hot_chains, n_chains = n_chains)
    end
end
