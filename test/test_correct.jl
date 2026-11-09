@testset "Test correctness" begin
    """
        correctness_setup(model, update; memory, n_hot_chains, n_tries)

    Sampler keyword arguments, a buffer of initial positions (chains, then hot chains, then
    memory) and the temperature each initial position is drawn at. Uses 3 chains, or the
    minimum a memoryless `update` requires if larger.
    """
    function correctness_setup(model, update; memory, n_hot_chains, n_tries)
        d = LogDensityProblems.dimension(model)
        n_chains = memory ? 3 : max(3, DifferentialEvolutionMetropolis.chains_required(update))
        ladder = vcat(ones(n_chains), [1 + (2 * sqrt(d) - 1) * h / n_hot_chains for h in 1:n_hot_chains])
        N₀ = memory ? 3 * n_chains * d : 0
        kwargs = (;
            memory, N₀, n_chains, n_hot_chains, n_tries, temperature_ladder = [ladder],
            progress = false, adapt = false, chain_type = DifferentialEvolutionOutput, silent = true,
        )
        initial_positions = [Vector{Float64}(undef, d) for _ in 1:(n_chains + n_hot_chains + N₀)]
        temperatures = vcat(ladder, ones(N₀))
        return kwargs, initial_positions, temperatures
    end

    draw(rng, model::IsotropicNormalModel, temperature) =
        model.mean .+ sqrt(temperature) .* randn(rng, length(model.mean))
    draw(rng, ::UnitIntervalModel, temperature) = rand(rng, 1)

    function draw_from_target!(rng, initial_positions, model, temperatures)
        for i in eachindex(initial_positions)
            initial_positions[i] .= draw(rng, model, temperatures[i])
        end
        return initial_positions
    end

    run_sampler(rng, model, update, n_samples, initial_positions, kwargs) =
        suppress_warnings(NO_SWAP_WARNING) do
        sample(
            rng, AbstractMCMC.LogDensityModel(model), update, n_samples;
            initial_position = initial_positions, kwargs...
        )
    end

    """
        t_test(estimates, expected)

    Two-sided one-sample t-test p-value that the mean of independent `estimates` equals `expected`.
    """
    function t_test(estimates, expected)
        n = length(estimates)
        t = (mean(estimates) - expected) / (std(estimates) / sqrt(n))
        return 2 * ccdf(TDist(n - 1), abs(t))
    end

    # Rank test from https://arxiv.org/pdf/2001.06465, inspired by MCMCTesting.jl.
    function test_for_correctness!(
            rng, p_values, sum_of_rank_of_ranks, update, model, kwargs, n, L,
            M_sampler, initial_positions, temperatures, complete_chain, ordinal_ranks
        )
        d = LogDensityProblems.dimension(model)
        n_chains = kwargs.n_chains
        for attempt in 1:n
            M = rand(rng, M_sampler)
            draw_from_target!(rng, initial_positions, model, temperatures)

            if M < L
                res = run_sampler(rng, model, update, L - M + 1, initial_positions, kwargs)
                complete_chain[M:L, :, 1:(end - 1)] .= res.samples
                complete_chain[M:L, :, end] .= res.ld
            else
                complete_chain[M, :, 1:(end - 1)] .= cat(initial_positions[1:n_chains]...; dims = 2)'
                complete_chain[M, :, end] .= [LogDensityProblems.logdensity(model, initial_positions[i]) for i in 1:n_chains]
            end

            if M > 1
                res = run_sampler(rng, model, update, M, initial_positions, kwargs)
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

        expected = n * n_chains / L
        for p in 1:(d + 1)
            statistic = sum(v -> (count(==(v), view(ordinal_ranks, 1:n, :, p)) - expected)^2 / expected, 1:L)
            p_values[p] = ccdf(Chisq(L - 1), statistic)
        end
        return nothing
    end

    function friedman_statistic(sum_of_rank_of_ranks, attempts, n_chains)
        return ((12 / (attempts * n_chains * (n_chains + 1))) * sum(sum_of_rank_of_ranks .^ 2)) - 3 * attempts * (n_chains + 1)
    end

    """
        sequential_testing(rng, update, model, α; n = 1000, L = 100, k = 7, kwargs...)

    Sequential rank-uniformity test across iterations, then a Friedman test for rank uniformity
    between the chains. Returns `true` if neither test rejects at level `α`.
    """
    function sequential_testing(rng, update, model, α; n = 1000, L = 100, k = 7, setup...)
        d = LogDensityProblems.dimension(model)
        kwargs, initial_positions, temperatures = correctness_setup(model, update; setup...)
        n_chains = kwargs.n_chains

        M_sampler = Distributions.sampler(DiscreteUniform(1, L))
        complete_chain = Array{Float64, 3}(undef, L, n_chains, d + 1)
        ordinal_ranks = Array{Int, 3}(undef, n, n_chains, d + 1)
        sum_of_rank_of_ranks = zeros(Int, n_chains + 1, d + 1)
        p_values = Vector{Float64}(undef, d + 1)

        β = α / k
        γ = β^(1 / k)
        for _ in 1:k
            test_for_correctness!(
                rng, p_values, sum_of_rank_of_ranks, update, model, kwargs, n, L,
                M_sampler, initial_positions, temperatures, complete_chain, ordinal_ranks
            )
            q = minimum(p_values) * length(p_values)
            if q ≤ β
                @warn "Failed rank-uniformity across all iterations test with p-value $(minimum(p_values))"
                return false
            elseif q > γ + β
                break
            else
                β = β / γ
            end
        end

        attempts = sum_of_rank_of_ranks[end, 1]
        statistics = [friedman_statistic(sum_of_rank_of_ranks[1:n_chains, p], attempts, n_chains) for p in 1:(d + 1)]
        p_values_within = ccdf.(Chisq(n_chains - 1), statistics)

        if minimum(p_values_within) * length(p_values_within) ≤ α
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

    """
        normal_moments_check(rng, update; n = 1000, n_samples = 1000, setup...)

    p-values that the first four moments of a long run on a standard normal match `0, 1, 0, 3`.
    """
    function normal_moments_check(rng, update; n = 1000, n_samples = 1000, setup...)
        model = IsotropicNormalModel([0.0])
        kwargs, initial_positions, temperatures = correctness_setup(model, update; setup...)
        estimates = zeros(n, 4)
        for r in 1:n
            draw_from_target!(rng, initial_positions, model, temperatures)
            samples = run_sampler(rng, model, update, n_samples, initial_positions, kwargs).samples
            for k in 1:4
                estimates[r, k] = mean(x -> x^k, samples)
            end
        end
        return [t_test(estimates[:, k], μ) for (k, μ) in enumerate((0.0, 1.0, 0.0, 3.0))]
    end

    """
        one_step_covariance_testing(rng, update; n = 10_000, setup...)

    p-values that, after one step on a standard normal, the average pairwise covariance between
    chains is 0 and the average squared cross moment is 1.
    """
    function one_step_covariance_testing(rng, update; n = 10_000, setup...)
        model = IsotropicNormalModel([0.0])
        kwargs, initial_positions, temperatures = correctness_setup(model, update; setup...)
        n_chains = kwargs.n_chains
        pairs = [(i, j) for i in 1:n_chains for j in (i + 1):n_chains]
        covariance = zeros(n)
        cross_moment = zeros(n)
        for r in 1:n
            draw_from_target!(rng, initial_positions, model, temperatures)
            x = run_sampler(rng, model, update, 2, initial_positions, kwargs).samples[end, :, 1]
            covariance[r] = mean(((i, j),) -> x[i] * x[j], pairs)
            cross_moment[r] = mean(((i, j),) -> (x[i] * x[j])^2, pairs)
        end
        return t_test(covariance, 0.0), t_test(cross_moment, 1.0)
    end

    """
        short_run_marginal_stationarity(rng, update; n = 10_000, n_steps = 4, setup...)

    p-value that, after `n_steps` steps on Uniform(0, 1), the mean squared deviation of the chains
    from 0.5 is 1/12.
    """
    function short_run_marginal_stationarity(rng, update; n = 10_000, n_steps = 4, setup...)
        model = UnitIntervalModel()
        kwargs, initial_positions, temperatures = correctness_setup(model, update; setup...)
        deviation = zeros(n)
        for r in 1:n
            draw_from_target!(rng, initial_positions, model, temperatures)
            x = run_sampler(rng, model, update, n_steps + 1, initial_positions, kwargs).samples[end, :, 1]
            deviation[r] = mean(y -> (y - 0.5)^2, x)
        end
        return t_test(deviation, 1 / 12)
    end

    rng = backwards_compat_rng(1234)
    α = 0.05
    sequential_model = IsotropicNormalModel(zeros(3))

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
            @test sequential_testing(rng, update(3), sequential_model, α; n_tries, setup...)
            covariance_p, cross_moment_p = one_step_covariance_testing(rng, update(1); n_tries, setup...)
            uniform_p = short_run_marginal_stationarity(rng, update(1); n_tries, setup...)
            moment_ps = normal_moments_check(rng, update(1); n_tries, setup...)
            # Bonferroni correction over this update's non-sequential p-values.
            α_corrected = α / (3 + length(moment_ps))
            @test covariance_p > α_corrected
            @test cross_moment_p > α_corrected
            @test uniform_p > α_corrected
            @test minimum(moment_ps) > α_corrected
        end
    end
end
