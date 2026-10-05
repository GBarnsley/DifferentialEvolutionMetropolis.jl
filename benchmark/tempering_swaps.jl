# Run from the repository root: julia --project=. benchmark/tempering_swaps.jl
# Investigation only: no production sampler API or data files are changed.
using AbstractMCMC, DifferentialEvolutionMetropolis, LogDensityProblems, Random
using Statistics, Printf, Test

struct SwapMixture
    dims::Int
    separation::Float64
end
LogDensityProblems.dimension(m::SwapMixture) = m.dims
LogDensityProblems.capabilities(::SwapMixture) = LogDensityProblems.LogDensityOrder{0}()
function LogDensityProblems.logdensity(m::SwapMixture, x)
    a = -((x[1] - m.separation)^2 + sum(abs2, @view x[2:end])) / 2
    b = -((x[1] + m.separation)^2 + sum(abs2, @view x[2:end])) / 2
    c = max(a, b)
    return c + log(exp(a - c) + exp(b - c)) - log(2)
end

# Apply replica exchange to the *live* state, after the DE step has swapped its
# proposal buffers. Keep temperatures, per-rung RNGs, models and adaptation fixed.
function exchange!(rng, state, pairs)
    accepted = 0
    for (i, j) in pairs
        ti = DifferentialEvolutionMetropolis.get_temperature(state.temperature_ladder, i)
        tj = DifferentialEvolutionMetropolis.get_temperature(state.temperature_ladder, j)
        logratio = (inv(ti) - inv(tj)) * (state.ld[j] - state.ld[i])
        if log(rand(rng)) < min(0, logratio)
            state.x[i], state.x[j] = state.x[j], state.x[i]
            state.ld[i], state.ld[j] = state.ld[j], state.ld[i]
            accepted += 1
        end
    end
    return accepted
end

function check_exchange()
    model = AbstractMCMC.LogDensityModel(SwapMixture(2, 0.0))
    sampler = setup_de_update(n_dims = 2)
    _, state = AbstractMCMC.step(
        Xoshiro(1), model, sampler; n_chains = 3, n_hot_chains = 1,
        temperature_ladder = [[1.0, 1.0, 1.0, 2.0]], memory = false,
        adapt = false, silent = true, initial_position = [[10.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    )
    @test exchange!(Xoshiro(2), state, [(1, 4)]) == 1 # positive ratio
    @test state.x[1] == [0.0, 0.0]
    @test state.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), state.x)
    @test parent(state.x_smpl_view) === state.x
    @test state.x_smpl_view[1] === state.x[1]
    @test exchange!(Xoshiro(2), state, [(1, 4)]) == 0 # ratio -25
    @test exchange!(Xoshiro(2), state, [(1, 2)]) == 1 # equal temperature
    # Normal DE stepping remains valid after exchange; do not use the pre-exchange sample.
    _, state = AbstractMCMC.step(Xoshiro(3), model, sampler, state; replica_exchange = false)
    @test state.ld == LogDensityProblems.logdensity.(Ref(model.logdensity), state.x)
    return @test parent(state.x_smpl_view) === state.x
end

function trial(dims, separation, hot, seed, swaps; warmup = 2000, draws = 6000)
    cold = max(4, 2dims)
    model = AbstractMCMC.LogDensityModel(SwapMixture(dims, separation))
    sampler = setup_de_update(n_dims = dims)
    init_rng = Xoshiro(seed)
    initial = [randn(init_rng, dims) for _ in 1:(cold + hot)]
    # Start all replicas in one mode, testing discovery rather than just shuffling
    # already-populated modes. The Gaussian control has separation zero.
    for x in initial
        x[1] -= separation
    end
    ladder = [[ones(cold)..., exp.(range(log(1.5), log(16.0); length = hot))...]]
    _, state = AbstractMCMC.step(
        Xoshiro(seed), model, sampler; n_chains = cold, n_hot_chains = hot,
        temperature_ladder = ladder, memory = false, adapt = false,
        silent = true, initial_position = initial
    )
    # Separate streams keep local-update RNG seeds matched between treatments.
    rng, swap_rng = Xoshiro(seed + 10000), Xoshiro(seed + 20000)
    occupancy = zeros(draws)
    second_moment = zeros(draws)
    previous = [x[1] > 0 for x in state.x[1:cold]]
    transitions = accepts = attempts = boundary_accepts = boundary_attempts = 0
    # One randomly selected cold chain participates at the cold/hot boundary;
    # remaining hot neighbours use alternating non-overlapping matchings.
    elapsed = @elapsed for t in 1:(warmup + draws)
        _, state = AbstractMCMC.step(rng, model, sampler, state; replica_exchange = false)
        if swaps
            c = rand(swap_rng, 1:cold)
            order = [c; collect((cold + 1):(cold + hot))]
            start = isodd(t) ? 1 : 2
            pairs = [(order[k], order[k + 1]) for k in start:2:(length(order) - 1)]
            n = 0
            for pair in pairs
                accepted = exchange!(swap_rng, state, (pair,))
                n += accepted
                if t > warmup && pair[1] <= cold
                    boundary_accepts += accepted
                    boundary_attempts += 1
                end
            end
            if t > warmup
                accepts += n
                attempts += length(pairs)
            end
        end
        if t > warmup
            signs = [x[1] > 0 for x in state.x[1:cold]]
            transitions += sum(signs .!= previous)
            occupancy[t - warmup] = mean(signs)
            second_moment[t - warmup] = mean(x[1]^2 for x in state.x[1:cold])
        end
        previous = [x[1] > 0 for x in state.x[1:cold]]
    end
    return (;
        occupancy = mean(occupancy), second_moment = mean(second_moment),
        transitions = transitions / (cold * draws),
        acceptance = attempts == 0 ? NaN : accepts / attempts,
        boundary_acceptance = boundary_attempts == 0 ? NaN : boundary_accepts / boundary_attempts, elapsed,
    )
end

function main()
    @testset "Replica exchange checks" begin
        check_exchange()
    end
    trial(2, 0.0, 4, 1, false; warmup = 2, draws = 2) # compile both paths
    trial(2, 0.0, 4, 1, true; warmup = 2, draws = 2)
    println("dims separation hot swaps occupancy_rmse second_moment_rmse transitions_per_chain_step swap_accept boundary_accept seconds")
    for (dims, separation, hot) in ((2, 0.0, 4), (2, 6.0, 4), (10, 6.0, 4), (10, 6.0, 12))
        for swaps in (false, true)
            runs = [trial(dims, separation, hot, seed, swaps) for seed in 1:32]
            occupancy_rmse = sqrt(mean((r.occupancy - 0.5)^2 for r in runs))
            moment_rmse = sqrt(mean((r.second_moment - (1 + separation^2))^2 for r in runs))
            @printf(
                "%d %.1f %d %s %.5f %.5f %.5f %.5f %.5f %.3f\n", dims, separation, hot, swaps,
                occupancy_rmse, moment_rmse, mean(r.transitions for r in runs),
                mean(r.acceptance for r in runs), mean(r.boundary_acceptance for r in runs),
                mean(r.elapsed for r in runs)
            )
        end
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
