# Deliberately invalid DE: every chain draws donors from the ensemble at the start of
# the step, rather than seeing the accepted updates of earlier chains. This is the
# simultaneous (parallel) memoryless kernel, evaluated deterministically even with
# one thread. Keep it in the tests; it does not preserve the joint target.
struct ParallelMemorylessDE <: DifferentialEvolutionMetropolis.AbstractDEMSampler{
        DifferentialEvolutionMetropolis.DEMSamplerMultiThread,
        DifferentialEvolutionMetropolis.DEMSamplerSingleTry,
    }
    γ::Float64
end

function DifferentialEvolutionMetropolis.proposal!(
        xₚ::V, x::V, state::DifferentialEvolutionMetropolis.DEMState,
        update::ParallelMemorylessDE, i::Int
    ) where {V <: AbstractVector}
    rng = state.rngs[i]
    indices = state.memory.indices_INTERNAL[i]
    DifferentialEvolutionMetropolis.fast_sample_chains!(
        rng, state.x, length(state.x), 2, indices,
        state.memory.ordered_indices_INTERNAL[i], i
    )
    # The bug: ignore the accepted positions in state.xₚ when selecting donors.
    x₁, x₂ = state.x[indices[1]], state.x[indices[2]]
    if x₁ == x₂
        xₚ .= x
        return (; offset = -Inf)
    end
    for p in eachindex(x)
        xₚ[p] = x[p] + update.γ * (x₁[p] - x₂[p]) + 1.0e-12 * randn(rng)
    end
    return (; offset = 0.0)
end

@testset "Correctness checks reject parallel memoryless DE" begin
    rng = backwards_compat_rng(1234)
    update(d) = ParallelMemorylessDE(2.38 / sqrt(2 * d))
    α = CORRECTNESS_TEST_LEVEL
    results = @test_logs (:warn, r"Failed rank-uniformity") match_mode = :any update_correctness_results(
        rng, update, α; memory = false, n_hot_chains = 0, n_tries = 1
    )

    # Run the same checks, sample sizes and corrected alpha as the valid samplers.
    # Covariance and odd moments need not detect this bug; require the strong signals.
    @test !results.sequential
    @test results.uniform_p ≤ α
    @test results.moment_ps[2] ≤ α
    @test results.moment_ps[4] ≤ α
end
