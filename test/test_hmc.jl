# Tests for the HMC/NUTS harness (ext/AdvancedHMCExt.jl): an AdvancedHMC sampler run as
# one update inside a composite scheme, sharing a single adapted kernel across all DE
# chains.

using AdvancedHMC
using ForwardDiff
using LogDensityProblemsAD
using LinearAlgebra: diag

# A correlated Gaussian with known mean/covariance — a unimodal target where the harness
# must recover the posterior and the metric must approach the marginal variances.
struct CorrelatedGaussianModel{D}
    dist::D
end
function LogDensityProblems.dimension(model::CorrelatedGaussianModel)
    return length(model.dist)
end
function LogDensityProblems.logdensity(model::CorrelatedGaussianModel, x::AbstractVector{<:Real})
    return logpdf(model.dist, x)
end
LogDensityProblems.capabilities(::Type{<:CorrelatedGaussianModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.capabilities(::CorrelatedGaussianModel) = LogDensityProblems.LogDensityOrder{0}()

# Pull the (single, shared) HMC adaptive state out of a composite state.
hmc_adaptive_state(state) = state.adaptive_state.adaptive_states[1]

@testset "HMC harness (AdvancedHMCExt)" begin

    μ = [2.0, -1.0, 0.5]
    Σ = [1.0 0.6 0.0; 0.6 1.5 -0.3; 0.0 -0.3 0.8]
    raw_model = CorrelatedGaussianModel(MvNormal(μ, Σ))
    grad_model = ADgradient(:ForwardDiff, raw_model)
    model = AbstractMCMC.LogDensityModel(grad_model)

    @testset "gradient requirement fails loudly on order-0 target" begin
        order0 = AbstractMCMC.LogDensityModel(raw_model)
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8)))
        @test_throws ErrorException sample(
            backwards_compat_rng(1), order0, scheme, 50;
            n_chains = 6, num_warmup = 20, memory = false, progress = false
        )
    end

    @testset "internal AdvancedHMC entry points still exist" begin
        # The harness depends on a wider surface than the AbstractMCMC layer; if any
        # of these disappear upstream, fail here rather than deep in a trajectory.
        for f in (
                :make_metric, :make_step_size, :make_integrator, :make_kernel,
                :make_adaptor, :Hamiltonian, :phasepoint, :transition, :update,
            )
            @test isdefined(AdvancedHMC, f)
        end
        for f in (
                :is_in_window, :is_window_end, :reset!, :adapt!,
                :initialize!, :finalize!, :getM⁻¹,
            )
            @test isdefined(AdvancedHMC.Adaptation, f)
        end
    end

    # Run with memory enabled (the package's primary mode: HMC writes into the shared
    # historical pool even though its own kernel does not read it) and with memory
    # disabled, to cover both update_state paths.
    @testset "recovers a unimodal Gaussian (mean & covariance)" begin
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8)))
        out = sample(
            backwards_compat_rng(42), model, scheme, 800;
            n_chains = 6, num_warmup = 300, memory = true, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        post_cov = cov(flat)
        @test isapprox(post_mean, μ; atol = 0.15)
        @test isapprox(post_cov, Σ; atol = 0.25)
    end

    @testset "warmup → sampling freezes ϵ and the metric" begin
        rng = backwards_compat_rng(99)
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8)))
        _, state = AbstractMCMC.step(rng, model, scheme; n_chains = 6, num_warmup = 200, memory = false, silent = true)
        for _ in 1:200
            _, state = AbstractMCMC.step_warmup(rng, model, scheme, state; num_warmup = 200)
        end
        a = hmc_adaptive_state(state)
        @test a.initialized
        frozen_ϵ = a.integrator.ϵ
        frozen_metric = copy(a.metric.M⁻¹)

        # Adaptation must have actually done something: a sensible step size and a
        # metric that moved away from the identity towards the marginal variances.
        # (Guards against a silent no-op that would still "freeze" at the defaults.)
        @test 0 < frozen_ϵ < 10
        @test frozen_metric != ones(length(μ))
        @test isapprox(frozen_metric, diag(Σ); atol = 0.5)

        # Sampling must not change the adaptive state: the metric/ϵ stay frozen.
        for _ in 1:50
            _, state = AbstractMCMC.step(rng, model, scheme, state)
        end
        a = hmc_adaptive_state(state)
        @test a.integrator.ϵ == frozen_ϵ
        @test a.metric.M⁻¹ == frozen_metric
    end

    @testset "interleaves with DE updates in a composite" begin
        scheme = setup_sampler_scheme(
            DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8)),
            setup_de_update();
            w = [0.5, 0.5]
        )
        out = sample(
            backwards_compat_rng(7), model, scheme, 1200;
            n_chains = 6, num_warmup = 400, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.2)
        @test isapprox(cov(flat), Σ; atol = 0.3)
    end

    @testset "threaded trajectory loop matches the serial result statistically" begin
        scheme = setup_sampler_scheme(DifferentialEvolutionMetropolis.setup_hmc_update(NUTS(0.8)))
        out = sample(
            backwards_compat_rng(13), model, scheme, 800;
            n_chains = 6, num_warmup = 300, parallel = true, progress = false,
            chain_type = DifferentialEvolutionOutput
        )
        flat = reshape(out.samples, :, length(μ))
        post_mean = vec(sum(flat; dims = 1) ./ size(flat, 1))
        @test isapprox(post_mean, μ; atol = 0.15)
        @test all(isfinite, flat)
    end
end
