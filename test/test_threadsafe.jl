@testset "thread-safe per-worker model copies" begin
    struct ScratchNormalModel{T <: Real, V <: SubArray}
        means::Matrix{T}
        means_view::V
        function ScratchNormalModel(means::Matrix{T}) where {T <: Real}
            #for the sake of this example
            means_view = view(means, 1, :)
            return new{T, typeof(means_view)}(means, means_view)
        end
    end
    struct RefNormalModel{T <: Real}
        means::Matrix{T}
        function RefNormalModel(means::Matrix{T}) where {T <: Real}
            return new{T}(means)
        end
    end
    function LogDensityProblems.dimension(m::ScratchNormalModel)
        return length(m.means)
    end
    function LogDensityProblems.dimension(m::RefNormalModel)
        return length(m.means)
    end
    function LogDensityProblems.logdensity(m::ScratchNormalModel, x::AbstractVector{<:Real})
        return -(sum(abs2, x .- m.means[:]) / 2) - (sum(abs2, x[1:length(m.means_view)] .- m.means_view) / 2)
    end
    function LogDensityProblems.logdensity(m::RefNormalModel, x::AbstractVector{<:Real})
        return -(sum(abs2, x .- m.means[:]) / 2) - (sum(abs2, x[axes(m.means, 2)] .- view(m.means, 1, :)) / 2)
    end
    LogDensityProblems.capabilities(::ScratchNormalModel) = LogDensityProblems.LogDensityOrder{0}()
    LogDensityProblems.capabilities(::RefNormalModel) = LogDensityProblems.LogDensityOrder{0}()

    dim = 4
    n_its = 200
    seed = 42
    sampler = setup_sampler_scheme(setup_de_update())

    means = [10.0 1.0; 0.1 0.5]

    ref_model = AbstractMCMC.LogDensityModel(RefNormalModel(means))
    scratch_model = AbstractMCMC.LogDensityModel(ScratchNormalModel(means))

    # Sequential scratch matches sequential pure model (same log-density, same RNG)
    out_ref = sample(
        backwards_compat_rng(seed), ref_model, sampler, n_its;
        parallel = false, progress = false
    )
    out_scratch_seq = sample(
        backwards_compat_rng(seed), scratch_model, sampler, n_its;
        parallel = false, progress = false
    )
    @test all(isequal(out_ref[i].x, out_scratch_seq[i].x) for i in eachindex(out_ref))
    @test all(isequal(out_ref[i].ld, out_scratch_seq[i].ld) for i in eachindex(out_ref))

    # Parallel scratch matches sequential scratch (per-worker deepcopy preserves determinism)
    out_scratch_par = sample(
        backwards_compat_rng(seed), scratch_model, sampler, n_its;
        parallel = true, progress = false
    )
    @test all(isequal(out_scratch_seq[i].x, out_scratch_par[i].x) for i in eachindex(out_scratch_seq))
    @test all(isequal(out_scratch_seq[i].ld, out_scratch_par[i].ld) for i in eachindex(out_scratch_seq))
end
