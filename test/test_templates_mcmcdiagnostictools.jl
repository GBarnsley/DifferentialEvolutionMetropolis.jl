using MCMCDiagnosticTools

@testset "templates (R̂ epochs)" begin
    ld = AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0]))

    @testset "non memory runs" begin
        DREAMz(ld, 1000, 2; memory = false, save_burnt = false)
    end
    @testset "memory runs" begin
        deMC(ld, 1000, 2; thinning = 2, memory = true)
    end
    @testset "thin memory" begin
        deMCzs(ld, 1000, 2; thinning = 2, memory = true, memory_thin_interval = 5)
    end
    @testset "non-refill memory" begin
        deMC(ld, 100, 2; memory = true, memory_size = 50, memory_refill = true, save_burnt = true)
    end
    @testset "parameter simplifying" begin
        deMCzs(ld, 1000, 3; thinning = 2, memory = false, p_snooker = 0.0)
        DREAMz(ld, 1000, 3; thinning = 2, memory = true, p_γ₂ = 0.0, save_burnt = true)
    end
    @testset "annealing and parallel tempering" begin
        deMCzs(ld, 100, 2; thinning = 2, memory = true, annealing = true)
    end
    @testset "parallel" begin
        DREAMz(ld, 1000, 3; thinning = 2, memory = true, parallel = true)
    end
end
