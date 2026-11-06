@testset "lack of Diagnostics Tools" begin
    @test_logs match_mode = :any (
        :error, "Please load MCMCDiagnosticTools.jl to use `r̂_stopping_criteria`",
    ) sample(
        backwards_compat_rng(1234),
        AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0])),
        setup_de_update(),
        r̂_stopping_criteria;
        progress = false
    )
end
