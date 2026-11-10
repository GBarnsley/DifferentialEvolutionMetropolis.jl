@testset "lack of Diagnostics Tools" begin
    @test_throws ErrorException sample(
        backwards_compat_rng(1234),
        AbstractMCMC.LogDensityModel(IsotropicNormalModel([-5.0, 5.0])),
        setup_de_update(),
        r̂_stopping_criteria;
        progress = false
    )
end
