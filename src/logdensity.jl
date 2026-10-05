# Keep existing wrappers unchanged; AbstractMCMC validates bare targets against
# the LogDensityProblems interface without transforming their parameters.
as_logdensity_model(model::LogDensityModel) = model
as_logdensity_model(model) = LogDensityModel(model)
