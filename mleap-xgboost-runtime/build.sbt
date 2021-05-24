import ml.combust.mleap.{Dependencies, Common}

Common.defaultMleapSettings
Dependencies.xgboostRuntime

resolvers += DefaultMavenRepository

libraryDependencies ++= Seq(
  "ai.h2o" % "xgboost-predictor" % "0.3.1"
)