import ml.combust.mleap.{Dependencies, Common}

Common.defaultMleapSettings
Dependencies.sparkTestkit

libraryDependencies += "com.databricks" % "spark-avro_2.10" % "2.0.1"
