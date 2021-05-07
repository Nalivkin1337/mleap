import ml.combust.mleap.{Dependencies, Common}

Common.defaultMleapSettings
Dependencies.xgboostSpark
dependencyOverrides += "com.esotericsoftware.kryo" % "kryo" % "2.21"
javaOptions in Test ++= sys.env.get("XGBOOST_JNI").map {
  jniPath => Seq(s"-Djava.library.path=$jniPath")
}.getOrElse(Seq())
