package ml.dmlc.xgboost4j.scala.spark.mleap

import java.io.File

import ml.combust.bundle.BundleFile
import ml.combust.bundle.serializer.SerializationFormat
import ml.combust.mleap.core.types.{ScalarType, StructField, StructType}
import ml.combust.mleap.runtime.frame
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row}
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FunSpec}
import resource.managed

/**
  * Created by hollinwilkins on 9/16/17.
  */
case class PowerPlantTable(AT: Double, V : Double, AP : Double, RH : Double, PE : Double)

class XGBoostRegressionModelParitySpec extends FunSpec
  with BeforeAndAfterAll {
	val LOCAL_DATA = "/home/freem/mleap_coherent/mleap_nalivkin/mleap/mleap_engagement_local/mleap_engagement_local"
  val spark = SparkSession.builder().
    master("local[2]").
    appName("XGBoostRegressionModelParitySpec").
    getOrCreate()

  override protected def afterAll(): Unit = {
    spark.stop()
  }

  private val xgboostParams: Map[String, Any] = Map(
    "eta" -> 0.3,
    "max_depth" -> 2,
    "missing" -> 0.0f,
    "objective" -> "reg:squarederror",
    "early_stopping_rounds" ->2,
    "num_round" -> 15
  )

  val dataset: DataFrame = {
    import spark.sqlContext.implicits._

    spark.sqlContext.sparkContext.textFile(this.getClass.getClassLoader.getResource("datasources/xgboost_training.csv").toString)
      .map(x => x.split(","))
      .map(line => PowerPlantTable(line(0).toDouble, line(1).toDouble, line(2).toDouble, line(3).toDouble, line(4).toDouble))
      .toDF
  }

  val sparkTransformer: Transformer = {
    val featureAssembler = new VectorAssembler()
      .setInputCols(Array("AT", "V", "AP", "RH"))
      .setOutputCol("features")
    val regressor = new XGBoostRegressor(xgboostParams).
      setFeaturesCol("features").
      setLabelCol("PE").
      setPredictionCol("prediction").
      fit(featureAssembler.transform(dataset)).
      setLeafPredictionCol("leaf_prediction").
      setContribPredictionCol("contrib_prediction").
      setTreeLimit(2)

    SparkUtil.createPipelineModel(Array(featureAssembler, regressor))
  }

  def equalityTest(sparkDataset: DataFrame,
                   mleapDataset: DefaultLeapFrame): Unit = {
    val sparkPredictionCol = sparkDataset.schema.fieldIndex("prediction")
    val mleapPredictionCol = mleapDataset.schema.indexOf("prediction").get

    val sparkFeaturesCol = sparkDataset.schema.fieldIndex("features")
    val mleapFeaturesCol = mleapDataset.schema.indexOf("features").get

    val sparkCollected = sparkDataset.collect()
    val collected = mleapDataset.collect()

    sparkCollected.zip(collected).foreach {
      case (sp, ml) =>
        val v1 = sp.getDouble(sparkPredictionCol)
        val v2 = ml.getDouble(mleapPredictionCol)

        assert(sp.getAs[Vector](sparkFeaturesCol).toDense.values sameElements ml.getTensor[Double](mleapFeaturesCol).toDense.rawValues)
        assert(Math.abs(v2 - v1) < 0.0001)
    }
  }

  var bundleCache: Option[File] = None

  def serializedModel(transformer: Transformer): File = {
    import ml.combust.mleap.spark.SparkSupport._

    implicit val sbc = SparkBundleContext.defaultContext.withDataset(transformer.transform(dataset))

    bundleCache.getOrElse {
      new File("/tmp/mleap/spark-parity").mkdirs()
      val file = new File(s"/tmp/mleap/spark-parity/${classOf[XGBoostRegressionModelParitySpec].getName}.zip")
      file.delete()

      for(bf <- managed(BundleFile(file))) {
        transformer.writeBundle.format(SerializationFormat.Json).save(bf).get
      }

      bundleCache = Some(file)
      file
    }
  }

  def mleapTransformer(transformer: Transformer)
                      (implicit context: SparkBundleContext): frame.Transformer = {
    import ml.combust.mleap.runtime.MleapSupport._

    (for(bf <- managed(BundleFile(serializedModel(transformer)))) yield {
      bf.loadMleapBundle().get.root
    }).tried.get
  }

  private val mleapSchema = StructType(StructField("AT", ScalarType.Double),
    StructField("V", ScalarType.Double),
    StructField("AP", ScalarType.Double),
    StructField("RH", ScalarType.Double)).get

  it("produces the same results") {
    val data = dataset.collect().map {
      r => Row(r.getDouble(0), r.getDouble(1), r.getDouble(2), r.getDouble(3))
    }
    val frame = DefaultLeapFrame(mleapSchema, data)
    val mleapT = mleapTransformer(sparkTransformer)
    val sparkDataset = sparkTransformer.transform(dataset)
    val mleapDataset = mleapT.transform(frame).get

    equalityTest(sparkDataset, mleapDataset)
  }

	/*it("amobee test") {
		import java.io.File

		import ml.combust.bundle.BundleFile
		import ml.combust.bundle.serializer.SerializationFormat
		import ml.combust.mleap.runtime.MleapSupport._
		import ml.combust.mleap.spark.SparkSupport._
		import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
		import org.apache.spark.ml.Pipeline
		import org.apache.spark.ml.bundle.SparkBundleContext
		import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
		import org.apache.spark.sql._
		import resource._

		import scala.util.Random

		val spark = SparkSession.builder().
			appName("bid_response_feedback").
			config("spark.master", "local").
			getOrCreate()
		spark.sparkContext.setLogLevel("WARN")
		import spark.implicits._

		// dummy data
		val N = 1000
		val X = Seq.fill(N)((Random.nextDouble, Random.nextDouble, (Random.nextInt%100).toString, Random.nextDouble))
		val data: DataFrame = X.toDF("num1", "num2", "cat", "y")
		// create pipeline
		val stringIndexer = new StringIndexer().setInputCol("cat").setOutputCol("indexed_cat").setHandleInvalid("keep")
		val encoder = new OneHotEncoderEstimator().
			setInputCols(Array("indexed_cat")).
			setOutputCols(Array("encoded_cat")).
			setDropLast(false)
		val vectorAssembler = new VectorAssembler().setInputCols(Array("num1","num2", "encoded_cat")).setOutputCol("assembled_features")

		val regressor = new XGBoostRegressor().
			setFeaturesCol("assembled_features").
			setLabelCol("y").
			setMissing(0.0f).
			setNumRound(300).
			setEta(0.1).
			setLambda(0.1).
			setObjective("reg:squarederror")  // reg:linear
		val stages = Array(stringIndexer, encoder, vectorAssembler, regressor)
		val pipeline = new Pipeline().setStages(stages)

		// train model
		val model = pipeline.fit(data)
		val sparkPred = model.transform(data)

		// serialize model
		val savePath = "/tmp/model1/" // "/tmp/model.zip"
		new File(savePath).delete()
		val sbc = SparkBundleContext().withDataset(sparkPred)
		for(bf <- managed(BundleFile(s"file:$savePath"))) {  // s"jar:file:$savePath"
			model.writeBundle.format(SerializationFormat.Json).save(bf)(sbc).get
		}

		/*sparkPred.select("num1", "num2", "cat", "prediction").
			withColumnRenamed("prediction", "prediction_spark")
			.show(20, false)*/

		// deserialize model
		val bundle = (for(bundleFile <- managed(BundleFile(s"file:$savePath"))) yield {  // s"jar:file:$savePath"
			bundleFile.loadMleapBundle().get
		}).opt.get
		val mleapPipeline = bundle.root

		// compare spark prediction with mleap prediction
		// mleapPipeline.transform(SparkDataFrameOps(data.cache()).toSparkLeapFrame).get.toSpark.show(10)

		mleapPipeline.sparkTransform(
			sparkPred.select("num1", "num2", "cat", "prediction").
				withColumnRenamed("prediction", "prediction_spark")
		).withColumnRenamed("prediction", "prediction_mleap").
			select("prediction_spark", "prediction_mleap").show(20, false)

		assert(true)
	}*/

	import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
	import org.apache.spark.ml.Pipeline
	import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

	def mlPipeline(): Pipeline = {

		val sIndexer1 = new StringIndexer().setInputCol("placement_id")
			.setOutputCol("indexed_placement_id").setHandleInvalid("keep")
		val sIndexer2 = new StringIndexer().setInputCol("fp_tld")
			.setOutputCol("indexed_fp_tld").setHandleInvalid("keep")
		val sIndexer3 = new StringIndexer().setInputCol("video_max_duration")
			.setOutputCol("indexed_video_max_duration").setHandleInvalid("keep")
		val sIndexer4 = new StringIndexer().setInputCol("video_player_height")
			.setOutputCol("indexed_video_player_height").setHandleInvalid("keep")
		val sIndexer5 = new StringIndexer().setInputCol("inventory_source_id")
			.setOutputCol("indexed_inventory_source_id").setHandleInvalid("keep")
		val sIndexer6 = new StringIndexer().setInputCol("browser_type")
			.setOutputCol("indexed_browser_type").setHandleInvalid("keep")
		val sIndexer7 = new StringIndexer().setInputCol("video_min_duration")
			.setOutputCol("indexed_video_min_duration").setHandleInvalid("keep")
		val sIndexer8 = new StringIndexer().setInputCol("video_player_width")
			.setOutputCol("indexed_video_player_width").setHandleInvalid("keep")
		val sIndexer9 = new StringIndexer().setInputCol("video_linear_type_id")
			.setOutputCol("indexed_video_linear_type_id").setHandleInvalid("keep")
		val sIndexer10 = new StringIndexer().setInputCol("dma_id")
			.setOutputCol("indexed_dma_id").setHandleInvalid("keep")
		val sIndexer11 = new StringIndexer().setInputCol("video_playback_method_id")
			.setOutputCol("indexed_video_playback_method_id").setHandleInvalid("keep")
		val sIndexer12 = new StringIndexer().setInputCol("video_player_size_id")
			.setOutputCol("indexed_video_player_size_id").setHandleInvalid("keep")

		val vectorAssembler = new VectorAssembler().setInputCols(Array(sIndexer1.getOutputCol,
			sIndexer2.getOutputCol, sIndexer3.getOutputCol, sIndexer4.getOutputCol, sIndexer5.getOutputCol,
			sIndexer6.getOutputCol, sIndexer7.getOutputCol, sIndexer8.getOutputCol, sIndexer9.getOutputCol,
			sIndexer10.getOutputCol, sIndexer11.getOutputCol, sIndexer12.getOutputCol))
			.setOutputCol("assembled_features")

		// xgboost parameters
		val xgbParam = Map(
			"objective" -> "binary:logistic",
			"num_round" -> 500,
			"max_depth" -> 6,
			"eta" -> 0.25f,
			"subsample" -> 1.0f,
			"colsample_bynode" -> 0.9f
		)

		// model definition
		val xgbClassifier = new XGBoostClassifier(xgbParam)
			.setFeaturesCol("assembled_features")
			.setLabelCol("label")
			.setWeightCol("weight")
			.setMissing(0.0f)

		// ML pipeline definition
		val pipeline = new Pipeline()
			.setStages(Array(sIndexer1, sIndexer2, sIndexer3, sIndexer4, sIndexer5, sIndexer6, sIndexer7,
				sIndexer8, sIndexer9, sIndexer10, sIndexer11, sIndexer12, vectorAssembler, xgbClassifier))

		pipeline
	}

	it("amobee int test") {
		import java.io.File

		import ml.combust.bundle.BundleFile
		import ml.combust.bundle.serializer.SerializationFormat
		import ml.combust.mleap.runtime.MleapSupport._
		import ml.combust.mleap.spark.SparkSupport._
		import org.apache.spark.ml.bundle.SparkBundleContext
		import org.apache.spark.sql._
		import resource._

		val spark = SparkSession.builder().
			appName("bid_response_feedback").
			config("spark.master", "local").
			getOrCreate()
		spark.sparkContext.setLogLevel("WARN")

		/*
		* Point testPath to the folder with the following files:
		*   - part-*.parquet
		*   - model/xxxSchema.json
		* */
		val testPath = LOCAL_DATA

		val data = spark.read.load(s"$testPath/part-*")//.sample(0.01f).cache()
		//data.show()
		println(data.count())
		val pipeline = mlPipeline()
		val model = pipeline.fit(data)

		val sparkPred = model.transform(data.limit(10))

		// serialize model
		val savePath = s"$testPath/model/model.zip" // "/tmp/model.zip"
		new File(savePath).delete()
		val sbc = SparkBundleContext().withDataset(sparkPred)
		for(bf <- managed(BundleFile(s"jar:file:$savePath"))) {  // s"jar:file:$savePath"
			model.writeBundle.format(SerializationFormat.Json).save(bf)(sbc).get
		}

		// deserialize model
		val bundle = (for(bundleFile <- managed(BundleFile(s"jar:file:$savePath"))) yield {  // s"jar:file:$savePath"
			bundleFile.loadMleapBundle().get
		}).opt.get
		val mleapPipeline = bundle.root

		// compare spark prediction with mleap prediction
		// mleapPipeline.transform(SparkDataFrameOps(data.cache()).toSparkLeapFrame).get.toSpark.show(10)

		mleapPipeline.sparkTransform(
			sparkPred.select("video_linear_type_id", "video_min_duration", "video_player_height", "weight", "video_player_size_id", "inventory_source_id", "video_playback_method_id", "dma_id", "video_max_duration", "video_player_width", "browser_type", "label", "placement_id", "fp_tld", "probability")
				.withColumnRenamed("probability", "probability_spark")
		).withColumnRenamed("probability", "probability_mleap").
			select("probability_spark", "probability_mleap").show(20, false)
	}
}


object XGBoostRegressionModelParitySpec {
	def probExtract(prob: Vector): Double = prob(1)
}

