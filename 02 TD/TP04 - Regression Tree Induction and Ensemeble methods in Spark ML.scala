// Databricks notebook source
// MAGIC %md * Author: Mohamed-Amine Baazizi
// MAGIC * Affiliation: LIP6 - Faculté des Sciences - Sorbonne Université
// MAGIC * Email: mohamed-amine.baazizi@lip6.fr
// MAGIC * Formation Continue Univ. Paris Dauphine, February 2025.

// COMMAND ----------

// MAGIC %md
// MAGIC # Regression Tree Induction and Ensemeble methods in Spark ML

// COMMAND ----------

// MAGIC %md
// MAGIC The goal of this lab session is to illustrate the Regression Tree induction algorithm using Spark ML. As in the previous lab, the focus will be on the data preparation phase (feature selection, extraction and transformation).
// MAGIC The lab is divided into two parts: 
// MAGIC * the first part introduces the decision tree regression in Spark ML, using a synthetic dataset 
// MAGIC * the second part is left as an exercice and is meant to code the end-to-end regression on a real-world data. A discussion on the quality of the obtained result is expected.
// MAGIC * The third part presents ensemble methods  

// COMMAND ----------

// MAGIC %md
// MAGIC ## Pre-requisite

// COMMAND ----------

val path = "/FileStore/tables/SparkML/"
val dbfsDir = "dbfs:" + path

// COMMAND ----------

// MAGIC %md ### Imports

// COMMAND ----------

//pipeline 
val _label = "label"
val _prefix = "indexed_"
val _featuresVec = "featuresVec"
val _featuresVecIndex = "features"

//metadata extraction
val _text = "textType"
val _numeric = "numericType"
val _other = "otherType"

var _featureIndices = Array(("","")) 

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, DecisionTreeRegressor}

import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import  org.apache.spark.ml.Pipeline 

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import spark.implicits._

import org.apache.spark.ml.regression.{RandomForestRegressor, RandomForestRegressionModel}


// COMMAND ----------

// MAGIC %md ### Useful methods

// COMMAND ----------

val description = Map("RMSE"->"Root Mean Squared Error", "MAE"->"Mean Absolute Error")

def printMetric(name: String, dtmet: Double, rfmet: Double) = 
    println(s"${description(name)} on test data \n \t - using the decision tree: $dtmet \t using the random forest: $rfmet ")




// COMMAND ----------


def getParams(tree: DecisionTreeRegressionModel) = Map("treeDepth"-> tree.depth 
                                                            ,"numNodes"-> tree.numNodes 
                                                            ,"numFeatures"-> tree.numFeatures)

// COMMAND ----------

def AutoPipelineReg(textCols: Array[String], numericCols: Array[String], maxCat: Int, handleInvalid: String):Pipeline = {
  //StringIndexer
  val inAttsNames = textCols 
  val outAttsNames = inAttsNames.map(_prefix+_)

  val stringIndexer = new StringIndexer()
                              .setInputCols(inAttsNames)
                              .setOutputCols(outAttsNames)
                              .setHandleInvalid(handleInvalid)
  
  val features = outAttsNames++numericCols
  
  _featureIndices = features.zipWithIndex.map{case (f,i)=> ("feature "+i,f)}

  
  //vectorAssembler
  val vectorAssembler = new VectorAssembler()
                            .setInputCols(features)
                            .setOutputCol(_featuresVec)
                            .setHandleInvalid(handleInvalid)
  
  //VectorIndexer
  val vectorIndexer = new VectorIndexer()
                            .setInputCol(_featuresVec)
                            .setOutputCol(_featuresVecIndex)
                            .setMaxCategories(maxCat)
                            .setHandleInvalid(handleInvalid)
  
  val pipeline = new Pipeline()
                    .setStages(Array(stringIndexer,vectorAssembler,vectorIndexer))
  


  return pipeline
}


// COMMAND ----------

case class MetaData(name: String, origType: String, colType: String, compRatio: Float, nbDistinctValues: Long)


//considers only three types: numeric, textual and other
def whichType(origType: String) = origType match {
  case "StringType" => _text
  case "IntegerType"|"DoubleType" => _numeric
  case _ => _other
}

def MDCompletenessDV(data: DataFrame): DataFrame = {
  val total_count = data.count()
  val res = data.dtypes.map{
    case(colName, colType)=>MetaData(colName, 
                                      colType, 
                                      whichType(colType),
                                      data.filter(col(colName).isNotNull).count.toFloat/total_count,
                                      data.select(colName).distinct().count)
  }.toList
  val metadata = res.toDS().toDF()
  metadata.persist()  
  metadata.count()
  return metadata
}

// COMMAND ----------

// MAGIC %md
// MAGIC # Part1: synthetic dataset

// COMMAND ----------

// MAGIC %md
// MAGIC This part will show how to use infer regression trees in Spark ML. 
// MAGIC It is very close to the inferrence of decision trees seen in the previous lab. 
// MAGIC The target attribute is `hours`.

// COMMAND ----------

// MAGIC %md
// MAGIC ## data loading

// COMMAND ----------

import spark.implicits._

case class tuple(outlook: String,temp: String,humidity: String,windy: String,hours: Double)

val data = Seq(tuple("rainy","hot","high","FALSE",25.0),
tuple("rainy","hot","high","TRUE",30.0),
tuple("overcast","hot","high","FALSE",46.0),
tuple("sunny","mild","high","FALSE",45.0),
tuple("sunny","cool","normal","FALSE",52.0),
tuple("sunny","cool","normal","TRUE",23.0),
tuple("overcast","cool","normal","TRUE",43.0),
tuple("rainy","mild","high","FALSE",35.0),
tuple("rainy","cool","normal","FALSE",38.0),
tuple("sunny","mild","normal","FALSE",46.0),
tuple("rainy","mild","normal","TRUE",48.0),
tuple("overcast","mild","high","TRUE",52.0),
tuple("overcast","hot","normal","FALSE",44.0),
tuple("sunny","mild","high","TRUE",30.0)
).toDS

data.count()
data.show()

// COMMAND ----------

// MAGIC %md ### Feature selection and transformation

// COMMAND ----------

val target = "hours"

// COMMAND ----------

val textCols = data.columns.filterNot(_.contains(target))
val numericCols: Array[String] = Array()

// COMMAND ----------

val maxCat = 32
val handleInvalid = "skip"

// COMMAND ----------

// val pipeline = AutoPipeline(textCols,numericCols,target,maxCat,handleInvalid)

// COMMAND ----------

// pipeline.getStages(0).asInstanceOf[StringIndexer].getOutputCols

// COMMAND ----------

val pipeline = AutoPipelineReg(textCols,numericCols,maxCat,handleInvalid)
val data_enc = pipeline.fit(data).transform(data)
                    .select(col(_featuresVecIndex), col(target))
                    .withColumnRenamed(target, _label)
data_enc.count()

// COMMAND ----------

data_enc.show()

// COMMAND ----------

// MAGIC %md ### Decision tree inference

// COMMAND ----------

// Train a DecisionTree model.
val dt = new DecisionTreeRegressor()
  
// Train model. This also runs the indexer.
val model = dt.fit(data_enc)

// val treeModel = model.asInstanceOf[DecisionTreeRegressionModel]
// println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

display(model) //println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

_featureIndices

// COMMAND ----------

// MAGIC %md
// MAGIC # Part 2: real data

// COMMAND ----------

// MAGIC %md
// MAGIC In this part we will adapt the instructions from the previous part to infer a regression tree from real-life data.
// MAGIC As usual, we need to extract some statistics about this data in order to identify the features that are really useful.

// COMMAND ----------

// MAGIC %md ### Data loading

// COMMAND ----------

val raw_data = spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(dbfsDir+"/autos.csv")
            .persist()
raw_data.count()

// COMMAND ----------

val target = "price"

// COMMAND ----------

val data = raw_data

// COMMAND ----------

// MAGIC %md ### Collecting data quality metrics 

// COMMAND ----------

var metadata = MDCompletenessDV(raw_data)
metadata.orderBy(col("nbDistinctValues").desc,$"colType").show(100)

// COMMAND ----------

// MAGIC %md ### Preprocessing
// MAGIC - Get rid of `dateCrawled`, `name`
// MAGIC - Convert `dateCreated` and `lastSeen` to timestamps then get rid of the time and keep only the date
// MAGIC - Bucketize `yearOfRegistration` by creating decades ranging from 0 to n
// MAGIC - Either remove `postalCode` or extract the city from it by integrating an external dataset 

// COMMAND ----------

// raw_data.select("monthOfRegistration").show(5,false)
// raw_data.select("dateCreated").show(5,false)
// raw_data.select(length($"postalCode")).distinct().show(5,false)
// raw_data.select("lastSeen").show(5,false)

// COMMAND ----------

val ppData =data
.drop("dateCrawled", "name", "postalCode")
.withColumn("dc",to_date(col("dateCreated")))
.withColumn("dayofmonthDC", dayofmonth($"dc"))
.withColumn("monthDC", month($"dc"))
.withColumn("yearDC", year($"dc"))
.withColumn("ls",to_date(col("lastSeen")))
.withColumn("dayofmonthLS", dayofmonth($"ls"))
.withColumn("monthLS", month($"ls"))
.withColumn("yearLS", year($"ls"))
// .select("dayofmonthDC", "monthDC", "yearDC", "dayofmonthLS", "monthLS", "yearLS")
//.select("yearLS").distinct().show()
.drop("dateCreated","lastSeen","dc","ls")
.where("yearOfRegistration between 1911 and 2019")
.where("price<=17500 and price>100")

// COMMAND ----------

metadata = MDCompletenessDV(ppData)
metadata.orderBy(col("nbDistinctValues").desc,col("colType")).show(100)

// COMMAND ----------

// MAGIC %md ##### bucketizer (optionnel)

// COMMAND ----------

import org.apache.spark.ml.feature.Bucketizer

val yearSplits = (1911 to 2021 by 10).map(_.toDouble).toArray

val bucketizer = new Bucketizer()
  .setInputCol("yearOfRegistration")
  .setOutputCol("bucketedYearOfRegistration")
  .setSplits(yearSplits)

val bucketedData = bucketizer.transform(ppData)
// bucketedData.select("yearOfRegistration","bucketedYearOfRegistration").show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Feature selection and transformation

// COMMAND ----------

// textCols
val textCols = metadata.filter(col("colType").contains(_text)&&col("nbDistinctValues")<=40)
                        .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

// numericCols
val numericCols = metadata.filter(col("colType").contains(_numeric)).filter(!col("name").contains(target))
                  .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val maxCat = 40

val handleInvalid = "skip"


// COMMAND ----------

val pipeline = AutoPipelineReg(textCols,numericCols,maxCat,handleInvalid)

// COMMAND ----------

val data = ppData
val pipeline = AutoPipelineReg(textCols,numericCols,maxCat,handleInvalid)
val data_s0_enc = pipeline.fit(data).transform(data)
                      .select(col(_featuresVecIndex), col(target))
                      .withColumnRenamed(target, _label)
data_s0_enc.count()

// COMMAND ----------

// MAGIC %md ### Decision tree inference

// COMMAND ----------

// Train a DecisionTree model.
val dt = new DecisionTreeRegressor()
  .setMaxBins(maxCat)

//split data 
val Array(trainingData_s0, testData_s0) = data_s0_enc.randomSplit(Array(0.7, 0.3))

// Train model. This also runs the indexer.
val model = dt.fit(trainingData_s0)

// val treeModel = model.asInstanceOf[DecisionTreeRegressionModel]
// println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

_featureIndices.foreach(println)

// COMMAND ----------

model.featureImportances

// COMMAND ----------

display(model)

// COMMAND ----------

// MAGIC %md ### Accuracy estimation

// COMMAND ----------

// Make predictions.
val predictions = model.transform(testData_s0)

// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

// COMMAND ----------

evaluator.explainParams

// COMMAND ----------

evaluator.setMetricName("mae")
val mae = evaluator.evaluate(predictions)
println(s" Mean Absolute Error (MAE) on test data = $mae")

// COMMAND ----------

ppData.select("price").describe().show()

// COMMAND ----------

// MAGIC %md # Part 3: Ensemble methods (random forests)

// COMMAND ----------

// MAGIC %md
// MAGIC Decision trees are very interesting for their interpretability but tend to overfit the training data which may lead to increase the variance when predicting the values for the new data.
// MAGIC Random forests mitigate this shortcoming by derviging different trees for the same data and averaging the predictions obtained from each tree to obtain the prediction for the new data. 
// MAGIC When building random forests randomness is introduced by training on a sample of the data and by restricting to a sub-set of attributes.
// MAGIC
// MAGIC The random forest inference of SparkML can be parametrized by setting the rate of the sample data and the size of the attributes as follows:
// MAGIC
// MAGIC * the ratio of the sample w.r.t. the original is set through the parameter  `subsamplingRate`
// MAGIC * the ratio of the attribute subset w.r.t. the entire features is set through the parameter `featureSubsetStrategy`
// MAGIC
// MAGIC It also possible to set other parameters like
// MAGIC * `numTrees` which fixes the number of trees to be inferred and
// MAGIC * `maxDepth` which sets the maximum depth of these trees. 
// MAGIC
// MAGIC The official documentation is below:
// MAGIC * https://spark.apache.org/docs/latest/mllib-ensembles.html#random-forests
// MAGIC
// MAGIC In the following we will train a regression tree and a random forest on our two datasets and compare, for each dataset, the precision of the tree against that of the forest.

// COMMAND ----------

// MAGIC %md 
// MAGIC  featureSubsetStrategy
// MAGIC  voir https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/regression/RandomForestRegressor.html#featureSubsetStrategy:org.apache.spark.ml.param.Param[String]
// MAGIC  
// MAGIC  
// MAGIC The number of features to consider for splits at each tree node. Supported options:
// MAGIC
// MAGIC - "auto": Choose automatically for task: If numTrees == 1, set to "all." If numTrees greater than 1 (forest), set to "sqrt" for classification and to "onethird" for regression.
// MAGIC - "all": use all features
// MAGIC - "onethird": use 1/3 of the features
// MAGIC - "sqrt": use sqrt(number of features)
// MAGIC - "log2": use log2(number of features)
// MAGIC - "n": when n is in the range (0, 1.0], use n * number of features. When n is in the range (1, number of features), use n features. (default = "auto")
// MAGIC
// MAGIC These various settings are based on the following references:
// MAGIC
// MAGIC - log2: tested in Breiman (2001)
// MAGIC - sqrt: recommended by Breiman manual for random forests
// MAGIC
// MAGIC The defaults of sqrt (classification) and onethird (regression) match the R randomForest package.
// MAGIC

// COMMAND ----------

// MAGIC %md #### Random forest inference

// COMMAND ----------

val rf = new RandomForestRegressor()
  .setMaxBins(maxCat)
  .setNumTrees(5)
  .setFeatureSubsetStrategy("onethird").setSubsamplingRate(0.5)

// Train model. This also runs the indexer.
val rf_model = rf.fit(trainingData_s0)



// COMMAND ----------

val rfModel = rf_model.asInstanceOf[RandomForestRegressionModel]
// rfModel.toDebugString

// COMMAND ----------

rfModel.trees.map(getParams).foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Predictions on test data

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")

val dt_predictions = model.transform(testData_s0)
val rf_predictions = rf_model.transform(testData_s0)


/*using the decision tree*/
evaluator.setMetricName("rmse")
val dt_rmse = evaluator.evaluate(dt_predictions)
evaluator.setMetricName("mae")
val dt_mae = evaluator.evaluate(dt_predictions)

/*using the random forest*/
val rf_mae = evaluator.evaluate(rf_predictions)
evaluator.setMetricName("rmse")
val rf_rmse = evaluator.evaluate(rf_predictions)

/**/
printMetric("RMSE",dt_rmse,rf_rmse)
printMetric("MAE",dt_mae,rf_mae)



// COMMAND ----------

// MAGIC %md
// MAGIC ### GBT inference

// COMMAND ----------

import org.apache.spark.ml.regression.{GBTRegressor,GBTRegressionModel}

val gbt = new GBTRegressor()
  .setMaxBins(maxCat)
  .setFeatureSubsetStrategy("onethird").setSubsamplingRate(0.5)

// Train model. This also runs the indexer.
val gbt_model = gbt.fit(trainingData_s0)

// COMMAND ----------

val gbtModel = gbt_model.asInstanceOf[GBTRegressionModel]

// COMMAND ----------

gbtModel.trees.map(getParams).foreach(println)

// COMMAND ----------

// MAGIC %md
// MAGIC ##### Predictions on test data

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator

// Select (prediction, true label) and compute test error.
val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")

val gbt_predictions = gbt_model.transform(testData_s0)



/*using the decision tree*/

/*using the random forest*/
evaluator.setMetricName("mae")
val gbt_mae = evaluator.evaluate(gbt_predictions)
evaluator.setMetricName("rmse")
val gbt_rmse = evaluator.evaluate(gbt_predictions)



