// Databricks notebook source
// MAGIC %md * Author: Mohamed-Amine Baazizi
// MAGIC * Affiliation: LIP6 - Faculté des Sciences - Sorbonne Université
// MAGIC * Email: mohamed-amine.baazizi@lip6.fr
// MAGIC * Formation Continue Univ. Paris Dauphine, January 2025.
// MAGIC

// COMMAND ----------

// MAGIC %md # Decision Tree Induction in Spark ML on real-world data

// COMMAND ----------

// MAGIC %md The goal of this lab session is to code the end-to-end induction using real-world data.
// MAGIC The official documentation on Spark ML is available here:
// MAGIC * https://spark.apache.org/docs/latest/ml-guide.html
// MAGIC * https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/index.html

// COMMAND ----------

// MAGIC %md ## Pre-requisite

// COMMAND ----------

val path = "/FileStore/tables/SparkML/"
val dbfsDir = "dbfs:" + path

// COMMAND ----------

// MAGIC %md ### Global parameters 

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

// COMMAND ----------

// MAGIC %md ### Global imports

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import  org.apache.spark.ml.Pipeline 

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import spark.implicits._

// COMMAND ----------

// MAGIC %md ### Parameterizing the DT Pipeline
// MAGIC Write the body of the AutoPipeline method which takes the following arguments :
// MAGIC - the array of textual columns 
// MAGIC - the array of numeric columns
// MAGIC - the target label
// MAGIC - the maxCat parameter

// COMMAND ----------


def AutoPipeline(textCols: Array[String], numericCols: Array[String], target: String, maxCat: Int, handleInvalid: String):Pipeline = {
  //StringIndexer
  val inAttsNames = textCols ++ Array(target)
  val outAttsNames = inAttsNames.map(_prefix+_)

  val stringIndexer = new StringIndexer()
                              .setInputCols(inAttsNames)
                              .setOutputCols(outAttsNames)
                              .setHandleInvalid(handleInvalid)
  
  val features = outAttsNames.filterNot(_.contains(target))++numericCols
  
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

// MAGIC %md ### Data quality metrics collection
// MAGIC
// MAGIC Write methods to collect, for each column:
// MAGIC - the completeness information: the ratio of tuples with a non-null value
// MAGIC - the number of distinct values 
// MAGIC
// MAGIC

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

// MAGIC %md ## Part1: Decision Tree induction on real data

// COMMAND ----------

// MAGIC %md ### Data loading

// COMMAND ----------

// MAGIC %md
// MAGIC We use a real-world data  picked from a Kaggle challenge on loan classification detection [1]. 
// MAGIC
// MAGIC [1] https://www.kaggle.com/arashnic/banking-loan-prediction

// COMMAND ----------

val raw_data = spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(dbfsDir+"/loan.csv")
            .persist()
raw_data.count()

// COMMAND ----------

val target = "Approved"

// COMMAND ----------

raw_data.groupBy(target).count().show()

// COMMAND ----------

// MAGIC %md Rebalance the class distribution (useful for this dataset)

// COMMAND ----------

import org.apache.spark.sql.functions.{rand, when, col}
val data = raw_data.withColumn("t", when(rand>0.5,0).otherwise(1)).drop(target).withColumnRenamed("t",target).drop("t")
data.groupBy(target).count().show()

// COMMAND ----------

// MAGIC %md ### Collecting data quality metrics 
// MAGIC

// COMMAND ----------

var metadata = MDCompletenessDV(data)
metadata.orderBy($"compRatio".desc).show(100)

// COMMAND ----------

// MAGIC %md ### Experimenting different strategies

// COMMAND ----------

// MAGIC %md #### Setting #0: consider all attributes, index textual attributes

// COMMAND ----------

// MAGIC %md ##### Feature selection and transformation

// COMMAND ----------

val textCols = metadata.filter(col("colType").contains(_text)).select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val numericCols = metadata.filter(col("colType").contains(_numeric)).filter(!col("name").contains(target))
                  .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val maxCat = 32
val handleInvalid = "skip"

// COMMAND ----------

val pipeline = AutoPipeline(textCols,numericCols,target,maxCat,handleInvalid)
val data_s0_enc = pipeline.fit(data).transform(data)
                    .select(col(_featuresVecIndex), col(_prefix+target))
                    .withColumnRenamed(_prefix+target, _label)
data_s0_enc.count()

// COMMAND ----------

// data_s0_enc.show(5,false)

// COMMAND ----------

// MAGIC %md ##### Decision tree inference

// COMMAND ----------

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val dt = new DecisionTreeClassifier()
  .setImpurity("entropy")
  .setMaxBins(maxCat)


//split data 
val Array(trainingData_s0, testData_s0) = data_s0_enc.randomSplit(Array(0.7, 0.3))

//train data
val model = dt.fit(trainingData_s0)

val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
// println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

// MAGIC %md what do you observe?

// COMMAND ----------

// MAGIC %md #### Setting #1: restrict textual attributes to those with #distinct values < threshold  

// COMMAND ----------

// MAGIC %md ##### Feature selection and transformation

// COMMAND ----------

val maxCat = 32
val handleInvalid = "skip"

// COMMAND ----------

metadata.where("nbDistinctValues<=32").show()

// COMMAND ----------

val textCols = metadata.filter(col("colType").contains(_text)&&col("nbDistinctValues")<32)
                        .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val numericCols = metadata.filter(col("colType").contains(_numeric)).filter(!col("name").contains(target))
                  .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val pipeline = AutoPipeline(textCols,numericCols,target,maxCat,handleInvalid)
val data_s1_enc = pipeline.fit(data).transform(data)
                      .select(col(_featuresVecIndex), col(_prefix+target))
                      .withColumnRenamed(_prefix+target, _label)
data_s1_enc.count()

// COMMAND ----------

// MAGIC %md ##### Decision tree inference

// COMMAND ----------

val dt = new DecisionTreeClassifier()
  .setImpurity("entropy")
  .setMaxBins(maxCat)

// COMMAND ----------

//split data 
val Array(trainingData_s1, testData_s1) = data_s1_enc.randomSplit(Array(0.7, 0.3))

// COMMAND ----------

//train data
val model = dt.fit(trainingData_s1)

// val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
// println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

display(model)

// COMMAND ----------

// MAGIC %md ##### Accuracy estimation

// COMMAND ----------

val predictions = model.transform(testData_s1)
predictions.where("label!=prediction").show(5)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// COMMAND ----------

evaluator.explainParams

// COMMAND ----------

// MAGIC %md #### Setting #2: consider complete attributes only - preporcess some attributes

// COMMAND ----------

// MAGIC %md ##### Feature selection and transformation

// COMMAND ----------

val selected2 = metadata.where("compRatio=1")
selected2.show()

// COMMAND ----------

// MAGIC %md Create `cData` by restricting `data` to complete columns

// COMMAND ----------

val completeColumns = metadata.where("compRatio=1").select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect
val cData = data.select(completeColumns.head, completeColumns.tail: _*)

// COMMAND ----------

// MAGIC %md
// MAGIC Preprocess `cData` such that:
// MAGIC -  `ID` which has too many values is dropped 
// MAGIC - `Lead_Creation_Date` is converted into a date then split into its components (dayofmonth, month, year)
// MAGIC - `Source` which has the format "S123" is converted into a column `numSource` of type `int`  by removing the prefix 'S' and casting the result to an integer
// MAGIC
// MAGIC Store the result in `ppCData` 

// COMMAND ----------

cData.select("Lead_Creation_Date").withColumn("date",to_date(col("Lead_Creation_Date"), "dd/MM/yy"))
/*complete here*/

// COMMAND ----------

val ppCData = cData.withColumn("date",to_date(col("Lead_Creation_Date"), "dd/MM/yy"))
.withColumn("dayofmonth", dayofmonth(col("date")))
.withColumn("month", month(col("date")))
.withColumn("year", year(col("date")))
.withColumn("numSource", split($"Source","S")(1).cast("int"))
.drop("ID","Lead_Creation_Date","Source")

// COMMAND ----------

metadata = MDCompletenessDV(ppCData)
metadata.show()

// COMMAND ----------

val maxCat = 32 
val handleInvalid = "error"

// COMMAND ----------

val textCols = metadata.filter(col("colType").contains(_text))
                        .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val numericCols = metadata.filter(col("colType").contains(_numeric)).filter(!col("name").contains(target))
                  .select("name").rdd.flatMap(x=>x.toSeq).map(x=>x.toString).collect

// COMMAND ----------

val pipeline = AutoPipeline(textCols,numericCols,target,maxCat,handleInvalid)

val data_s2_enc = pipeline.fit(ppCData).transform(ppCData)
                      .select(col(_featuresVecIndex), col(_prefix+target))
                      .withColumnRenamed(_prefix+target, _label)

data_s2_enc.count()

// COMMAND ----------

val feautresMap = (textCols++numericCols).zipWithIndex.map{case(fname,idx) => (idx,fname)}.toMap

// COMMAND ----------

// MAGIC %md ##### Decision tree inference

// COMMAND ----------

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val dt = new DecisionTreeClassifier()
  .setImpurity("entropy")
  .setMaxBins(maxCat)


//split data 
val Array(trainingData_s2, testData_s2) = data_s2_enc.randomSplit(Array(0.7, 0.3))

//train data
val model = dt.fit(trainingData_s2)

val treeModel = model.asInstanceOf[DecisionTreeClassificationModel]
// println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

// println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

// MAGIC %md ##### Accuracy estimation

// COMMAND ----------

val predictions = model.transform(testData_s2)
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// COMMAND ----------

// MAGIC %md ## Part2: Model selection and parameter tuning

// COMMAND ----------

// MAGIC %md
// MAGIC Deriving the best model requires to experiment several hyper-parameters and to split data in several manners. SparkML facilitates these two tasks using the CrossValidator class whose documentation is here https://spark.apache.org/docs/latest/ml-tuning.html
// MAGIC Essentially, cross validation allows for building different (train,test) data in order to find the best model.
// MAGIC It is combined with Grid Search which allows for trying different combinations of parameters.
// MAGIC
// MAGIC We start by examining the possible parameters that can be set for a decision tree by invoking `extractParamMap` than we use `ParamGridBuilder` to set the parameters values that we would like to experiment.
// MAGIC We reuse the models trained in part 1.
// MAGIC
// MAGIC

// COMMAND ----------

model.extractParamMap

// COMMAND ----------

treeModel.explainParams

// COMMAND ----------

// MAGIC %md ### Param builder

// COMMAND ----------

//create a grid setting maxBins and minInstancesPerNode parameters
val dt_paramGrid = new ParamGridBuilder()
        .addGrid(dt.maxDepth, Array(4,6)) //try with two values 
        .addGrid(dt.minInstancesPerNode, Array(10,100)) //try with two values 
        .build()
//the total is 4 combinations of parameters

// COMMAND ----------

// MAGIC %md ### Cross-validation

// COMMAND ----------

//create k folds with k=5. 
val cv = new CrossValidator()
            .setEstimator(dt)
            .setEstimatorParamMaps(dt_paramGrid)
            .setEvaluator(new BinaryClassificationEvaluator)
            .setNumFolds(5)  // 5 pair de donnees entrainement/validation
            .setParallelism(20) //Nb Coeur dans le cluster. 

// COMMAND ----------

//train the different models
val cvModel = cv.fit(data_s2_enc)

// COMMAND ----------

val bestModel = cvModel.bestModel.asInstanceOf[DecisionTreeClassificationModel]

// COMMAND ----------

cvModel.getEstimator

// COMMAND ----------

cvModel.avgMetrics

// COMMAND ----------

cvModel.explainParams

// COMMAND ----------

bestModel.extractParamMap

// COMMAND ----------

bestModel

// COMMAND ----------

bestModel.featureImportances

// COMMAND ----------

val predictions = bestModel.transform(testData_s2)
// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator()
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

// COMMAND ----------

// MAGIC %md ## Part3: Explainability

// COMMAND ----------

feautresMap

// COMMAND ----------

bestModel.featureImportances

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row

// Prepare training documents from a list of (id, text, label) tuples.
val training = spark.createDataFrame(Seq(
  (0L, "a b c d e spark", 1.0),
  (1L, "b d", 0.0),
  (2L, "spark f g h", 1.0),
  (3L, "hadoop mapreduce", 0.0)
)).toDF("id", "text", "label")

// Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
val tokenizer = new Tokenizer()
  .setInputCol("text")
  .setOutputCol("words")
val hashingTF = new HashingTF()
  .setNumFeatures(1000)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("features")
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.001)
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, hashingTF, lr))

// Fit the pipeline to training documents.
val model = pipeline.fit(training)



// COMMAND ----------

// Prepare test documents, which are unlabeled (id, text) tuples.
val test = spark.createDataFrame(Seq(
  (4L, "spark i j k"),
  (5L, "l m n"),
  (6L, "spark hadoop spark"),
  (7L, "apache hadoop")
)).toDF("id", "text")

// Make predictions on test documents.
model.transform(test)
  .select("id", "text", "probability", "prediction").show()
  // .collect()
  // .foreach { case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
  //   println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  //     }
