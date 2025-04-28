// Databricks notebook source
// MAGIC %md * Author: Mohamed-Amine Baazizi
// MAGIC * Affiliation: LIP6 - Faculté des Sciences - Sorbonne Université
// MAGIC * Email: mohamed-amine.baazizi@lip6.fr
// MAGIC * Formation Continue Univ. Paris Dauphine, Septembre 2022.
// MAGIC

// COMMAND ----------

// MAGIC %md # Decision Tree Induction in Spark ML

// COMMAND ----------

// MAGIC %md The goal of this lab session is to illustrate the Decision Tree induction algorithm using Spark ML. The focus will be on the data preparation phase which most official documentations tend to ignore. The lab is dedicated to using decision trees for classification using both continuous and categorical features. 
// MAGIC The lab is divided into three parts: 
// MAGIC * the first part recalls some useful feature extraction methods 
// MAGIC * the second part introduces the mechanism for inferring a decision tree in Spark ML, using a synthetic dataset 
// MAGIC * the third part is dedicated to automate the feature transformation pipeleline for inferring a decision tree

// COMMAND ----------

// MAGIC %md ## Spark ML in a nutshell 
// MAGIC There are two libararies implementing mainstream ML algorithms in Spark: one that is based on RDDs and will no longer be maintained, and another one based on Dataframe and which will be used in this lab.
// MAGIC The Dataframe-based ML library is more interesting in that it facilitates the feature extraction/transformation operations and offers an abstraction that allows chaining many operations in a pipeline that can be reused and optimized.
// MAGIC For more details, check 
// MAGIC * https://spark.apache.org/docs/latest/ml-guide.html
// MAGIC * https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/index.html
// MAGIC

// COMMAND ----------

// MAGIC %md Load both files into the local file system and note down the paths to the loaded files in the DBFS

// COMMAND ----------

// MAGIC %md ## Pre-requisite

// COMMAND ----------

// dbutils.fs.rm(dbfsDir,true)

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

// MAGIC %md
// MAGIC ### Data extraction (valid for the entire Spark ML part - run only once)

// COMMAND ----------

// MAGIC %sh
// MAGIC #download the data
// MAGIC #wget --no-verbose https://nuage.lip6.fr/s/LYJ5DPCsS8RSzwB/download/MLData.tgz -O /tmp/MLData.tgz
// MAGIC wget --no-verbose https://nuage.lip6.fr/s/89BG8HD9r3iE693/download/MLData.tgz -O /tmp/MLData.tgz
// MAGIC

// COMMAND ----------

// MAGIC %sh
// MAGIC #decompress the archive
// MAGIC tar -xzvf /tmp/MLData.tgz  --directory /tmp/

// COMMAND ----------

// MAGIC %sh
// MAGIC rm  /tmp/MLData.tgz
// MAGIC rm /tmp/MLData/\._loan.csv
// MAGIC ls -hal /tmp/MLData
// MAGIC

// COMMAND ----------

// create a directory in the Distributed File System (DFS)
println("dbfsDir est :" + dbfsDir)
dbutils.fs.mkdirs(dbfsDir)
display(dbutils.fs.ls(dbfsDir))

// COMMAND ----------

// copy the content of Books from into the DFS
dbutils.fs.cp("file:/tmp/MLData/", dbfsDir, recurse=true)
display(dbutils.fs.ls(dbfsDir))

// COMMAND ----------

// MAGIC %md ## First part: feature extraction methods and pipelines

// COMMAND ----------

// MAGIC %md Most ML algorithms operate on numeric data. Spark MLlib provides several methods for transforming data in a suitable format. These methodes are described here
// MAGIC https://spark.apache.org/docs/latest/ml-features.html
// MAGIC In this lab, we need the following feature transformers:
// MAGIC * `StringIndexer`: which maps strings to numeric indices following one of  four possible encoding schemes (cf. documentation) 
// MAGIC * `IndexToString`: which maps the numeric indices back to the strings they obtained from
// MAGIC * `VectorInedexer`: which encodes categorical features using category indices allowing algorithms like decision tree induction to take categorical features into account
// MAGIC
// MAGIC To illustrate these transformers using the synthetic dataset, assign to `workingDir` the path of the directory containing the data downloaded in the previous step then run the different commands.
// MAGIC

// COMMAND ----------

// MAGIC %md ### Loading the data

// COMMAND ----------

import spark.implicits._

case class tuple(age: String,income: String,student: String,credit_rating: String,buys_computer: String)

val data = Seq(tuple("young","high","no","fair","no"),
               tuple("young","high","no","excellent","no"),
               tuple("middle","high","no","fair","yes"),
               tuple("senior","medium","no","fair","yes"),
               tuple("senior","low","yes","fair","yes"),
               tuple("senior","low","yes","excellent","no"),
               tuple("middle","low","yes","excellent","yes"),
               tuple("young","medium","no","fair","no"),
               tuple("young","low","yes","fair","yes"),
               tuple("senior","medium","yes","fair","yes"),
               tuple("young","medium","yes","excellent","yes"),
               tuple("middle","medium","no","excellent","yes"),
               tuple("middle","high","yes","fair","yes"),
               tuple("senior","medium","no","excellent","no")).toDS()
data.printSchema
data.show()

// COMMAND ----------

// MAGIC %md ### String Indexer

// COMMAND ----------

// MAGIC %md The following snippet creates a new column `indexed_age` encoding the age column.

// COMMAND ----------


import org.apache.spark.ml.feature.StringIndexer

val field = "age"
val ageIndexer =  new StringIndexer()
          .setInputCol(field)
          .setOutputCol("indexed_"+field)

val ageIndexed = ageIndexer.fit(data).transform(data)
ageIndexed.show()

// COMMAND ----------

ageIndexed.select("age", "indexed_age").distinct().show()

// COMMAND ----------

// MAGIC %md The default encoding uses frequency and assigns the lowest index to the most frequent label. In case of ties, lexicographic order is used to break the tie. 
// MAGIC The following dataframe shows the frequency of each categroy.
// MAGIC Observe that "senior" has 0 while "young" has index 1. 

// COMMAND ----------

data.groupBy("age").count().show()

// COMMAND ----------

// MAGIC %md ### IndexToString

// COMMAND ----------

// MAGIC %md The following snippet creates a new column `originalAge` decoding the `indexed_age` column

// COMMAND ----------

import org.apache.spark.ml.feature.IndexToString

val inputColSchema = ageIndexed.schema(ageIndexer.getOutputCol)
val ageConverter = new IndexToString()
  .setInputCol(ageIndexer.getOutputCol)
  .setOutputCol("originalAge")

val ageConverted = ageConverter.transform(ageIndexed)
ageConverted.show()

// COMMAND ----------

// MAGIC %md ### Vector Assembler

// COMMAND ----------

// MAGIC %md It is possible to create a vector starting from many columns of Double type. To illusatre, we encode the `income` column using a StringIndexer than create a vector of the indexed `age` and `income` columns. 

// COMMAND ----------

//encode income
val field = "income"
val incomeIndexer =  new StringIndexer()
          .setInputCol(field)
          .setOutputCol("indexed_"+field)

val ageIncomeIndexed = incomeIndexer.fit(ageIndexed).transform(ageIndexed)
ageIncomeIndexed.show()

// COMMAND ----------

//assemble age and income
import org.apache.spark.ml.feature.VectorAssembler

val vecAssembler = new VectorAssembler()
                    .setInputCols(Array("indexed_age","indexed_income"))
                    .setOutputCol("ageIncomeVec")

val ageIncomeIndexedVec = vecAssembler.transform(ageIncomeIndexed)
ageIncomeIndexedVec.drop("age","income").show()

// COMMAND ----------

// MAGIC %md ### One hot encoding
// MAGIC

// COMMAND ----------

import org.apache.spark.ml.feature.OneHotEncoder

val oneHotEncoder = new OneHotEncoder()
  .setInputCols(Array("indexed_age", "indexed_income"))
  .setOutputCols(Array("category_age", "category_income"))
  .setDropLast(false)


val oneHotEncoderModel = oneHotEncoder.fit(ageIncomeIndexed)

val encoded = oneHotEncoderModel.transform(ageIncomeIndexed)
encoded.show()

// COMMAND ----------

// MAGIC %md ### Vector Indexer

// COMMAND ----------

// MAGIC %md The following snippet is used for indexing categorical features by mapping the original values to indices ranging from 0 to the number of distinct values. The vector indexer method first decides whether a  feature is categorical by comparing the `MaxCategories` parameter with  the number of distinct values.
// MAGIC

// COMMAND ----------

import org.apache.spark.ml.feature.VectorIndexer

val vecIndexer = new VectorIndexer()
  .setInputCol("ageIncomeVec")
  .setOutputCol("ageIncomeVecInd")
  .setMaxCategories(3)

val vecIndexerModel = vecIndexer.fit(ageIncomeIndexedVec)
val categoricalFeatures: Set[Int] = vecIndexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} " +
  s"categorical features: ${categoricalFeatures.mkString(", ")}")

val ageIncomeIndexedVecInd = vecIndexerModel.transform(ageIncomeIndexedVec)
ageIncomeIndexedVecInd.select("ageIncomeVec","ageIncomeVecInd").show()

// COMMAND ----------

// MAGIC %md ### VectorIndexer: another example

// COMMAND ----------

import org.apache.spark.ml.linalg.{Vectors,Vector}

case class tuple (vec: Vector)
val sample = spark.createDataFrame(
    Seq( tuple(Vectors.dense(1.0, 1.0, 18.0) )
        , tuple(Vectors.dense(0.0, 2.0, 20.0) )
        , tuple(Vectors.dense(1.0, 0.0, 18.0) )
        , tuple(Vectors.dense(2.0, 3.0, 11.0) )
       )).toDF("input_vec")
sample.show()


// COMMAND ----------

import org.apache.spark.ml.linalg.{Vectors,Vector}

case class tuple (vec: Vector)
val sparse_sample = spark.createDataFrame(
    Seq( tuple(Vectors.dense(1.0, 1.0, 18.0) )
        , tuple(Vectors.dense(0.0, 2.0, 20.0) )
        , tuple(Vectors.dense(1.0, 0.0, 18.0).toSparse )
        , tuple(Vectors.dense(2.0, 3.0, 11.0).toSparse )
       )).toDF("input_vec")
sparse_sample.show(truncate=false)


// COMMAND ----------

import org.apache.spark.ml.feature.VectorIndexer

val vecIndexer = new VectorIndexer()
  .setInputCol("input_vec")
  .setOutputCol("output_vec")
  .setMaxCategories(3)

val vecIndexerModel = vecIndexer.fit(sample)
val categoricalFeatures: Set[Int] = vecIndexerModel.categoryMaps.keys.toSet
println(s"Chose ${categoricalFeatures.size} " +
  s"categorical features: ${categoricalFeatures.mkString(", ")}")

val sample_indexed = vecIndexerModel.transform(sample)
sample_indexed.show()

// COMMAND ----------

// MAGIC %md ### Pipelines

// COMMAND ----------

// MAGIC %md The `pipeline` class are used for combining several algorithms in one workflow. A `pipeline` can chain two kinds of operations:
// MAGIC * `transformers` which are used either for transforming features or for performing prediction (based on a trained model)  
// MAGIC * `estimators` which are used for training an ML model on the data
// MAGIC
// MAGIC The following snippet uses a pipeline for feature transformation purposes by first indexing all the attributes of the synthetic dataset, than creating a vector of features from the indexed attributes, and finally by indexing this vector to account for categorical features.

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer

/*index the label attribute*/
val labName = "buys_computer"
val stringIndexerLabel = new StringIndexer()
    .setInputCol(labName)
    .setOutputCol("label") 

// COMMAND ----------

val inFeatureAtts = data.columns.filterNot(_.contains(labName))
val outFeatureAtts = inFeatureAtts.map("indexed_"+_)
val stringIndexerFeatures = new StringIndexer().setInputCols(inFeatureAtts).setOutputCols(outFeatureAtts)


// COMMAND ----------

/*register the indexed fields*/
val indexedFields = stringIndexerFeatures.getOutputCols

// COMMAND ----------

/*create a map to register the correspondance between attribute names and their feature indice.
Eg. age will be feature 0 etc 
This will be useful to read the inferred decision tree*/
val featureIndices = indexedFields.zipWithIndex.map{case(strInd,ind)=>("feature "+ind,strInd)}.toMap

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

/*create a vector of features from all the indexed attributes 
except the target column label */
val vectorAssembler = new VectorAssembler()
                    .setInputCols(indexedFields)
                    .setOutputCol("featuresVec")

// COMMAND ----------

import org.apache.spark.ml.feature.VectorIndexer

/* index the vector of features to account for categorical features*/
val maxCat = 3
val vectorIndexer = new VectorIndexer()
  .setInputCol("featuresVec")
  .setOutputCol("features")
  .setMaxCategories(maxCat)

// COMMAND ----------

/*chain the tranformations in one pipeline*/
import  org.apache.spark.ml.Pipeline 
val pipeline = new Pipeline()
                    .setStages(Array(stringIndexerLabel,stringIndexerFeatures,vectorAssembler,vectorIndexer))


// COMMAND ----------

val train_data_model = pipeline.fit(data)

// COMMAND ----------

train_data_model.stages.foreach(println)

// COMMAND ----------

val train_data_allatts = train_data_model.transform(data)
val train_data = train_data_allatts.select("features","label")

// COMMAND ----------

train_data.show()

// COMMAND ----------

// MAGIC %md ## Second part: decision tree inference on synthetic data

// COMMAND ----------

// MAGIC %md ### Inference

// COMMAND ----------

import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val dt = new DecisionTreeClassifier()
//   .setLabelCol("label")
//   .setFeaturesCol("features")

val dtModel = dt.fit(train_data)
println(s"Learned classification tree model:\n ${dtModel.toDebugString}")

// COMMAND ----------

display(dtModel)

// COMMAND ----------

// MAGIC %md ### Interpreting the inferred tree

// COMMAND ----------

// MAGIC %md On a sheet of paper, map the feature indexes to the original columns (age, income, ...) then draw the tree.
// MAGIC Use the `featureIndices` map to recover the original feautre names.

// COMMAND ----------

// MAGIC %md ### Predicting using the inferred tree 

// COMMAND ----------

// MAGIC %md Create a set of fictious tuples adhering to the same schema of `data` then predict the label of the tuples. 
// MAGIC * age: young, senior, middle
// MAGIC * income: high, medium, low
// MAGIC * student: yes, no
// MAGIC * credit_rating: fair, excellent

// COMMAND ----------

import spark.implicits._
val test_df = Seq(("young","high","no","fair"),
                (("senior","high","yes","excellent"))
                 )
            .toDF("age","income","student","credit_rating")
test_df.show()

// COMMAND ----------

val predictionPipeline = new Pipeline().setStages(pipeline.getStages.slice(1,pipeline.getStages.size))


// COMMAND ----------

pipeline.getStages.slice(1,pipeline.getStages.size).foreach(println)

// COMMAND ----------

val test_data = predictionPipeline.fit(test_df).transform(test_df).select("features")
test_data.show()

// COMMAND ----------

val predictions = dtModel.transform(test_data)//.select("features","prediction")

// Select example rows to display.
predictions.show(false)

// COMMAND ----------

// MAGIC %md ## Third part: automation preparation
// MAGIC

// COMMAND ----------

// MAGIC %md ### Parameterizing the DT Pipeline
// MAGIC Write the body of the AutoPipeline method which takes the following arguments :
// MAGIC - the array of textual columns 
// MAGIC - the array of numeric columns
// MAGIC - the target label
// MAGIC - the maxCat parameter
// MAGIC - the handleInvalid parameter

// COMMAND ----------

def AutoPipeline(textCols: Array[String], numericCols: Array[String], target: String, maxCat: Int, handleInvalid: String):Pipeline = {
  //StringIndexer
  val inAttsNames = textCols ++ Array(target)
  val outAttsNames = inAttsNames.map(_prefix+_)

  val stringIndexer = new StringIndexer()
                              //to be completed
  
  val features = outAttsNames.filterNot(_.contains(target))++numericCols
  
  //vectorAssembler
  val vectorAssembler = new VectorAssembler()
                            //to be completed
  
  //VectorIndexer
  val vectorIndexer = new VectorIndexer()
                            //to be completed
  
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

// MAGIC %md End
