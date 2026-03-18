from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
import json


spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Lab 5 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")



#Load dataset and preprocessing
rawdata = spark.read.csv('../Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('../Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]


schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])


StringColumns = [x.name for x in rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))


trainingData, testData = rawdata.randomSplit([0.7, 0.3], 42)

vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features') 
vecTrainingData = vecAssembler.transform(trainingData)
vecTrainingData.select("features", "labels").show(5)

# Combine stages into pipeline
dt = DecisionTreeClassifier(labelCol="labels", featuresCol="features")
stages = [vecAssembler, dt]
pipeline = Pipeline(stages=stages)

# Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [1, 10, 30]) \
    .addGrid(dt.maxBins, [10, 100, 1000]) \
    .addGrid(dt.impurity, ['entropy', 'gini']) \
    .build()

# Evaluator for both the crossvalidation, and later for checking accuracy
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

# Make Crossvalidator object
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel = crossval.fit(trainingData)
prediction = cvModel.transform(testData)
accuracy = evaluator.evaluate(prediction)

print("Accuracy for best dt model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict = {param[0].name: param[1] for param in cvModel.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict, indent = 4))

# Combine stages into pipeline
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=42)
stages = [vecAssembler, rf]
pipeline = Pipeline(stages=stages)

# Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [1, 5, 10]) \
    .addGrid(rf.maxBins, [2, 10, 20]) \
    .addGrid(rf.numTrees, [1, 5, 10]) \
    .addGrid(rf.featureSubsetStrategy, ['all','sqrt', 'log2']) \
    .addGrid(rf.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()

# Make Crossvalidator object, we use the same evaluator as for the previous exercise
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel = crossval.fit(trainingData)
prediction = cvModel.transform(testData)
accuracy = evaluator.evaluate(prediction)

print("Accuracy for best rf model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict = {param[0].name: param[1] for param in cvModel.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict, indent = 4))


#Load dataset and preprocessing
rawdataw = spark.read.csv('../Data/winequality-white.csv', sep=';', header='true')
rawdataw.cache()

StringColumns = [x.name for x in rawdataw.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    rawdataw = rawdataw.withColumn(c, col(c).cast("double"))
rawdataw = rawdataw.withColumnRenamed('quality', 'labels')    

trainingDataw, testDataw = rawdataw.randomSplit([0.7, 0.3], 42)

vecAssemblerw = VectorAssembler(inputCols=StringColumns[:-1], outputCol="features")


rf = RandomForestRegressor(labelCol="labels", featuresCol="features", maxDepth=5, numTrees=3, seed=42)

stages = [vecAssemblerw, rf]
pipeline = Pipeline(stages=stages)

#Fit model
pipelineModel = pipeline.fit(trainingDataw)

#Extract Feature importances
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# feature importance calculated by averaging the decrease in impurity over trees
featureImp = pd.DataFrame(
  list(zip(vecAssemblerw.getInputCols(), pipelineModel.stages[-1].featureImportances)),
  columns=["feature", "importance"])
print(featureImp.sort_values(by="importance", ascending=False).to_string())
