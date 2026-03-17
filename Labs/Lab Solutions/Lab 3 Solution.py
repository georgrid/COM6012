from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import json

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 3 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

#Read and prepare data
rawdata = spark.read.csv('./Data/spambase.data')
rawdata.cache()
ncolumns = len(rawdata.columns)
spam_names = [spam_names.rstrip('\n') for spam_names in open('./Data/spambase.data.names')]
number_names = np.shape(spam_names)[0]
for i in range(number_names):
    local = spam_names[i]
    colon_pos = local.find(':')
    spam_names[i] = local[:colon_pos]

# For being able to save files in a Parquet file format, later on, we need to rename
# two columns with invalid characters ; and (
spam_names[spam_names.index('char_freq_;')] = 'char_freq_semicolon'
spam_names[spam_names.index('char_freq_(')] = 'char_freq_leftp'

schemaNames = rawdata.schema.names
spam_names[ncolumns-1] = 'labels'
for i in range(ncolumns):
    rawdata = rawdata.withColumnRenamed(schemaNames[i], spam_names[i])

#Cast columns to type Double
for i in range(ncolumns):
    rawdata = rawdata.withColumn(spam_names[i], rawdata[spam_names[i]].cast(DoubleType()))

#Training/testing split
(trainingData, testData) = rawdata.randomSplit([0.7, 0.3], 42)

#Vectorize features
vecAssembler = VectorAssembler(inputCols = spam_names[0:ncolumns-1], outputCol = 'features') 

#L1 Logistic regression. Note that elasticNetParam is set to 1
#elasticNetParam controls α in regularization term: α(λ∥w∥1)+(1−α)(λ2∥w∥22)
lrL1 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=1, family="binomial")

# Pipeline for the model with L1 regularisation
stageslrL1 = [vecAssembler, lrL1]
pipelinelrL1 = Pipeline(stages=stageslrL1)
pipelineModellrL1 = pipelinelrL1.fit(trainingData)

predictions = pipelineModellrL1.transform(testData)
# With Predictions
evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("L1 Accuracy = %g " % accuracy)

#w_L1 now holds the weights for each feature
w_L1 = pipelineModellrL1.stages[-1].coefficients.values

#np.argsort(np.abs(w_L1)) will return the indices if you were to sort by the absolute of the weights
#(remember we care about the magnitude of the weights, not the sign/direction)
#For example if we called np.argsort(np.array([10, 5, 15])), it will return [1, 0, 2]
#We then use those index span_names with those indices to get back the feature names
#and join them together with '\n's into a single string for printing
L1_features = "\n".join([spam_names[i] for i in np.argsort(np.abs(w_L1))[::-1][:5]])

print(f'5 most relevant features L1 features: \n{L1_features}')

#L2 Logistic regression. Note that elasticNetParam is set to 0
#elasticNetParam controls α in regularization term: α(λ∥w∥1)+(1−α)(λ2∥w∥22)
lrL2 = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=0, family="binomial")


# Pipeline for the model with L2 regularisation
stageslrL2 = [vecAssembler, lrL2]
pipelinelrL2 = Pipeline(stages=stageslrL2)
pipelineModellrL2 = pipelinelrL2.fit(trainingData)

predictions = pipelineModellrL2.transform(testData)
# With Predictions

accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

#w_L2 now holds the weights for each feature
w_L2 = pipelineModellrL2.stages[-1].coefficients.values

#np.argsort(np.abs(w_L2)) will return the indices if you were to sort by the absolute of the weights
#(remember we care about the magnitude of the weights, not the sign/direction)
#For example if we called np.argsort(np.array([10, 5, 15])), it will return [1, 0, 2]
#We then use those index span_names with those indices to get back the feature names
#and join them together with '\n's into a single string for printing
L2_features = "\n".join([spam_names[i] for i in np.argsort(np.abs(w_L2))[::-1][:5]])

print(f'5 most relevant features L2 features: \n{L2_features}')

#Elastic Net Logistic regression. Note that elasticNetParam is set to a number between 0 and 1
#elasticNetParam controls α in regularization term: α(λ∥w∥1)+(1−α)(λ2∥w∥22)
lrEN = LogisticRegression(featuresCol='features', labelCol='labels', maxIter=50, regParam=0.01, \
                          elasticNetParam=0.5, family="binomial")


# Pipeline for the model with EN regularisation
stageslrEN = [vecAssembler, lrEN]
pipelinelrEN = Pipeline(stages=stageslrEN)
pipelineModellrEN = pipelinelrEN.fit(trainingData)

predictions = pipelineModellrEN.transform(testData)
# With Predictions

accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % accuracy)

#w_EN now holds the weights for each feature
w_EN = pipelineModellrEN.stages[-1].coefficients.values

#np.argsort(np.abs(w_EN)) will return the indices if you were to sort by the absolute of the weights
#(remember we care about the magnitude of the weights, not the sign/direction)
#For example if we called np.argsort(np.array([10, 5, 15])), it will return [1, 0, 2]
#We then use those index span_names with those indices to get back the feature names
#and join them together with '\n's into a single string for printing
EN_features = "\n".join([spam_names[i] for i in np.argsort(np.abs(w_EN))[::-1][:5]])

print(f'5 most relevant features Elastic Net features: \n{EN_features}')

lr = LogisticRegression(featuresCol='features', labelCol='labels', family='binomial')

stageslr = [vecAssembler, lr]
pipelinelr = Pipeline(stages=stageslr)


#Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
#FUN FACT: replacing 0.0 and 1.0 with 0 and 1 will return a Java cast conversion error
#This is because elasticNetParam requires a certain type (float in this case)
paramGrid = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0.0, 0.2, 0.5, 0.7, 1.0]) \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
    .addGrid(lr.maxIter, [25, 50, 100]) \
    .build()


evaluator = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

# Make Crossvalidator object
crossval = CrossValidator(estimator=pipelinelr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel = crossval.fit(trainingData)
prediction = cvModel.transform(testData)
accuracy = evaluator.evaluate(prediction)

print("Accuracy for best lm model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict = {param[0].name: param[1] for param in cvModel.bestModel.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict, indent = 4))
