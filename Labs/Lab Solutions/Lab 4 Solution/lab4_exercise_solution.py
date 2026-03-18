from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 4 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

# Read in the hourly bike sharing data
rawdata = spark.read.csv('./Data/hour.csv', header=True)
rawdata.cache()

# As in Part 1 of the lab, select the requried columns and 
# convert to DoubleType
schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
new_rawdata = rawdata.select(schemaNames[2:ncolumns])

new_schemaNames = new_rawdata.schema.names
from pyspark.sql.types import DoubleType
new_ncolumns = len(new_rawdata.columns)
for i in range(new_ncolumns):
    new_rawdata = new_rawdata.withColumn(new_schemaNames[i], new_rawdata[new_schemaNames[i]].cast(DoubleType()))
print('\nORIGINAL SCHEMA\n')
new_rawdata.printSchema()

# ---- EXERCISE 1 ----
print('---- EXERCISE 1 ----')

# Now we one hot encode the required categorical variables, and drop the original non-encoded
# columns from our dataset as they are no longer required
ohe = OneHotEncoder(inputCols=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'],
                    outputCols=['season_ohe', 'yr_ohe', 'mnth_ohe', 'hr_ohe', 'holiday_ohe', 'weekday_ohe', 'workingday_ohe', 'weathersit_ohe'])
ohe_model = ohe.fit(new_rawdata)
ohe_rawdata = ohe_model.transform(new_rawdata).drop(*['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit'])
print('\nOHE SCHEMA\n')
ohe_rawdata.printSchema()

# Now, we select the feature we wish to use: temp, atemp, hum and windspeed, alongside the OHE versions of
# the season, yr, mnth, hr, holiday, weekday, workingday and weathersit features
feature_cols = ohe_rawdata.schema.names[:4] + ohe_rawdata.schema.names[7:]
assembler = VectorAssembler(inputCols = feature_cols, outputCol = 'features') 

# Split the data into train and test sets
(trainingData, testData) = ohe_rawdata.randomSplit([0.7, 0.3], 42)

# Define the Poisson model
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')

# Construct and fit pipeline to data
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)

# Evaluate the model RMSE on the test set
predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("\nRMSE = %g \n" % rmse)

print('\n Model Coefficients')
print(pipelineModel.stages[-1].coefficients)

# ---- EXERCISE 2 ----
print('\n---- EXERCISE 2 ----\n')

# Define the four different models
# Both 'normal' and 'l-bfgs' solver will enable OWL-QN for l1 or elastic-net regularization.
l1_qn = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,
                          elasticNetParam=1, solver='normal')
en_qn = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,
                          elasticNetParam=0.5, solver='normal')
l2_lbgfs = LinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,
                            elasticNetParam=0, solver='l-bfgs')
l2_irls = GeneralizedLinearRegression(featuresCol='features', labelCol='cnt', maxIter=50, regParam=0.01,\
                                      family='gaussian', link='identity', solver='irls')

# Fit the four model pipelines, make predictions on the test data, and evaluate the RMSE for each
l1_qn_model = Pipeline(stages=[assembler, l1_qn]).fit(trainingData)
en_qn_model = Pipeline(stages=[assembler, en_qn]).fit(trainingData)
l2_lbgfs_model = Pipeline(stages=[assembler, l2_lbgfs]).fit(trainingData)
l2_irls_model = Pipeline(stages=[assembler, l2_irls]).fit(trainingData)

l1_qn_preds = l1_qn_model.transform(testData)
en_qn_preds = en_qn_model.transform(testData)
l2_lbgfs_preds = l2_lbgfs_model.transform(testData)
l2_irls_preds = l2_irls_model.transform(testData)

rmse_l1_qn = evaluator.evaluate(l1_qn_preds)
rmse_en_qn = evaluator.evaluate(en_qn_preds)
rmse_l2_lbgfs = evaluator.evaluate(l2_lbgfs_preds)
rmse_l2_irls = evaluator.evaluate(l2_irls_preds)

print("L1 (OWL-QN optimisation) RMSE = %g " % rmse_l1_qn)
print("ElasticNet (0.5 w/ OWL-QN optimisation) RMSE = %g " % rmse_en_qn)
print("L2 (LBGFS optimisation) RMSE = %g " % rmse_l2_lbgfs)
print("L2 (IRLS optimisation) RMSE = %g " % rmse_l2_irls)

spark.stop()