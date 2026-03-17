from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise Linear Regression") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

# Linear Regression

df = spark.read.load('./Data/Advertising.csv',  format="csv", inferSchema="true", header="true").cache()
df.show()

df2=df.drop('_c0').cache()
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])

transformed= transData(df2).cache()
(trainingData, testData) = transformed.randomSplit([0.6, 0.4])
trainingData = trainingData.cache()
testData = testData.cache()
from pyspark.ml.regression import LinearRegression

def train(model, train, test):
    lrModel = model.fit(train)
    predictions = lrModel.transform(test)
    predictions.show(5)
    from pyspark.ml.evaluation import RegressionEvaluator

    evaluator = RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"model reg param {model.getRegParam()}")
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

lr = LinearRegression()
train(lr, trainingData, testData)
lr.setRegParam(0.1)
train(lr, trainingData, testData)
lr.setRegParam(0.2)
train(lr, trainingData, testData)
lr.setRegParam(0.5)
train(lr, trainingData, testData)
