from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# Initialise spark session
spark = SparkSession.builder.appName("Q1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read in data and split words (tab separated)
lines = spark.read.text("/users/ijt25gm/com6012/ScalableML/Data/ml-latest-small/////").rdd
parts = lines.map(lambda row: row.value.split("\t"))

# Convert the text (str) into numbers (int or float),
# then convert RDD to DataFrame
ratingsRdd = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRdd).cache()
