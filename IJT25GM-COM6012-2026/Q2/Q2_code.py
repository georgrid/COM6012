from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, to_date, month, weekday, when
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator

import math
import matplotlib.pyplot as plt

spark = (
    SparkSession.builder
    .master("local[10]")     # Use 10 cores 
    .appName("COM6012 Assignment Q2")      # Job name
    .config("spark.local.dir", "/mnt/parscratch/users/ijt25gm")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("\n\nQ2 Results")

# Load data
logfile = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)    
    .csv("/mnt/parscratch/users/com6012_2026/data/dft_traffic_counts_raw_counts.csv")
)







spark.stop()