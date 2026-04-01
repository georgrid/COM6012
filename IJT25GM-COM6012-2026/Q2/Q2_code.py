from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, weekday

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
logFile = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)    
    .csv("/mnt/parscratch/users/com6012_2026/data/dft_traffic_counts_raw_counts.csv")
)

# Preprocessing
# Convert count_date column into date format
logFile = logFile.withColumn("count_date", to_date(col("count_date"), "yyyy-MM-dd"))

# Create month and weekday columns
logFile = (
    logFile.withColumn("month", month(col("count_date")))
    .withColumn("weekday", weekday(col("count_date")))
)



spark.stop()