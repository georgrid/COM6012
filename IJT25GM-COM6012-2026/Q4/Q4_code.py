from pyspark.sql import SparkSession


spark = (
    SparkSession.builder
    .master("local[10]")     # Use 10 cores 
    .appName("COM6012 Assignment Q4")      # Job name
    .config("spark.local.dir", "/mnt/parscratch/users/ijt25gm")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ4 Results")

seed = 250117677   # Registration number

##### TASK A #####
print("\nTask A:")

# Load data
ratings = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)    
    .csv("/mnt/parscratch/users/com6012_2026/data/ml-20m/ratings.csv")
)