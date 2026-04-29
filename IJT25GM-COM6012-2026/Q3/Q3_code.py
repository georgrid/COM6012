from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType

spark = (
    SparkSession.builder
    .master("local[10]")     # Use 10 cores 
    .appName("COM6012 Assignment Q3")      # Job name
    .config("spark.local.dir", "/mnt/parscratch/users/ijt25gm")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ3 Results")


##### TASK A #####
print("\nTask A:")

# Manually define schema to improve speed
schema = StructType(
    [StructField("label", DoubleType(), True)] +                               # Column 0 = label
    [StructField(f"feature_{i}", DoubleType(), True) for i in range(1, 29)]    # Columns 1-28 = features
)

# Load data
logFile = spark.read \
    .option("header", False) \
    .schema(schema) \
    .csv("/users/ijt25gm/com6012/ScalableML/Data/HIGGS.csv")

logFile = logFile.cache()
logFile.count()

logFile.show(10, False)


spark.stop()