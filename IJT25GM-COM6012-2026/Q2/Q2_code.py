from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, weekday, upper, trim, when
from pyspark.sql.types import DoubleType

from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StandardScaler, StringIndexer

spark = (
    SparkSession.builder
    .master("local[10]")     # Use 10 cores 
    .appName("COM6012 Assignment Q2")      # Job name
    .config("spark.local.dir", "/mnt/parscratch/users/ijt25gm")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ2 Results")


##### TASK A #####
print("\nTask A:")

# Load data
logFile = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)    
    .csv("/mnt/parscratch/users/com6012_2026/data/dft_traffic_counts_raw_counts.csv")
)

# Convert count_date column into date format
logFile = logFile.withColumn("count_date", to_date(col("count_date"), "yyyy-MM-dd"))

# Create month and weekday columns
logFile = (
    logFile.withColumn("month", month(col("count_date")))
    .withColumn("weekday", weekday(col("count_date")))
)

# Convert values in direction_of_travel column to uppercase
logFile = logFile.withColumn("direction_of_travel", upper(trim(col("direction_of_travel"))))

# Drop rows where all_motor_vehicles is NULL or "NULL"
logFile = logFile.replace("NULL", None)
logFile = logFile.na.drop(subset=['all_motor_vehicles'])

# Select only required columns
categorical_cols = [
    "direction_of_travel",
    "hour",
    "region_ons_code",
    "local_authority_code",
    "month",
    "weekday"
]
numeric_cols = ['latitude', 'longitude']
selected_cols = ["year"] + categorical_cols + numeric_cols + ["all_motor_vehicles"]
data = logFile.select(*selected_cols)

# Ensure that numerical features are of type double
for c in numeric_cols + ["all_motor_vehicles"]:
    data = data.withColumn(c, col(c).cast(DoubleType()))

# Assemble numerical features into a vector
num_assembler = VectorAssembler(
    inputCols=numeric_cols,
    outputCol="numeric_features"
)
data = num_assembler.transform(data)

# Standardise numerical features using StandardScaler
scaler = StandardScaler(
    inputCol="numeric_features",
    outputCol="scaled_numeric_features",
    withMean=True,
    withStd=True
)
data = scaler.fit(data).transform(data)

# Index categorical columns
indexed_data = data
for c in categorical_cols:
    indexer = StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    indexed_data = indexer.fit(indexed_data).transform(indexed_data)

# Apply one-hot encoder to indexed columns
ohe = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_ohe" for c in categorical_cols]
)
ohe_model = ohe.fit(indexed_data)
ohe_data = ohe_model.transform(indexed_data)

# Combine one-hot encoded categorical features and standardised numerical features
feature_cols = [f"{c}_ohe" for c in categorical_cols] + ["scaled_numeric_features"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
final_data = assembler.transform(ohe_data)

# Split data by year into training, validation and testing sets
train_data = final_data.filter((col("year") >= 2000) & (col("year") <= 2021))
val_data = final_data.filter((col("year") >= 2022) & (col("year") <= 2023))
test_data = final_data.filter(col("year") == 2024)

train_data = train_data.select("features", "all_motor_vehicles")
val_data = val_data.select("features", "all_motor_vehicles")
test_data = test_data.select("features", "all_motor_vehicles")

print(f"Training size: {train_data.count()}")
print(f"Validation size: {val_data.count()}")
print(f"Test size: {test_data.count()}")


spark.stop()