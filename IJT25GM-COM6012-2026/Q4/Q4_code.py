from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

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
ratings = spark.read.load(
    "/mnt/parscratch/users/com6012_2026/data/ml-20m/ratings.csv",
    format='csv',
    inferSchema="true",
    header="true"
).cache()

print("Ratings schema:")
ratings.printSchema()

print("First few ratings:")
ratings.show(5, False)

n_ratings = ratings.count()
print(f"Total number of ratings: {n_ratings}")

# Sort data by timestamp
window = Window.orderBy('timestamp')

ratings_sorted = ratings \
    .orderBy('timestamp') \
    .withColumn('row_num', row_number().over(window)) \
    .cache()

# Define training split sizes
train_fractions = [0.4, 0.6, 0.8]

splits = {}

for frac in train_fractions:
    split_index = int(n_ratings * frac)

    train = ratings_sorted \
        .filter(col('row_num') <= split_index) \
        .drop('row_num') \
        .cache()
    
    test = ratings_sorted \
        .filter(col('row_num') > split_index) \
        .drop('row_num') \
        .cache()
    
    splits[frac] = (train, test)

    print(f"\nTraining fraction: {frac}")
    print(f"  Training size: {train.count()}")
    print(f"  Test size: {test.count()}")

spark.stop()