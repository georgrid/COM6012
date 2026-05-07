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
    "/users/ijt25gm/com6012/ScalableML/Data/ml-20m/ratings.csv",
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

    print(f"Training fraction: {frac}")
    print(f"  Training size: {train.count()}")
    print(f"  Test size: {test.count()}")

print("")

# Define evaluators
rmse_evaluator = RegressionEvaluator(
    metricName='rmse',
    labelCol='rating',
    predictionCol='prediction'
)
mse_evaluator = RegressionEvaluator(
    metricName='mse',
    labelCol='rating',
    predictionCol='prediction'
)
mae_evaluator = RegressionEvaluator(
    metricName='mae',
    labelCol='rating',
    predictionCol='prediction'
)

als = ALS(
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    seed=seed,
    coldStartStrategy='drop'   # Ensures there are no NaN evaluation metrics
)

results = []

# Loop over train/test splits
print("\nALS setting 1:")
for frac, (train, test) in splits.items():
    
    model = als.fit(train)
    predictions = model.transform(test)

    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    results.append((frac, 'setting 1', rmse, mse, mae))

    print(f"{int(frac * 100)}% training split")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")


# Setting 2
als2 = ALS(
    userCol='userId',
    itemCol='movieId',
    ratingCol='rating',
    seed=seed,
    coldStartStrategy='drop',
    rank=25,
    regParam=0.15,
    maxIter=10
)
models_setting_2 = {}

print("\nALS setting 2:")
for frac, (train, test) in splits.items():

    model = als2.fit(train)
    predictions = model.transform(test)

    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    results.append((frac, 'setting 2', rmse, mse, mae))
    models_setting_2[frac] = model

    print(f"{int(frac * 100)}% training split")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")



spark.stop()