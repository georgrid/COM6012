from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import explode, col
from pyspark.sql import Row
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Initialise spark session
spark = SparkSession.builder.appName("Q1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

##### TASK 1 #####
# Read in data
movies = spark.read.csv("/users/ijt25gm/com6012/ScalableML/Data/ml-latest-small/movies.csv", header=True, inferSchema=True)
ratings = spark.read.csv("/users/ijt25gm/com6012/ScalableML/Data/ml-latest-small/ratings.csv", header=True, inferSchema=True)


##### TASK 2 #####
print("\n========== TASK2 ==========\n")
# Prepare the training / test data
myseed = 6012
training, test = ratings.randomSplit([0.8, 0.2], myseed)

training = training.cache()
test = test.cache()

# Initialise rank parameter values
ranks = [5, 10, 15, 20, 25]
rmse_values = []

for rank in ranks:
    # Build the recommendation model using ALS on the training data, for the different rank parameters
    # Set the cold start strategy to `drop` to ensure we don't get NaN evaluation metrics
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        seed=myseed,
        coldStartStrategy="drop"
    )

    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Rank = {rank} | RMSE = {str(rmse)}")

    rmse_values.append(rmse)

# Plot the resulting RMSE values against the rank values
plt.figure(figsize=(6, 4))
plt.plot(ranks, rmse_values, marker='o')
plt.xlabel("Rank")
plt.ylabel("RMSE")
plt.title("RMSE vs Rank for ALS on MovieLens")
plt.tight_layout()
plt.savefig("/users/ijt25gm/com6012/ScalableML/Code/lab6/rmse_vs_rank.png", dpi=300)

print("\nSaved figure: rmse_vs_rank.png\n\n")


##### TASK 3 #####
print("========== TASK 3 ==========\n")
# Choose best performing rank
best_rank = 25

# Train final model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=best_rank,
    seed=myseed,
    coldStartStrategy="drop"
)
model = als.fit(training)

# Choose one user
chosen_user_id = 1

# Make DataFrame containing only that user
one_user = spark.createDataFrame([(chosen_user_id,)], ["userId"])

# Get top 5 recommendations for that user
user_recs = model.recommendForUserSubset(one_user, 5)

# Expand the nested recommendations column
top5 = user_recs.select(
    col("userId"),
    explode(col("recommendations")).alias("rec")
).select(
    col("userId"),
    col("rec.movieId").alias("movieId"),
    col("rec.rating").alias("predictedRating")
)

# Join with movies to get titles and genres
top5_with_info = top5.join(
    movies,
    on="movieId",
    how="inner"
).select(
    "userId",
    "movieId",
    "title",
    "genres",
    "predictedRating"
)
top5_with_info.show(truncate=False)

spark.stop()
