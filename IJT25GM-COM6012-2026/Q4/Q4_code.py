from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import desc, avg, explode, split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

spark = (
    SparkSession.builder
    .master('local[10]')     # Use 10 cores 
    .appName('COM6012 Assignment Q4')      # Job name
    .config('spark.local.dir', '/mnt/parscratch/users/ijt25gm')
    .getOrCreate()
)
spark.sparkContext.setLogLevel('ERROR')
print("\n\nQ4 Results")

seed = 250117677   # Registration number


##############################################################
########## TASK A
print("\nTask A:")

# Load data
ratings = spark.read.load(
    '/users/ijt25gm/com6012/ScalableML/Data/ml-20m/ratings.csv',
    format='csv',
    inferSchema='true',
    header='true'
).cache()

n_ratings = ratings.count()
print(f"Total number of ratings: {n_ratings}\n")

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

# Convert results to dataframe
results_df = pd.DataFrame(
    results,
    columns=['train_fraction', 'als_setting', 'RMSE', 'MSE', 'MAE']
)

# Create training split column
results_df['training_split'] = (results_df['train_fraction'] * 100).astype(int).astype(str) + '%'

# Keep rows in order
results_df['als_setting'] = pd.Categorical(
    results_df['als_setting'],
    categories=['setting 1', 'setting 2'],
    ordered=True
)
results_df['training_split'] = pd.Categorical(
    results_df['training_split'],
    categories=['40%', '60%', '80%'],
    ordered=True
)

results_df = results_df.sort_values(['als_setting', 'training_split'])

fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

metrics = ['RMSE', 'MSE', 'MAE']
settings = ['setting 1', 'setting 2']

for ax, setting in zip(axes, settings):
    setting_data = results_df[results_df['als_setting'] == setting]

    for metric in metrics:
        ax.plot(
            setting_data['training_split'],
            setting_data[metric],
            marker='o',
            ms=5,
            label=metric
        )
    
    ax.set_title(f"ALS {setting.capitalize()}")
    ax.grid(axis='y', alpha=0.2)"
axes[0].legend(metrics, ncol=3, fontsize=8)
axes[0].set_ylabel("Error", fontsize=12)
fig.supxlabel("Training Split", fontsize=12)

plt.tight_layout()
plt.savefig('Q4_fig1.png', dpi=300)
plt.close()


##############################################################
########## TASK B
print("\nTask B:")

# Convert ALS attributes into dense feature vectors
def transData(data):
    return data.rdd.map(
        lambda r: (r.id, Vectors.dense(r.features))
    ).toDF(['userId', 'features'])

cluster_results = []
largest_cluster_users = {}

for frac, model in models_setting_2.items():

    dfFeatureVec = transData(model.userFactors).cache()

    # Fit k-means with k=25
    kmeans = KMeans().setK(25).setSeed(seed)
    kmeans_model = kmeans.fit(dfFeatureVec)

    # Generate cluster predictions
    predictions = kmeans_model.transform(dfFeatureVec).cache()

    # Keep top 5 largest clusters
    top_clusters = predictions \
        .groupBy('prediction') \
        .count() \
        .orderBy(desc('count')) \
        .limit(5) \
        .collect()

    print(f"\nTop 5 largest clusters ({int(frac * 100)}% training split):")
    for rank, row in enumerate(top_clusters, start=1):

        cluster_id = row['prediction']
        cluster_size = row['count']

        cluster_results.append((frac, rank, cluster_id, cluster_size))
        
        print(f"  Rank {rank}: cluster {cluster_id}, size = {cluster_size}")

    # Save users in largest cluster
    largest_cluster_id = top_clusters[0]['prediction']

    largest_cluster_users[frac] = predictions \
        .filter(predictions.prediction == largest_cluster_id) \
        .select('userId') \
        .cache()
    
# Convert cluster results into dataframe for plotting
cluster_df = pd.DataFrame(
    cluster_results,
    columns=['train_fraction', 'rank', 'cluster_id', 'cluster_size']
)

# Convert splits into labels
cluster_df['training_split'] = (
    (cluster_df['train_fraction'] * 100).astype(int).astype(str) + '%'
)

cluster_df = cluster_df.sort_values(['train_fraction', 'rank'])

plt.figure(figsize=(7, 4))

for training_split in ['40%', '60%', '80%']:
    split_data = cluster_df[cluster_df['training_split'] == training_split]
    plt.plot(
        split_data['rank'],
        split_data['cluster_size'],
        marker='o',
        ms=5,
        label=training_split
    )

plt.xticks([1, 2, 3, 4, 5])
plt.xlabel("Cluster Rank", fontsize=12)
plt.ylabel("Cluster Size", fontsize=12)
plt.legend(title="Training Split")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig('Q4_fig2.png', dpi=300)
plt.close()

# Load movies data for genres
movies = spark.read.load(
    '/users/ijt25gm/com6012/ScalableML/Data/ml-20m/movies.csv',
    format='csv',
    inferSchema='true',
    header='true'
).cache()

genre_results = []

print("\nTop 10 genres from top movies in largest user cluster:")

for frac, (train, test) in splits.items():

    # Get users in the largest cluster for each split
    largest_cluster_users_split = largest_cluster_users[frac]

    # Use training set only
    ratings_largest_cluster = train \
    .join(largest_cluster_users_split, on='userId', how='inner') \
    .cache()

    # Find average rating for each movie among users in largest cluster
    movies_largest_cluster = ratings_largest_cluster \
        .groupBy('movieId') \
        .agg(avg('rating').alias('avg_rating')) \
        .cache()
    
    # Keep movies with avg rating >= 4
    top_movies = movies_largest_cluster \
        .filter(col('avg_rating') >= 4.0) \
        .cache()
    
    # Get genres
    top_movies_with_genres = top_movies \
        .join(movies, on='movieId', how='inner') \
        .select('movieId', 'avg_rating', 'title', 'genres') \
        .cache()
    
    # Split genres and count
    top_genres = top_movies_with_genres \
        .select(explode(split(col('genres'), '\\|')).alias('genre')) \
        .groupBy('genre') \
        .count() \
        .orderBy(desc('count')) \
        .limit(10) \
        .collect()

    print(f"\nTop 10 genres ({int(frac * 100)}% training split):")

    for rank, row in enumerate(top_genres, start=1):

        genre = row['genre']
        count = row['count']

        genre_results.append((frac, rank, genre, count))

        print(f"  Rank {rank}: {genre}, count = {count}")

spark.stop()