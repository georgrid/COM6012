from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, month, weekday, upper, trim, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
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


##### TASK B #####
print("\n\nTask B:")

# Cache datasets
train_data = train_data.cache()
val_data = val_data.cache()
test_data = test_data.cache()

train_data.count()
val_data.count()
test_data.count()

# Candidate regParam values
reg_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Create seeds using student registration number (250117677)
seeds = [17677, 17678, 17679, 17680, 17681]

# Use MSE on validation and test sets
evaluator = RegressionEvaluator(
    labelCol="all_motor_vehicles",
    predictionCol="prediction",
    metricName="mse"
)

results = []

for reg_param in reg_params:
    mse_values = []

    print(f"regParam = {reg_param}")

    for seed in seeds:
        # Sample 80% of the training set
        train_sample = train_data.sample(
            withReplacement=False,
            fraction=0.8,           # Sample 80% of training set
            seed=seed
        )

        # Define Poisson model
        glm_poisson = GeneralizedLinearRegression(
            featuresCol="features",
            labelCol="all_motor_vehicles",
            maxIter=50,
            regParam=reg_param,
            family="poisson",
            link="log"
        )

        # Fit the model
        model = glm_poisson.fit(train_sample)

        # Create predictions
        predictions = model.transform(val_data)

        # Calculate validation MSE
        mse = evaluator.evaluate(predictions)
        mse_values.append(mse)

    # Mean and std of MSE values
    mean_mse = sum(mse_values) / len(mse_values)
    std_mse = math.sqrt(sum((x - mean_mse)**2 for x in mse_values) / len(mse_values))

    results.append((reg_param, mean_mse, std_mse))

    print(f"  Mean validation MSE = {mean_mse:.2f}")
    print(f"  Standard deviation = {std_mse:.2f}\n")

reg_param_values = [x[0] for x in results]
mean_mse_values = [x[1] for x in results]
std_mse_values = [x[2] for x in results]

# Plot mean validation MSE against regParam
plt.figure(figsize=(12,6))
plt.errorbar(reg_param_values, mean_mse_values, yerr=std_mse_values, fmt='o-', capsize=5)

plt.xscale("log")
plt.xlabel("regParam")
plt.ylabel("Mean validation MSE")
plt.title("Mean Validation MSE vs regParam")
plt.tight_layout()
plt.savefig("Q2_fig1.png")
plt.close()

# Select best regParam value for evaluation on the test set
best_result = min(results, key=lambda x: x[1])
best_reg_param = best_result[0]
print(f"Optimal regParam value: {best_reg_param}")

# Combine training and validation sets
train_val_data = train_data.union(val_data)
train_val_data = train_val_data.cache()
train_val_data.count()

# Retrain model using best regParam
glm_poisson_tuned = GeneralizedLinearRegression(
    featuresCol='features',
    labelCol='all_motor_vehicles',
    maxIter=50,
    regParam=best_reg_param,
    family='poisson',
    link='log'
)
model_tuned = glm_poisson_tuned.fit(train_val_data)

# Evaluate model on test data
predictions = model_tuned.transform(test_data)
test_mse = evaluator.evaluate(predictions)
print(f"Final test MSE = {test_mse:.2f}")

# Get learned model coefficients
coefficients = model_tuned.coefficients.toArray()
print("\nFinal model coefficients:")
for i in range (0, len(coefficients), 5):
    print(coefficients[i:i+5])


##### TASK C #####
print("\n\nTask C:")

# Calculate median value on training set only
median_motor_vehicles = train_data.approxQuantile(
    'all_motor_vehicles',
    [0.5],   # 0.5: median
    0.01     # small relative error: more expensive but more accurate
)[0]
print(f"Median value of `all_motor_vehicles`: {median_motor_vehicles}")

# Add new binary target column `traffic` to training, validation and test data
train_data = train_data.withColumn(
    'traffic',
    when(col('all_motor_vehicles') <= median_motor_vehicles, 0).otherwise(1)
)
val_data = val_data.withColumn(
    'traffic',
    when(col('all_motor_vehicles') <= median_motor_vehicles, 0).otherwise(1)
)
test_data = test_data.withColumn(
    'traffic',
    when(col('all_motor_vehicles') <= median_motor_vehicles, 0).otherwise(1)
)

# Prepare data for logistic regression
train_data_lr = train_data.select("features", "traffic")
val_data_lr = val_data.select("features", "traffic")
test_data_lr = test_data.select("features", "traffic")

# Calculate mean validation accuracy
evaluator = MulticlassClassificationEvaluator(
    labelCol="traffic",
    predictionCol="prediction",
    metricName="accuracy"
)

# Candidate regParam and elasticNetParam values
reg_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
elastic_net_params = [0.0, 0.2, 0.5, 0.8, 1.0]

# Same as task B but looping over two parameters
results = []

for reg_param in reg_params:
    for elastic_net_param in elastic_net_params:

        accuracy_values = []

        print(f"regParam = {reg_param}, elasticNetParam = {elastic_net_param}")

        for seed in seeds:
            # Sample 80% of the training set
            train_sample = train_data_lr.sample(
                withReplacement=False,
                fraction=0.8,           # Sample 80% of training set
                seed=seed
            )

            # Define logistic regression model
            lr = LogisticRegression(
                featuresCol="features",
                labelCol="traffic",
                maxIter=50,
                regParam=reg_param,
                elasticNetParam=elastic_net_param
            )
            model = lr.fit(train_sample)

            # Create predictions and calculate validation accuracy
            predictions = model.transform(val_data_lr)
            accuracy = evaluator.evaluate(predictions)
            accuracy_values.append(accuracy)

        # Mean and std of accuracy values
        mean_accuracy = sum(accuracy_values) / len(accuracy_values)
        std_accuracy = math.sqrt(sum((x - mean_accuracy)**2 for x in accuracy_values) / len(accuracy_values))

        results.append((reg_param, elastic_net_param, mean_accuracy, std_accuracy))

        print(f"  Mean validation accuracy = {mean_accuracy:.4f}")
        print(f"  Standard deviation = {std_accuracy:.4f}\n")

# Plot mean validation accuracy for all combinations of regParam and elasticNetParam
plt.figure(figsize=(12, 6))

for elastic_net_param in elastic_net_params:
    subset = [x for x in results if x[1] == elastic_net_param]

    reg_param_values = [x[0] for x in subset]
    mean_accuracy_values = [x[2] for x in subset]
    std_accuracy_values = [x[3] for x in subset]

    plt.errorbar(
        reg_param_values,
        mean_accuracy_values,
        yerr=std_accuracy_values,
        fmt='o-',
        capsize=5,
        label=f"elasticNetParam = {elastic_net_param}"
    )

plt.xscale("log")
plt.xlabel("regParam")
plt.ylabel("Mean Validation Accuracy")
plt.title("Mean Validation Accuracy vs regParam")
plt.legend()
plt.tight_layout()
plt.savefig("Q2_fig2.png")
plt.close()

# Select best hyperparameter combination
best_result = max(results, key=lambda x: x[2])
best_reg_param = best_result[0]
best_elastic_net_param = best_result[1]

print(f"Optimal regParam value: {best_reg_param}")
print(f"Optimal elasticNetParam value: {best_elastic_net_param}")

# Combine training and validation sets
train_val_data_lr = train_data.union(val_data_lr)
train_val_data_lr = train_val_data_lr.cache()
train_val_data_lr.count()

# Retrain model using best hyperparameters
lr_tuned = LogisticRegression(
    featuresCol="features",
    labelCol="traffic",
    maxIter=50,
    regParam=best_reg_param,
    elasticNetParam=best_elastic_net_param
)
model_tuned = lr_tuned.fit(train_val_data_lr)

# Evaluate model on test data
predictions = model_tuned.transform(test_data_lr)
test_accuracy = evaluator.evaluate(predictions)
print(f"\nFinal test accuracy = {test_mse:.4f}")

# Get learned model coefficients
coefficients = model_tuned.coefficients.toArray()
print("\nFinal model coefficients:")
for i in range (0, len(coefficients), 5):
    print(coefficients[i:i+5])

spark.stop()