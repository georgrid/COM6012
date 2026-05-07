from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json

spark = (
    SparkSession.builder
    .master("local[10]")     # Use 10 cores 
    .appName("COM6012 Assignment Q3")      # Job name
    .config("spark.local.dir", "/mnt/parscratch/users/ijt25gm")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ3 Results")

seed = 250117677   # Registration number

##### TASK A #####
print("\nTask A:")

# Manually define schema to improve speed
schema = StructType(
    [StructField("label", DoubleType(), True)] +                               # Column 0 = label
    [StructField(f"feature_{i}", DoubleType(), True) for i in range(1, 29)]    # Columns 1-28 = features
)

# Load data
higgs_df = spark.read \
    .option("header", False) \
    .schema(schema) \
    .csv("/users/ijt25gm/com6012/ScalableML/Data/HIGGS.csv")

# Stratified sampling: sample 2% of each class
fractions = {0.0: 0.02, 1.0: 0.02}
sampled_df = higgs_df.sampleBy("label", fractions, seed=seed)

sampled_df = sampled_df.cache()
sampled_df.count()

# Create feature vector
feature_cols = [f"feature_{i}" for i in range(1, 29)]

vecAssembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features'
)


# Random Forest cross-validation
rf = RandomForestClassifier(
    labelCol='label',
    featuresCol='features',
    seed=seed
)

# Combine stages into pipeline
rf_pipeline = Pipeline(stages=[vecAssembler, rf])

# Create parameter grid for cross-validation
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [3, 5, 7]) \
    .addGrid(rf.maxBins, [16, 32, 64]) \
    .addGrid(rf.numTrees, [10, 20, 40]) \
    .build()

# Accuracy evaluator
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

# Make crossvalidator object
rf_crossval = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_paramGrid,
    evaluator=evaluator,
    numFolds=3
)

rf_cvModel = rf_crossval.fit(sampled_df)

# Find best parameters
rf_best_pipeline = rf_cvModel.bestModel
best_rf = rf_best_pipeline.stages[-1]

rf_param_dict = {
    param[0].name: param[1]
    for param in best_rf.extractParamMap().items()
}

print("Best Random Forest Parameters:")
print(json.dumps(rf_param_dict, indent=4))


# Repeat process for Gradient Boosted Tree cross-validation
gbt = GBTClassifier(
    labelCol='label',
    featuresCol='features',
    seed=seed
)

gbt_pipeline = Pipeline(stages=[vecAssembler, gbt])

gbt_paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5, 7]) \
    .addGrid(gbt.maxBins, [16, 32, 64]) \
    .addGrid(gbt.maxIter, [10, 20, 40]) \
    .build()

gbt_crossval = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=evaluator,
    numFolds=3
)
gbt_cvModel = gbt_crossval.fit(sampled_df)

gbt_best_pipeline = gbt_cvModel.bestModel
best_gbt = gbt_best_pipeline.stages[-1]

gbt_param_dict = {
    param[0].name: param[1]
    for param in best_gbt.extractParamMap().items()
}

print("\nBest GBT Parameters:")
print(json.dumps(gbt_param_dict, indent=4))

# Split the small dataset into train/test sets
train, test = sampled_df.randomSplit([0.7, 0.3], seed=seed)
train = train.cache()
test = test.cache()

# Create final RF and GBT models using best hyperparameters
rf = RandomForestClassifier(
    labelCol='label',
    featuresCol='features',
    seed=seed,
    maxDepth=best_rf.getMaxDepth(),
    maxBins=best_rf.getMaxBins(),
    numTrees=best_rf.getNumTrees
)

gbt = GBTClassifier(
    labelCol='label',
    featuresCol='features',
    seed=seed,
    maxDepth=best_gbt.getMaxDepth(),
    maxBins=best_gbt.getMaxBins(),
    maxIter=best_gbt.getMaxIter()
)

# Fit both models to the training set
rf_pipeline = Pipeline(stages=[vecAssembler, rf])
gbt_pipeline=Pipeline(stages=[vecAssembler, gbt])

rf_model = rf_pipeline.fit(train)
gbt_model = gbt_pipeline.fit(train)

# Evaluate on training and test sets
rf_train_acc = evaluator.evaluate(rf_model.transform(train))
rf_test_acc = evaluator.evaluate(rf_model.transform(test))

gbt_train_acc = evaluator.evaluate(gbt_model.transform(train))
gbt_test_acc = evaluator.evaluate(gbt_model.transform(test))

print("\nPerformance on sampled dataset:")
print("Random Forest:")
print(f" training accuracy: {rf_train_acc}")
print(f" test accuracy: {rf_test_acc}")
print("Gradient Boosted Tree:")
print(f" training accuracy: {gbt_train_acc}")
print(f" test accuracy: {gbt_test_acc}")


##### TASK B #####
print("\n\nTask B:")

# Split full dataset into training and test sets
train_full, test_full = higgs_df.randomSplit([0.7, 0.3], seed=seed)

print(f"Size of full training set: {train_full.count()}")
print(f"Size of full test set: {test_full.count()}")

# Fit RF and GBT models to full training set
rf_model_full = rf_pipeline.fit(train_full)
gbt_model_full = gbt_pipeline.fit(train_full)

# Evaluate on the full training and test sets
rf_full_train_acc = evaluator.evaluate(rf_model_full.transform(train_full))
rf_full_test_acc = evaluator.evaluate(rf_model_full.transform(test_full))

gbt_full_train_acc = evaluator.evaluate(gbt_model_full.transform(train_full))
gbt_full_test_acc = evaluator.evaluate(gbt_model_full.transform(test_full))

print("\nPerformance on full dataset:")
print("Random Forest:")
print(f" training accuracy: {rf_full_train_acc}")
print(f" test accuracy: {rf_full_test_acc}")
print("Gradient Boosted Tree:")
print(f" training accuracy: {gbt_full_train_acc}")
print(f" test accuracy: {gbt_full_test_acc}")

spark.stop()