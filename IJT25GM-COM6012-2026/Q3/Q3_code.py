from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import randomForestClassifier, GBTClassifier
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
logFile = spark.read \
    .option("header", False) \
    .schema(schema) \
    .csv("/users/ijt25gm/com6012/ScalableML/Data/HIGGS.csv")

logFile = logFile.cache()
logFile.count()

# Stratified sampling: sample 2% of each class
fractions = {0.0: 0.02, 1.0: 0.02}
sampled_df = logFile.sampleBy("label", fractions, seed=seed)

sampled_df = sampled_df.cache()
sampled_df.count()

# Create feature vector
feature_cols = [f"feature_{i}" for i in range(1, 29)]

vecAssembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

# Random Forest cross-validation
rf = randomForestClassifier(
    labelCol="label",
    featuresCol="features",
    seed=seed
)

# Combine stages into pipeline
rf_pipeline = Pipeline(stages=[vecAssembler, rf])

# Create parameter grid for cross-validation
rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.maxBins, [32, 64, 128]) \
    .addGrid(rf.numTrees, [20, 50, 100]) \
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

spark.stop()