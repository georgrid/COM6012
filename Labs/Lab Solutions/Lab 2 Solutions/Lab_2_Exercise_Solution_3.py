
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise Logistic Regression") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 
# document classification

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

training = spark.createDataFrame([
        (0, "a b c d e spark", 1.0),
        (1, "b d", 0.0),
        (2, "spark f g h", 1.0),
        (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(training)
# create new test data
test = spark.createDataFrame([
        (4, "spark hadoop"),
        (5, "spark a b c"),
        (6, "mapreduce spark")
], ["id", "text"])
prediction = model.transform(test)
prediction.show()
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
