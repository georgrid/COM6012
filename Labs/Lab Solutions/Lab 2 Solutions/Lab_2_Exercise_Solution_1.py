from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise Log Mining") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("./Data/NASA_access_log_Aug95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
logFile.show(20, False)

# Pre-split once to avoid repeating F.split(...) four times
parts = F.split(F.col("value"), " ")

# split into 5 columns using regex and split
data = (
    logFile
    .withColumn("host", F.regexp_extract(F.col("value"), r"^(.*) - -.*", 1))
    .withColumn("timestamp", F.regexp_extract(F.col("value"), r".* - - \[(.*)\].*", 1))  # raw string fixes SyntaxWarning
    .withColumn("request", F.regexp_extract(F.col("value"), r'.*"(.*)".*', 1))
    .withColumn("HTTP reply code", parts[F.size(parts) - 2].cast("int"))
    # bytes can be '-', so turn it into NULL before casting
    .withColumn(
        "bytes in the reply",
        F.when(parts[F.size(parts) - 1] == "-", F.lit(None))
         .otherwise(parts[F.size(parts) - 1])
         .cast("int")
    )
    .drop("value")
    .cache()
)
data.show(20, False)

# number of unique hosts

n_hosts = data.select('host').distinct().count()
print("==================== Question 2 ====================")
print(f"There are {n_hosts} unique hosts")
print("====================================================")

# most visited host

host_count = data.select('host').groupBy('host').count().sort('count', ascending=False)
host_max = host_count.select("host").first()['host']
print("==================== Question 3 ====================")
print(f"The most frequently visited host is {host_max}")
print("====================================================")


