from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col

# Initialise spark session
spark = SparkSession.builder.appName("Q1").getOrCreate()

# TASK A
# Load data
logFile = spark.read.text("Data/NASA_access_log_Jul95")

# Extract hostname, which is the first field in each request
hosts = logFile.select(split(logFile.value, " ").getItem(0).alias("host"))

def host_metrics(data, suffix):
    # Filter for specific category
    filtered = data.filter(col("host").endswith(suffix))
    
    # Calculate metrics
    total = filtered.count()
    unique = filtered.select("host").distinct().count()

    return total, unique

total_ac, unique_ac = host_metrics(hosts, ".ac.uk")
total_co, unique_co = host_metrics(hosts, ".co.uk")
total_gov, unique_gov = host_metrics(hosts, ".gov.uk")

print(f"Sector: Academic | total requests: {total_ac} | unique hosts: {unique_ac}")
print(f"Sector: Company | total requests: {total_co} | unique hosts: {unique_co}")
print(f"Sector: Government | total requests: {total_gov} | unique hosts: {unique_gov}")

spark.stop()