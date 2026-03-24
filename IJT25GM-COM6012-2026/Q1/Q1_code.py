from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, concat_ws, element_at

# Initialise spark session
spark = SparkSession.builder.appName("Q1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ1 Results")


##### TASK A #####
print("\nTask A:")

# Load data
logFile = spark.read.text("../../Data/NASA_access_log_Jul95")

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


##### TASK B #####
print("\nTask B:")

total_shef, _ = host_metrics(hosts, "shef.ac.uk")
print(f"1. Total number of requests from the University of Sheffield domain: {total_shef}")

# Extract all academic hosts
academic_hosts = hosts.filter(col("host").endswith(".ac.uk"))

# Split each hostname into its components (www.shef.ac.uk -> ["www", "shef", "ac", "uk"])
split_host = split(col("host"), "\\.")

# Extract institution names by joining the final three components with "."
academic_domains = academic_hosts.select(
    col("host"),
    concat_ws(".",
              element_at(split_host, -3),   # shef
              element_at(split_host, -2),   # ac
              element_at(split_host, -1)    # uk
    ).alias("institution")
)

# Get number of requests from each institution
institution_counts = academic_domains.groupBy("institution").count()

# Create DataFrame of institutions with more requests than Sheffield
greater_than_shef = institution_counts.filter(col("count") > total_shef).withColumnRenamed("count", "total_requests")

# Count number of institutions with more requests than Sheffield
num_institutions = greater_than_shef.count()
print(f"2. Number of institutions that made more requests than Sheffield: {num_institutions}")

# Print to output and save DataFrame as a csv file
print("3. Institutions with more requests than Sheffield:")
greater_than_shef.show(truncate=False)
greater_than_shef.write.csv("Q1_B3_output.csv", header=True, mode="overwrite")


spark.stop()