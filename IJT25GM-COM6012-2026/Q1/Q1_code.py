from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, concat_ws, element_at, sum, dayofmonth, hour, regexp_replace, to_timestamp
import matplotlib.pyplot as plt

# Initialise spark session
spark = SparkSession.builder.appName("Q1").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("\n\nQ1 Results")

# Load and prepare the data
logFile = spark.read.text("../../Data/NASA_access_log_Jul95")

# Split each log line
split_col = split(col("value"), " ")

# Extract host and raw timestamp
logs = logFile.select(
    split_col.getItem(0).alias("host"),
    concat_ws(
        " ",
        split_col.getItem(3),
        split_col.getItem(4)
    ).alias("raw_time")
)

# Remove square brackets from timestamp
logs = logs.withColumn(
    "raw_time",
    regexp_replace(col("raw_time"), "\\[|\\]", "")
)

# Convert to Spark timestamp
logs = logs.withColumn(
    "timestamp",
    to_timestamp(col("raw_time"), "dd/MMM/yyyy:HH:mm:ss Z")
)

# Keep required columns
logs = logs.select("host", "timestamp")


##### TASK A #####
print("\nTask A:")

def host_metrics(data, suffix):
    # Filter for specific category
    filtered = data.filter(col("host").endswith(suffix))
    
    # Calculate metrics
    total = filtered.count()
    unique = filtered.select("host").distinct().count()

    return total, unique

total_ac, unique_ac = host_metrics(logs, ".ac.uk")
total_co, unique_co = host_metrics(logs, ".co.uk")
total_gov, unique_gov = host_metrics(logs, ".gov.uk")

print(f"Sector: Academic | total requests: {total_ac} | unique hosts: {unique_ac}")
print(f"Sector: Company | total requests: {total_co} | unique hosts: {unique_co}")
print(f"Sector: Government | total requests: {total_gov} | unique hosts: {unique_gov}")


##### TASK B #####
print("\nTask B:")

total_shef, _ = host_metrics(logs, "shef.ac.uk")
print(f"1. Total number of requests from the University of Sheffield domain: {total_shef}")

def extract_domain(data, suffix: str, alias: str):
    # Extract all desired hosts
    filtered_hosts = data.filter(col("host").endswith(suffix))

    # Split each hostname into its components (www.shef.ac.uk -> ["www", "shef", "ac", "uk"])
    split_host = split(col("host"), "\\.")

    # Extract institution names by joining the final three components with "."
    institution_domains = filtered_hosts.select(
        col("host"),
        col("timestamp"),
        concat_ws(".",
                element_at(split_host, -3),   # shef
                element_at(split_host, -2),   # ac
                element_at(split_host, -1)    # uk
        ).alias(alias)
    )

    return institution_domains

academic_domains = extract_domain(logs, ".ac.uk", "institution")

# Get number of requests from each institution
institution_counts = academic_domains.groupBy("institution").count().withColumnRenamed("count", "total_requests")

# Create DataFrame of institutions with more requests than Sheffield
greater_than_shef = institution_counts.filter(col("total_requests") > total_shef)

# Count number of institutions with more requests than Sheffield
num_institutions = greater_than_shef.count()
print(f"2. Number of institutions that made more requests than Sheffield: {num_institutions}")

# Print to output and save DataFrame as a csv file
print("3. Institutions with more requests than Sheffield:")
greater_than_shef.show(truncate=False)
greater_than_shef.write.csv("Q1_output.csv", header=True, mode="overwrite")


##### Task C #####
print("Task C:")

# Same process as exercise B:
company_domains = extract_domain(logs, ".co.uk", "company")

# Get number of requests from each company, in descending order
company_counts = company_domains.groupBy("company").count().withColumnRenamed("count", "total_requests")
company_counts = company_counts.orderBy(col("total_requests").desc())

# Get the 9 most active companies
top_companies = company_counts.limit(9)
print("Top-9 active companies by total number of requests:")
top_companies.show(truncate=False)

# Calculate total number of requests from all other companies
top_total = top_companies.agg(sum("total_requests")).collect()[0][0]
total = company_counts.agg(sum("total_requests")).collect()[0][0]
other_total = total - top_total

# Create labels and values for figure
top9_list = top_companies.collect()
labels = [row["company"] for row in top9_list]
values = [row["total_requests"] for row in top9_list]

labels.append("Other .co.uk hosts")
values.append(other_total)

colours = ["steelblue"]* 9 + ["darkorange"]

# Plot top 9 companies against all others
plt.figure(figsize=(7, 4))
plt.bar(labels, values, color=colours, alpha=1.0)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.xlabel("Company")
plt.ylabel("Total Requests")
plt.grid(axis='y', linestyle='-', alpha=0.2)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig("Q1_fig1.png")


##### TASK D #####
print("Task D:")

# Get most active UK institution and company domains
top_institution = institution_counts.orderBy(col("total_requests").desc()).limit(1)
top_company  = top_companies.limit(1)

# Get all government domains
government_domains = extract_domain(logs, ".gov.uk", "government")

# Extract domain names
top_institution_name = top_institution.collect()[0]["institution"]
top_company_name = top_company.collect()[0]["company"]
print(f"Most active academic institution: {top_institution_name}")
print(f"Most active company: {top_company_name}")

# Filter original data for top institution / company
top_institution_requests = academic_domains.filter(col("institution") == top_institution_name)
top_company_requests = company_domains.filter(col("company") == top_company_name)

def create_heatmap(data, title, filename):
    # Group by day of month and hour of day
    heatmap_data = data.groupBy(
        dayofmonth("timestamp").alias("day"),
        hour("timestamp").alias("hour")
    ).count()

    # Convert to pandas
    pdf = heatmap_data.toPandas()

    # Convert rows -> hours, columns -> days
    pivot = pdf.pivot(index="hour", columns="day", values="count").fillna(0)

    # Ensure that plot shows all values e.g. day1, hour 0
    pivot = pivot.reindex(index=range(24), columns=range(1, 29), fill_value=0)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar().set_label("Number of Requests", fontsize=20)
    plt.xlabel("Day of Month", fontsize=20)
    plt.ylabel("Hour of Day", fontsize=20)
    #plt.title(title, fontsize=16)
    plt.xticks(range(0, 28, 2), range(1, 29, 2), fontsize=15)
    plt.yticks(range(0, 24, 2), range(0, 24, 2), fontsize=15)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

create_heatmap(
    top_institution_requests,
    f"Access Pattern for Top UK Academic Institution ({top_institution_name})",
    "Q1_fig2.png"
)
create_heatmap(
    top_company_requests,
    f"Access Pattern for Top UK Company ({top_company_name})",
    "Q1_fig3.png"
)
create_heatmap(
    government_domains,
    "Access Pattern for All UK Government Domains Combined",
    "Q1_fig4.png"
)

spark.stop()