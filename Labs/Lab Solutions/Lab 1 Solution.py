from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[1]") \
        .appName("Lab 1 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/your_username") \
        .getOrCreate()


sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("./Data/NASA_access_log_Aug95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
logFile.show(20, False)

# Q1: How many requests are there in total ?
print("==================== Question 1 ====================")
print(f"There are {logFile.count()} requests in total.")
print("====================================================")



# Q2 How many requests are from gateway.timken.com ? 

q2_out = logFile.filter(logFile.value.contains("gateway.timken.com")).cache() # cache it for Question 6
q2_out.show(20, False)
print("==================== Question 2 ====================")
print(f"There are {q2_out.count()} requests are from gateway.timken.com.")
print("====================================================")


# Q3 How many requests are on 15th August 1995 ?

q3_out = logFile.filter(logFile.value.contains("15/Aug/1995"))
q3_out.show(20,False)
print("==================== Question 3 ====================")
print(f"There are {q3_out.count()} requests are on 15th August 1995.")
print("====================================================")


# Q4 How many 404 (page not found) errors are there in total ?
q4_out = logFile.filter(logFile.value.contains(" 404 "))
q4_out.show(20,False)
print("==================== Question 4 ====================")
print(f"There are {q4_out.count()} requests are 404.")
print("====================================================")

# Q5 How many 404 errors are there on 15 th August?
q5_out1 = logFile.filter(logFile.value.contains("15/Aug")).cache()  # cache it for the next step
q5_out2 = q5_out1.filter(q5_out1.value.contains(" 404 "))
q5_out2.show(20, False)
print("==================== Question 5 ====================")
print(f"There are {q5_out2.count()} requests are 404 on 15th August.")
print("====================================================")

# Q6 How many 404 errors from gateway.timken.com are there on 15th August ?
q6_out1 = q2_out.filter(q2_out.value.contains("15/Aug")).cache()
q6_out1.show(20,False)
q6_out2 = q6_out1.filter(q6_out1.value.contains(" 404 "))
q6_out2.show(20,False)
print("==================== Question 6 ====================")
print(f"There are {q6_out2.count()} requests are 404 on 15th August from gateway.timken.com.")
print("====================================================")
