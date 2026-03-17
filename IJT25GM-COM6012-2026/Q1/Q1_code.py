from pyspark.sql.functions import split

hosts = logFile.select(split(logFile.value, " ").getItem(0).alias("host"))

hostsUni = hosts.filter(hosts.host.endswith(".ac.uk"))