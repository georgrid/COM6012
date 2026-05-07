from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Lab 7 Exercise") \
        .config("spark.local.dir","/mnt/parscratch/users/YOUR_USERNAME") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

iris = spark.read.load('../Data/iris.csv', format = 'csv', inferSchema = 'true', header = "true").cache()

iris.show(20, False)

from pyspark.ml.linalg import Vectors

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features','id'])

dfFeatureVec= transData(iris).cache()
from pyspark.ml.feature import PCA

# normalization
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(withMean = True, withStd = True, inputCol = 'features',outputCol = 'scaled_features')
scaler_model = scaler.fit(dfFeatureVec)
dfFeatureVec = scaler_model.transform(dfFeatureVec).drop('features').withColumnRenamed('scaled_features','features')

dfFeatureVec.show(20,False)

# DataFrame PCA API
pca = PCA(k = 2, inputCol = 'features').setOutputCol('pca_features')
pca_model = pca.fit(dfFeatureVec)

print("DataFrame PCA")
pca_feature = pca_model.transform(dfFeatureVec)
pca_feature.show(20,False)

import numpy as np

setosa = np.array([row.pca_features.toArray() for row in pca_feature.filter(pca_feature.id == "setosa").collect()])
versicolor = np.array([row.pca_features.toArray() for row in pca_feature.filter(pca_feature.id == "versicolor").collect()])
virginica = np.array([row.pca_features.toArray() for row in pca_feature.filter(pca_feature.id == "virginica").collect()])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.scatter(setosa[:,0], setosa[:,1], c = 'b', label = 'setosa')
plt.scatter(versicolor[:,0], versicolor[:,1], c = 'g', label = 'versicolor')
plt.scatter(virginica[:,0], virginica[:,1], c = 'r', label = 'virginica')
plt.legend()
plt.savefig('../Output/Lab7_plot_PCA.png')

# DataFrame PCA components
print("DataFrame PC")
pc = pca_model.pc
print(pc)

from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors

# RDD PCA API
iris_rm = RowMatrix(dfFeatureVec.rdd.map(lambda x: Vectors.dense(x[1].tolist())))
rdd_pc = iris_rm.computePrincipalComponents(2)
print("RDD PC:")
print(rdd_pc)
projected = iris_rm.multiply(rdd_pc)
print("RDD PCA projected features")
print(projected.rows.collect())

# RDD SVD API
svd = iris_rm.computeSVD(2,True)
s = svd.s
V = svd.V
print("RDD SVD for PCA")
print(V)
