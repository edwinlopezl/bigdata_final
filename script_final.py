from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.sql import functions as F

import sys

spark = SparkSession\
      .builder\
      .appName("TrabajoFinal")\
      .getOrCreate()

if len(sys.argv) != 2:
    print("Uso: python script_final.py <nombre_del_archivo.csv>")
    sys.exit(1)

archivo_csv = sys.argv[1]

# Carga de la base
df = spark.read.csv(archivo_csv, header = True, inferSchema = True)

# Limpieza de datos
# Eliminacion de label neutro, y creacion de label numerico
df = df.filter((col("sentiment") == "positive") | (col("sentiment") == "negative"))
df = df.withColumn('label', when(col('sentiment') == 'positive', 1).otherwise(0))

# Segmentacion en train y test
train_ratio = 0.7
test_ratio = 1 - train_ratio
training, test = df.randomSplit([train_ratio, test_ratio], seed = 42)

# Tokenizer, hashing, y la regresion logistica
tokenizer = Tokenizer(inputCol = "text", outputCol = "words")
hashingTF = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter = 10, regParam = 0.001)

# Configura el pipeline para darle input a la regresión logística
pipeline = Pipeline(stages = [tokenizer, hashingTF, lr])

# Ajusta el modelo al conjunto de entrenamiento
model = pipeline.fit(training)

# Obtiene salida del conjunto de testeo
prediction = model.transform(test)
selected = prediction.select("text", "label", "probability", "prediction")
selected = selected.withColumn("prediction", col("prediction").cast("int"))

# Prepara metricas de evaluacion del modelo
confusion_matrix = (
    selected.groupBy("label", "prediction")
    .count()
    .groupBy("label")
    .pivot("prediction")
    .agg(F.sum("count").alias("count"))
    .na.fill(0)
)

confusion_matrix = confusion_matrix.orderBy("label", ascending = True)

print('                                   ')
print('                                   ')
print('-----------------------------------')
print('------- MATRIZ DE CONFUSION -------')
print('-----------------------------------')
print('                                   ')
confusion_matrix.show()

# Metricas - Modelo de clasificacion
tn = confusion_matrix.filter(col("label") == 0).select("0").first()[0]
tp = confusion_matrix.filter(col("label") == 1).select("1").first()[0]
fn = confusion_matrix.filter(col("label") == 1).select("0").first()[0]
fp = confusion_matrix.filter(col("label") == 0).select("1").first()[0]

precision = tp / (tp + fp) 
recall = tp / (tp + fn)  
accuracy = (tp + tn) / (tp + fp + tn + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print('                                   ')
print('                                   ')
print('-----------------------------------')
print('----- METRICAS DEL EJERCICIO ------')
print('-----------------------------------')
print('                                   ')
print("\nPrecision: {}".format(round(precision, 4)))
print("Recall: {}".format(round(recall, 4)))
print("F1 Score: {}".format(round(f1_score, 4)))
print("Accuracy: {}".format(round(accuracy, 4)))

spark.stop()