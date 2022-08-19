from pyspark.sql import SparkSession

from process import process
from model import *
from test import *
from recommend import *
from matrix import *

spark = SparkSession.builder\
        .master("local")\
        .appName("recommendation")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()

movie = spark.read.load("ml-latest/movies.csv", format='csv', header = True)
rate = spark.read.load("ml-latest/ratings.csv", format='csv', header = True)
link = spark.read.load("ml-latest/links.csv", format='csv', header = True)
tag = spark.read.load("ml-latest/tags.csv", format='csv', header = True)

train, test = process(rate)
model = create(train)

predict(model, test)

# top 10 for user 575
Recommend(movie,model,10,575)
# top 15 for user 232
Recommend(movie,model,15,232)

# movie id 471, method 1, top 10 similar
out21,ssd2=dist_sim(movie, model, 10, 471)

# movie id 471, method 2, top 10 similar
out22,inner2=cos_sim(movie, model, 10, 471)

print(out21)
print(out21)




