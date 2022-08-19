from pyspark.sql.types import IntegerType, FloatType

def process(df):
    rating = df.withColumn("userId", df["userId"].cast(IntegerType()))
    rating = rating.withColumn("movieId", rating["movieId"].cast(IntegerType()))
    rating = rating.withColumn("rating", rating["rating"].cast(FloatType()))

    movie_ratings = rating.drop('timestamp')

    # Create test and train set
    (train, test) = movie_ratings.randomSplit([0.8, 0.2])
    return train, test