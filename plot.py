import matplotlib.pyplot as plt
import numpy as np
# bar chart of ratings
def plot(df):
    rating = df.drop('timestamp')
    count=rating.select('rating').groupBy('rating').count().toPandas()
    plt.figure(figsize=[12,6])
    plt.bar(x='rating',height='count',data=count,width=0.5)
    plt.title('Ratings Distribution')
    plt.xticks(np.arange(0.5,5.5,0.5))
    plt.show()