from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
import numpy as np

def create(data):
    model = ALS(maxIter=5, rank=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop", seed=6)

    #Tune model using ParamGridBuilder
    params = ParamGridBuilder()\
            .addGrid(model.maxIter, [3, 5, 10])\
            .addGrid(model.regParam, [0.1, 0.01, 0.001])\
            .addGrid(model.rank, [5, 10, 15, 20, 25])\
            .addGrid(model.alpha, [0.1, 0.01, 0.001])\
            .build()

    # Define evaluator as RMSE
    eval = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")


    cv = CrossValidator.load('/content/drive/MyDrive/ml-latest-small/cv')

    #Fit ALS model to training data
    cvModel = cv.fit(data)

    #Extract best model from the tuning exercise using ParamGridBuilder
    bestModel=cvModel.bestModel

    # Check the best parameters
    best_params = cvModel.getEstimatorParamMaps()[np.argmin(cvModel.avgMetrics)]
    print('Best ALS model parameters by CV:')
    for i,j in best_params.items():
        print('-> '+i.name+': '+str(j))

    #Extract best model from the tuning exercise using ParamGridBuilder
    prediction_train=cvModel.transform(data)
    rmse_train = eval.evaluate(prediction_train)
    print("Root-mean-square error for training data is " + str(rmse_train))
    return bestModel