from pyspark.ml.evaluation import RegressionEvaluator

def predict(model, data):
    eval = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    #Generate predictions and evaluate using RMSE
    predict=model.transform(data)
    rmse = eval.evaluate(predict)
    #Print evaluation metrics and model parameters
    print ("RMSE = "+str(rmse))
    print ("**Best Model**")
    print (" Rank: ", str(model._java_obj.parent().getRank())),
    print (" MaxIter: ", str(model._java_obj.parent().getMaxIter())),
    print (" RegParam: ", str(model._java_obj.parent().getRegParam()))
    predictions.show()