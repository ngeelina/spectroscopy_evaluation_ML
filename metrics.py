from sklearn.metrics import mean_squared_error
import numpy as np
#from sklearn.metrics import median_absolute_error


class MetricsCalculator():
    def __init__(self):
        pass

    #def evaluate(self, model_name, y_true_glucose, y_pred_glucose):
    def evaluate(self, model_name, y_true_glucose, y_pred_glucose):

        mse = mean_squared_error(y_true_glucose, y_pred_glucose)
        mse = mse.mean()
        print(model_name, np.sqrt(mse))

        # mae = median_absolute_error(Y_test, Y_pred)  Multioutput not supported in median_absolute_error