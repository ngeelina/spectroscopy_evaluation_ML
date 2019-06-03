from sklearn.linear_model import Ridge
from model_trainer import ModelTrainer
from metrics import MetricsCalculator
from plotter import Visualizer

class RidgeRegressor():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train

    def perform_ridge_regression(self):
        print('*********************************************RIDGE REGRESSION**************************************************')
        model_trainer = ModelTrainer()
        ridge = Ridge(alpha = 1.0)
        Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(ridge, self.X_train, self.X_test,
                                                                                   self.Y_train, self.Y_test)
        evl = MetricsCalculator()
        evl.evaluate('root mean square error for ridge regression',y_true_glucose, y_pred_glucose)

        viz = Visualizer()
        viz.visualize('ridge regression', y_true_glucose, y_pred_glucose)
