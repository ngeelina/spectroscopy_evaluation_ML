from sklearn.linear_model import Lasso
from model_trainer import ModelTrainer
from metrics import MetricsCalculator
from plotter import Visualizer

class LassoRegressor():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train

    def perform_lasso_regression(self):

        print('................................... LASSO REGRESSION ............................................')
        model_trainer = ModelTrainer()
        lasso = Lasso()
        Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(lasso, self.X_train, self.X_test,
                                                                                   self.Y_train, self.Y_test)
        evl = MetricsCalculator()
        evl.evaluate('root mean square error for lasso regression',y_true_glucose, y_pred_glucose)

        viz = Visualizer()
        viz.visualize('lasso regression', y_true_glucose, y_pred_glucose)