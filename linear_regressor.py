from sklearn.linear_model import LinearRegression
from model_trainer import ModelTrainer
from metrics import MetricsCalculator
from plotter import Visualizer


class LnrRegression():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train

    def perform_linear_regression(self):

        print('------------------------------------------LINEAR REGRESSION------------------------------------------')
        model_trainer = ModelTrainer()
        linear_reg = LinearRegression()
        Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(linear_reg, self.X_train, self.X_test,
                                                                                   self.Y_train, self.Y_test)
        evl = MetricsCalculator()
        evl.evaluate('root mean square error for linear regression',y_true_glucose, y_pred_glucose)

        viz = Visualizer()
        viz.visualize('linear regression', y_true_glucose, y_pred_glucose)

