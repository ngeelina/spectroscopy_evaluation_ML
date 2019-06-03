from train_test_splitter import TrainTestSplitter
from sklearn.svm import SVR
import pandas as pd
from model_trainer import ModelTrainer

class SupportVectorRegressor():
    def __init__(self, X_train, X_test, Y_train, Y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.Y_train = Y_train

    def perform_SVR(self):
        print('SVRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
        model_trainer = ModelTrainer()
        svr = SVR(gamma='poly', C=1e3, epsilon=0.2)
        Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(svr, self.X_train, self.X_test,
                                                                                   self.Y_train, self.Y_test)


        # trainTestSplitter = TrainTestSplitter()
        # X_train, X_test, Y_train, Y_test = trainTestSplitter.splitDatasets()
        # svr = SVR(gamma='scale', C=1.0, epsilon=0.2)
        # svr.fit(X_train, Y_train)
        # Y_pred = svr.predict(X_test)
        # Y_test = pd.DataFrame(Y_test)
        # print('y true..............', Y_test)
        # print('y_pred...................', Y_pred)
        # y_true_glucose = Y_test.iloc[:, 0]
        # y_pred_glucose = [item[0] for item in Y_pred]
        # y_pred_glucose = pd.DataFrame(y_pred_glucose)
        # print('glucose.true......', y_true_glucose)
        # print('glucose.predicted......', y_pred_glucose)
