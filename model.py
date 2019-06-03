import pandas as pd
from train_test_splitter import TrainTestSplitter
from partial_least_square import PartialLeastSquare
from linear_regressor import LnrRegression
from lasso_regressor import LassoRegressor
from ridge_regressor import RidgeRegressor
from neural_network import NeuralNetwork
from support_vector_regressor import SupportVectorRegressor

class Model():

    def __init__(self):
        pass

    def call_models(self):
        trainTestSplitter = TrainTestSplitter()
        X_train, X_test, Y_train, Y_test = trainTestSplitter.splitDatasets()

        print('Xtrain', X_train.shape)
        print('Ytrain', Y_train.shape)
        print('Xtest', X_test.shape)
        print('Ytest', Y_test.shape)

        pls = PartialLeastSquare(X_train, X_test, Y_train, Y_test)
        pls.perform_PLS()
        # lnr_reg = LnrRegression(X_train, X_test, Y_train, Y_test)
        # lnr_reg.perform_linear_regression()
        # lasso_ref = LassoRegressor(X_train, X_test, Y_train, Y_test)
        # lasso_ref.perform_lasso_regression()
        # ridge_ref = RidgeRegressor(X_train, X_test, Y_train, Y_test)
        # ridge_ref.perform_ridge_regression()
        #nn = NeuralNetwork(X_train, X_test, Y_train, Y_test)
        #nn.perform_NN()

        #svr = SupportVectorRegressor(X_train, X_test, Y_train, Y_test)
        #svr.perform_SVR()


# def main():
#    m = Model()
#    m.call_models()
#
# if __name__ == '__main__':
#    main()







