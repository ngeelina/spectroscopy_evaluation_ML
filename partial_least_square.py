from sklearn.cross_decomposition import PLSRegression
#from sklearn.model_selection import cross_val_score
from model_trainer import ModelTrainer
from metrics import MetricsCalculator
from plotter import Visualizer
import matplotlib.pyplot as plt

class PartialLeastSquare():

        def __init__(self,X_train, X_test, Y_train, Y_test):
                self.X_train = X_train
                self.X_test = X_test
                self.Y_test = Y_test
                self.Y_train = Y_train

        def perform_PLS(self):
                print(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, PARTIAL LEAST SQUARE ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
                model_trainer = ModelTrainer()

                pls = PLSRegression(n_components=20, scale=True, max_iter=5000, tol=1e-06, copy=True)
                Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(pls, self.X_train, self.X_test, self.Y_train, self.Y_test)

                evl = MetricsCalculator()
                evl.evaluate('root mean square error for partial least square',y_true_glucose, y_pred_glucose)

                viz = Visualizer()
                viz.visualize('pls',y_true_glucose, y_pred_glucose)


                # Y_test, Y_pred = model_trainer.train_model(pls, self.X_train,
                #                                            self.X_test, self.Y_train,
                #                                            self.Y_test)
                # evl.evaluate('root mean square error for partial least square', Y_test, Y_pred)
                # viz.visualize('pls', Y_test, Y_pred)

                                #loo = cross_val_score.LeaveOneOut(len(Y))
                #scores = cross_val_score(pls, self.X_train, self.Y_train)
                #print ('rmsecv..............', scores)

                # plt.plot(y_true_glucose, 'r--')
                # plt.plot( y_pred_glucose, 'b--')
                # plt.show()


