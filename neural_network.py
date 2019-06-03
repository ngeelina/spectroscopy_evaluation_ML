from sklearn.neural_network import MLPRegressor
from model_trainer import ModelTrainer
from metrics import MetricsCalculator
from plotter import Visualizer

class NeuralNetwork():

    def __init__(self, X_train, X_test, Y_train, Y_test):
                self.X_train = X_train
                self.X_test = X_test
                self.Y_test = Y_test
                self.Y_train = Y_train

    def perform_NN(self):
        print('/////////////////////////////////////////////////// NEURAL NETWORK ///////////////////////////////////')
        model_trainer = ModelTrainer()
        nn = MLPRegressor(hidden_layer_sizes=(200, ), activation='relu', solver='adam', alpha=0.1, batch_size='auto',
                          learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=3000, shuffle=True,
                          random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                          early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
        Y_test, Y_pred, y_true_glucose, y_pred_glucose = model_trainer.train_model(nn, self.X_train, self.X_test,
                                                                                   self.Y_train, self.Y_test)

        evl = MetricsCalculator()
        evl.evaluate('root mean square error for Neural network',y_true_glucose, y_pred_glucose)

        viz = Visualizer()
        viz.visualize('neural  network', y_true_glucose, y_pred_glucose)