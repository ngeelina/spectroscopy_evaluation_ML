import matplotlib.pyplot as plt


class Visualizer():

    def __init__(self):
        pass

    def visualize(self, model_name, y_true_glucose, y_pred_glucose):

        #plt.plot(y_true_glucose, 'r--')
        #plt.plot(y_pred_glucose, 'b--')
        plt.plot(y_true_glucose, y_pred_glucose, "r.")
        plt.title (model_name)

        #plt.scatter(y_true_glucose, y_pred_glucose, s=50, c='red', alpha=0.5)
        plt.show()
