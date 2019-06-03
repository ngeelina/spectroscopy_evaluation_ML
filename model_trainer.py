import pandas as pd


class ModelTrainer():
    def __init__(self):
        pass

    #def train_model(self, model):
    def train_model(self, model, X_train, X_test, Y_train, Y_test):
        #print('inside train model')

        #Y_train['glukose'] = Y_train['glukose'].astype(float)
        #Y_train = Y_train.values.ravel()
        #Y_train = np.array(y)
        #print(Y_train)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        #Y_pred = pd.DataFrame(Y_pred)

        Y_test = pd.DataFrame(Y_test)

        #true_pred = Y_test.copy()

        print('+++++++++++++++++ Y True +++++++++++++++++++++')
        print(Y_test)
        print('+++++++++++++++++ Y Predicted +++++++++++++++++++++')
        print(Y_pred)
        #true_pred['predicted_glukose'] = Y_pred
        #print(true_pred)

        y_true_glucose = Y_test[['glukose']]

        true_pred= y_true_glucose.copy()
        y_pred_glucose = [item[0] for item in Y_pred]

        true_pred['predicted_glukose'] = y_pred_glucose

        print(true_pred)

        return Y_test, Y_pred, y_true_glucose, y_pred_glucose
        #return Y_test, Y_pred

