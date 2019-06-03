from sklearn.model_selection import train_test_split
from dataset_combiner import DatasetCombiner as dataSetCombiner

class TrainTestSplitter():

    def __init__(self):
        pass


    def splitDatasets(self):
        x,y = dataSetCombiner.combine_dataset(self)
        x = x.drop(['Probenname', 'glukose','harnstoff', 'kreatinin', 'phosphat', 'laktat'], axis = 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        # print('inside train test splitter')
        # print('x_train', x_train)
        # print()
        # print('x_test', x_test)
        # print()
        # print('y_train', y_train)
        # print()
        # print('y_test', y_test)
        return x_train, x_test, y_train, y_test

