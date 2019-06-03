import pandas as pd
from file_reader import FileReader as fileReader
from lab_file_combiner import LabFileCombiner as labFileCombiner

class DatasetCombiner():

    def __init__(self):
        pass

    def combine_dataset(self):
        data_df_merged = pd.DataFrame()
        data_df_merged = pd.merge(fileReader.combined_data_file, labFileCombiner.combined_label_file[['Probenname', 'glukose','harnstoff', 'kreatinin', 'phosphat', 'laktat']], how='inner', on='Probenname')
        data_df_y = data_df_merged[['glukose','harnstoff', 'kreatinin', 'phosphat', 'laktat']]
        #data_df_y = data_df_merged[['glukose']]
        #print('targets from dataset_combiner')
        #print('t', data_df_y)
        return data_df_merged, data_df_y
