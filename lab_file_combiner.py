import pandas as pd

class LabFileCombiner():

    def read_lab_file(sheet_name):
        path_excel = pd.ExcelFile('../Dataset/Konzentrationen.xlsx')
        lab_file = pd.read_excel(path_excel, sheet_name,header = None,index_col=None)
        lab_file=pd.DataFrame(lab_file)
        lab_file.drop(0, inplace=True)
        return lab_file


    lab_file_df1=read_lab_file('Kalibrierproben')
    #print(lab_file_df1)
    lab_file_df2 = read_lab_file('Test-Set-Proben')

    def combine_lab_file(df1, df2):
        combined_df = pd.DataFrame()
        combined_df = pd.concat([df1, df2], join_axes=[df1.columns])
        combined_df = combined_df.reset_index()
        combined_df = combined_df.drop(['index'], axis=1)
        combined_df.rename(
            columns={0: 'Probenname', 1: 'glukose', 2: 'harnstoff', 3: 'kreatinin', 4: 'phosphat', 5: 'laktat'},
            inplace=True)
        return combined_df
        #print(combined_df)


    combined_label_file=combine_lab_file(lab_file_df1, lab_file_df2)

