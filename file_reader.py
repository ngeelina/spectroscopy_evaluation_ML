import glob
import os
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import rampy as rp

class FileReader():
    pathTrain = "../Dataset/Kalibrierproben/*.txt"
    pathTest = '../Dataset/Test-Set-Proben/*.txt'
    pathLab='../Dataset/Konzentrationen.xlsx'


    def read_file(filePath):
        files=glob.glob(filePath)
        print("len",len(files))

        absorption_dataframe=pd.DataFrame()

        for file in files:

            filename_w_ext = os.path.basename(file)
            file_name = os.path.splitext(filename_w_ext)[0]
            file_data = pd.read_fwf(file, header=None, names={'wavenumber', 'absorption'})

            data_wavenumber = file_data['wavenumber']
            #print('file reader', data_wavenumber)
            data_absorption = file_data['absorption']
            #print(data_absorption)
            f_linear=interp1d(data_wavenumber, data_absorption,kind='linear')
            wavenumber = np.linspace(950, 3500, 4000)
            #wavenumber = np.arange(1050, 3000, 50)
            #wavenumber=np.linspace(data_wavenumber.min(),data_wavenumber.max(),50)
            #print('wavenumber frm file reader', wavenumber)
            #print(max(data_wavenumber))
            absorption=f_linear(wavenumber)
            absorption = rp.normalise(absorption, method="intensity")
            #print("absorption",absorption)
            absorption_df=pd.DataFrame(absorption)
            absorption_df=absorption_df.T
            absorption_df['Probenname'] = file_name

            absorption_dataframe = pd.concat([absorption_dataframe, absorption_df])

        return absorption_dataframe

    a = read_file(pathTrain)
    #print(a)
    b = read_file(pathTest)
    #print(b)

    def combine_data(df1, df2):
        combined_df = pd.DataFrame()
        combined_df = pd.concat([df1, df2], join_axes=[df1.columns])
        combined_df = combined_df.reset_index()
        combined_df = combined_df.drop(['index'], axis=1)
        return combined_df


    combined_data_file=combine_data(a, b)





