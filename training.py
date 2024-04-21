# -*- mode: python -*-

import math
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import warnings

import auxf
from metrics import Metrics
from func import Func


class Training:

    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        """
        Runs the training process for three models: MLR, MQR, and RFR.
        """
        # Creating new directories:
        pathxmodel = auxf.newdirectory(self.parameters.dest + '/temp', 'models')
        # Models:
        self.mlr(self.parameters.dest + '/temp/LUT/OLCI_LUT.csv', self.parameters.dest + '/temp/LUT/MSI_LUT.csv', pathxmodel)

    def mlr(self, olci, msi, dest: str) -> None:
        # Reads the dataset of input and output:
        dfOLCI = pd.read_csv(olci, sep=',')
        dfMSI = pd.read_csv(msi, sep=',')
        # Removes the NaN values in dataframes:
        nan_values = dfOLCI.loc[pd.isna(dfOLCI["Oa04"]), :].index
        if len(nan_values):
            dfOLCI = dfOLCI.drop(nan_values)
            dfMSI = dfMSI.drop(nan_values)
        # Training the models per band:
        for i in self.parameters.OLCIBAND:
            X = dfMSI[[j for j in self.parameters.MSIBAND]].to_numpy() # Input features
            y = dfOLCI[i].to_numpy() # Output features
            # Stores the values:
            Xtrain = []
            ytrain = []
            Xtest = []
            ytest = []
            model = []
            metricGENERAL = []
            metricMAE = []
            for j in range(0, 101):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70) # Monte Carlo process.
                popt, pcov = curve_fit(Func.func_mlr, X_train.transpose(), y_train)
                y_pred = Func.func_mlr(x=X_test.transpose(), A0=popt[0], A1=popt[1], A2=popt[2], A3=popt[3], A4=popt[4])
                metricMAE.append(Metrics.MAELOG(y_test, y_pred))
                warnings.filterwarnings("ignore")
                metricGENERAL.append({'mae_log': Metrics.MAELOG(y_test, y_pred), 'biaslog': Metrics.BIASLOG(y_test, y_pred)})
                model.append(popt)
                Xtrain.append(X_train)
                ytrain.append(y_train)
                Xtest.append(X_test)
                ytest.append(y_test)
            # Selects the best model based on MAE metric:
            without_nan = [value for value in metricMAE if not (math.isnan(value)) == True] # Removes the MAE invalids.
            minimumMAE = np.min(without_nan)
            index = metricMAE.index(minimumMAE)
            general = metricGENERAL[index]
            # Save the models:
            dfout = pd.DataFrame({'Xtrain': [Xtrain[index]], 'ytrain': [ytrain[index]], 'Xtest': [Xtest[index]], 'ytest': [ytest[index]], 'model': [model[index]], 'metrics': [[general]]})
            dfout.to_pickle(dest + '/' + 'MLR_Model_' + str(i) + '.pkl')
        return None


from metadata import Metadata

# path_MSI = r'/Volumes/rsp/s23aqua_validation/S2A_MSIL1C_20210809T131251_N0500_R138_T23KLP_20230116T033833.SAFE'
# path_OLCI = r'/Volumes/rsp/s23aqua_validation/S3A_OL_1_EFR____20210809T121855_20210809T122155_20210810T171811_0179_075_066_3420_LN1_O_NT_002.SEN3'
# roi = r'/Volumes/rsp/s23aqua_validation/ROI/roi_billings_4326.shp'
# dest = r'/Volumes/rsp/s23aqua_validation/out'
#
# a = Metadata(path_MSI, path_OLCI, roi, dest)
# a.run()
#
# b = Training(a)
# b.run()