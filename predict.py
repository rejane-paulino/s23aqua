# -*- mode: python -*-

import os
import glob
import numpy as np
import pandas as pd

import auxf
from func import Func


class Predict:

    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        """
        Runs the prediction for all models.
        """
        # Loads MSI bands as input:
        paths = [os.path.normpath(band) for band in glob.glob(os.path.join(self.parameters.dest + '/temp/smoothedMSI', '*.tif')) if 'B02' in band or 'B03' in band or 'B04' in band or 'B05' in band]
        paths.sort()
        paths = paths + [os.path.normpath(mask) for mask in glob.glob(os.path.join(self.parameters.dest + '/temp/resampled', '*.tif')) if 'watermask' in mask]
        index = self.parameters.MSIBAND + ['watermask']
        array = auxf.loadarray(paths, index)
        # Arranges the input:
        values = self.input(array)
        self.predmlr(self.parameters.dest + '/temp/models', values, array, self.parameters.dest + '/temp/smoothedMSI/B02.tif', self.parameters.dest)


    def predmlr(self, pathxmodel: str, Xarray: list, array: dict, reference: str, dest:str) -> None:
        """
        Predicts values to MLR model.
        """
        for y_band in self.parameters.OLCIBAND:
            # Access the model:
            model = pd.read_pickle(pathxmodel + '/' + 'MLR_Model_' + y_band + '.pkl')['model'][0]
            y_pred = Func.func_mlr(x=Xarray[0].transpose(), A0=model[0], A1=model[1], A2=model[2], A3=model[3], A4=model[4])
            # Returns to original shape-size and saves ssa product:
            ssar = y_pred.reshape(Xarray[1])
            ssamasked = ssar * np.where(array['watermask'] == 1, 1, np.nan) # it is used to mask water pixels.
            ssamasked_out = np.where(ssamasked == np.nan, self.parameters.nodata, ssamasked) # It is necessary because the image does not cover all the reservoir.
            auxf.export(ssamasked_out, 'S23AQUA_' + self.parameters.date + '_' + self.parameters.grid + '_' + y_band, reference, dest)
        return None


    def input(self, array: dict) -> list:
        """
        Arranges the image values.
        """
        # Stacks the bands:
        x, y = array['B02'].shape
        img = np.zeros((x, y, 4))
        img[:, :, 0] = array['B02']
        img[:, :, 1] = array['B03']
        img[:, :, 2] = array['B04']
        img[:, :, 3] = array['B05']
        # Reduces the array (X, Y, 4) to 2D dimension:
        shape_reduced = (img.shape[0] * img.shape[1], img.shape[2])
        img_reshape = img.reshape(shape_reduced)
        return [img_reshape, img[:, :, 0].shape]
