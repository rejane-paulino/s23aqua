# -*- mode: python -*-

import pandas as pd
import geopandas as gpd
import os
import glob
import rasterio
import numpy as np

import auxf

class LUT:

    def __init__(self, parameters, sensor):
        self.parameters = parameters
        self.sensor = sensor

    def run(self):
        """
        Runs the LUT.
        """
        # Creating new directories:
        pathxLUT = auxf.newdirectory(self.parameters.dest + '/temp', 'LUT')
        # LUT with data obtained by image:
        coor = self.coordinates(self.parameters.dest + '/temp/shapefiles/pointsxtrain.shp')
        # Considering the MSI:
        if self.sensor == self.parameters.MSI:
            paths = [band for band in glob.glob(os.path.join(self.parameters.dest + '/temp/smoothedMSI', '*.tif')) if 'B02' in band or 'B03' in band or 'B04' in band or 'B05' in band]
            images = [rasterio.open(i) for i in paths] # access to images.
            data = []
            for point_coors, point_id in zip(coor[0], coor[1]):
                value = self.values(images, point_coors)
                data.append(self.merge(value, point_id, self.parameters.MSIBAND, self.parameters.MSI))
            data = pd.concat(data)
            valuesperband = []
            for i in self.parameters.MSIBAND:
                filterband = data.loc[data['band'] == i]
                valuesperband.append(filterband.filter(['value']).reset_index().rename(columns={'value': i}))
            output = [valuesperband[0].iloc[:, 1:]]
            for j in valuesperband[1:]:
                output.append(output[-1].merge(j.iloc[:, 1:], left_index=True, right_index=True))
            output[-1].reset_index().iloc[:, 1:].to_csv(pathxLUT + '/MSI_LUT.csv', sep=',')
        else:
            # Considering the OLCI:
            paths = [band for band in glob.glob(os.path.join(self.parameters.dest + '/temp/reprojectedOLCI', '*.tif')) if 'Oa04' in band or 'Oa05' in band or 'Oa06' in band or 'Oa07' in band or 'Oa08' in band
                     or 'Oa09' in band or 'Oa10' in band or 'Oa11' in band]
            images = [rasterio.open(i) for i in paths] # access to images.
            data = []
            for point_coors, point_id in zip(coor[0], coor[1]):
                value = self.values(images, point_coors)
                data.append(self.merge(value, point_id, self.parameters.OLCIBAND, self.parameters.OLCI))
            data = pd.concat(data)
            valuesperband = []
            for i in self.parameters.OLCIBAND:
                filterband = data.loc[data['band'] == i]
                valuesperband.append(filterband.filter(['value']).reset_index().rename(columns={'value': i}))
            output = [valuesperband[0].iloc[:, 1:]]
            for j in valuesperband[1:]:
                output.append(output[-1].merge(j.iloc[:, 1:], left_index=True, right_index=True))
            output[-1].reset_index().iloc[:, 1:].to_csv(pathxLUT + '/OLCI_LUT.csv', sep=',')

    def coordinates(self, grid_points: str):
        """
        Retrievals the point coordinates (x, y).
        """
        # Output:
        coor = []
        point_id = []
        # Coordinates and point id:
        data = gpd.read_file(grid_points)
        for i in data.index:
            point = data.loc[data.index == int(i)]
            coor.append(point['geometry'])
            point_id.append(i)
        return [coor, point_id]

    def extract(self, image, point_coors):
        """
        Extracts the values from image pixels.
        """
        x = np.round((point_coors.x), 5)
        y = np.round((point_coors.y), 5)
        row, col = image.index(x, y)
        return image.read(1)[row, col]

    def values(self, paths: list, point_coors) -> list:
        """
        Reflectance values extracted per band.
        """
        output = []
        for i in paths:
            extract_point_value = self.extract(i, point_coors)
            output.append(pd.DataFrame(extract_point_value).transpose())
        return output

    def merge(self, values, point: str, index: list, sensor):
        """
        Merges the values in a row.
        """
        trans = []
        for i in range(0, len(index)):
            trans.append(values[i].transpose().set_axis([str(index[i])], axis='columns'))
        output = [trans[0]]
        for j in range(1, len(trans)):
            output.append(output[-1].merge(trans[j], left_index=True, right_index=True))
        # Includes some extra information:
        data = output[-1].transpose().set_axis(['value'], axis='columns')
        data.insert(1, 'band', data.index)
        data.insert(2, 'point', point)
        data.insert(3, 'sensor', sensor)
        return data.set_index(pd.DataFrame({'id': [x for x in len(index) * [0, ]]})['id'])
