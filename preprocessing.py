# -*- mode: python -*-

import os
import glob
import shutil
import numpy as np
from osgeo import gdal
import geopandas as gpd
from scipy.signal import convolve2d

import auxf


class Preprocessing:

    def __init__(self, parameters, sensor):
        self.parameters = parameters
        self.sensor = sensor

    def run(self):
        """
        Runs the pre-processing over OLCI and MSI images.
        """
        if self.sensor == self.parameters.MSI:
            # Creates a new directory:
            pathxresample = auxf.newdirectory(self.parameters.dest + '/temp', 'resampled')
            pathxsmoothed = auxf.newdirectory(self.parameters.dest + '/temp', 'smoothedMSI')
            pathxshapefiles = auxf.newdirectory(self.parameters.dest + '/temp', 'shapefiles')
            # Resamples the spectral bands from 20 m to 10 m:
            self.resample(self.parameters.path_MSI, pathxresample, self.parameters.nodata)
            # Reads the arrays:
            paths = [os.path.normpath(band) for band in glob.glob(os.path.join(pathxresample, '*.tif')) if 'B02' in band or 'B03' in band or 'B04' in band or 'B05' in band or 'B12' in band]
            paths.sort()
            array = auxf.loadarray(paths, self.parameters.MSIBAND + ['B12'])
            # WaterMask:
            watermask = self.loadwmask(array)
            auxf.export(watermask, 'watermask', paths[0], pathxresample)
            # Cutting the bands:
            self.boudingbox(self.parameters.roi, pathxshapefiles, self.parameters.geocode)
            paths = [os.path.normpath(band) for band in glob.glob(os.path.join(pathxresample, '*.tif')) if 'B02' in band or 'B03' in band or 'B04' in band or 'B05' in band or 'watermask' in band]
            paths.sort()
            for i in paths:
                auxf.cutbands(pathxshapefiles + '/roi_clip.shp', i, i.split('\\')[-1][:-4], pathxresample)
            array = auxf.loadarray(paths, self.parameters.MSIBAND + ['watermask'])
            # Fills in the NoData values:
            for i in array:
                array[i] = np.where(array[i] > self.parameters.nodata, array[i], 0) # Fill in the NaNData
            # PSF - Punctual Scattering Function (s: 31-by-31 pixels, dev. along and cross track 3 pixels)
            psf = self.GaussianKernel(31, 3)
            # Resamples the image based on PSF:
            for num, i in enumerate(self.parameters.MSIBAND):
                smoothing = self.Conv2D(array[i], psf)
                smoothing_mask = smoothing * array['watermask']
                out = np.where(smoothing_mask != 0, smoothing_mask, self.parameters.nodata)
                # Exports bands:
                auxf.export(out, i, paths[num], pathxsmoothed)
        else:
            # Creates a new directory:
            pathxreprojectedOLCI = auxf.newdirectory(self.parameters.dest + '/temp', 'reprojectedOLCI')
            paths = [os.path.normpath(band) for band in glob.glob(os.path.join(self.parameters.path_OLCI, '*.TIF')) if 'Oa04' in band or 'Oa05' in band or 'Oa06' in band or 'Oa07' in band or 'Oa08' in band
                     or 'Oa09' in band or 'Oa10' in band or 'Oa11' in band]
            paths.sort()
            for i, j in zip(paths, self.parameters.OLCIBAND):
                self.reproject_images(i, self.parameters.geocode, j, pathxreprojectedOLCI)


    def resample(self, path: str, dest: str, nodata):
        """
        It resamples the pixel size to 10 m.
        """
        for band in os.listdir(path):
            if 'B05' in band or 'B12' in band:
                # Resample pixel -> 10 m:
                filename = path + '/' + band
                try:
                    dataset = gdal.Open(filename)
                    if dataset is not None:
                        gdal.Warp(dest + '/' + band[:-4] + '.tif', filename, xRes=10, yRes=10, resampleAlg='bilinear', srcNodata=nodata)
                except:
                    pass
            else:
                # It copies the other bands:
                shutil.copy2(os.path.join(path, band), dest)
        return None


    def loadwmask(self, array: dict):
        """
        Returns the water mask.
        """
        mndwi = (array['B03'] - array['B12']) / (array['B03'] + array['B12'])
        return np.where(mndwi >= .05, 1, 0) # default --threshold equal to 0.05.


    def GaussianKernel(self, sizeXYkernel: int, std: int):
        """
        Creates the PSF.
        """
        # Creates the grids:
        x, y = np.mgrid[-sizeXYkernel // 2 + 1:sizeXYkernel // 2 + 1, -sizeXYkernel // 2 + 1:sizeXYkernel // 2 + 1]
        # Recovers the gaussian kernel --2D:
        XY = (x ** 2) + (y ** 2)
        exp_ = np.exp(-(XY / (2 * (std ** 2))))
        GaussianKernel = (1 / (2 * np.pi * (std ** 2))) * exp_
        return GaussianKernel / np.sum(GaussianKernel) # Normalization process.


    def Conv2D(self, array, kernel):
        """
        Applies the PSF on the image.
        """
        smooth = convolve2d(in1=array, in2=kernel, mode='same', boundary='fill', fillvalue=0)
        ''' Notice that: 
            - mode == same -> takes account 'filling' of the input array borders, that is, the output array shape is not reduced;
            - boundary == fill -> fills the input array with fillvalue;
            - fillvalue == 0 -> value to the filling. '''
        return smooth


    def reproject_images (self, path: str, epsg_string: str, index: str, dest: str):
        """
        Re-projects the OLCI images to MSI projection.
        """
        input_raster = gdal.Open(path)
        output_raster = dest + '/' + index + '.tif'
        gdal.Warp(output_raster, input_raster, dstSRS=epsg_string)
        return None


    def boudingbox (self, roi: str, dest: str, crs: str) -> None:
        """
        Generates a buffer of 2.5 km from reservoir edge.
        """
        geometry = gpd.read_file(roi)
        gdf = geometry.to_crs(crs)  # geotransform.
        boundingbox = gdf['geometry'].envelope
        buffered = boundingbox.buffer(2500, join_style=2)
        buffered.to_file(dest + '/roi_clip.shp')
        return None
