# -*- mode: python -*-

import glob
from pathlib import Path
from osgeo import gdal


class Metadata:

    def __init__(self,
                 path_MSI: str,
                 path_OLCI: str,
                 roi: str,
                 dest: str):

        self.OLCI = 'OLCI_S3'
        self.MSI = 'MSI_S2'
        self.OLCIBAND = ['Oa04', 'Oa05', 'Oa06', 'Oa07', 'Oa08', 'Oa09', 'Oa10', 'Oa11']
        self.MSIBAND = ['B02', 'B03', 'B04', 'B05']
        self.nodata = -9999

        self.path_MSI = path_MSI
        self.path_OLCI = path_OLCI
        self.roi = roi
        self.dest = dest
        self.geocode = str('nan')

    def run (self):
        """
        Runs the Metadata.
        """
        # General information from images:
        self.msiid = str(Path(self.path_MSI).parent.name)
        self.olciid = str(Path(self.path_OLCI).parent.name)
        self.date = str(self.msiid[11:19])
        self.grid = (self.msiid[38:44])
        # Geocode - MSI:
        self.geocode = self.get_crs(glob.glob(self.path_MSI + '/' + '*_B02.tif')[0])


    def get_crs(self, raster_path):
        raster = gdal.Open(raster_path)
        return raster.GetProjection()


    def get_nodata(self, raster_path):
        raster = gdal.Open(raster_path)
        return raster.GetRasterBand(1).GetNoDataValue()


