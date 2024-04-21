# -*- mode: python -*-

import glob
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
        self.date = self.path_MSI[self.path_MSI.find('S2'):][11:19]
        self.grid = self.path_MSI[self.path_MSI.find('S2'):][38:44]
        self.msiid = self.path_MSI[self.path_MSI.find('S2'):]
        self.olciid = self.path_OLCI[self.path_OLCI.find('S3'):]
        # Geocode - MSI:
        self.geocode = self.get_crs(glob.glob(self.path_MSI + '/' + 'B02*.tif')[0])

    def get_crs(self, raster_path):
        raster = gdal.Open(raster_path)
        return raster.GetProjection()
