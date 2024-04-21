# -*- mode: python -*-

import os
from osgeo import osr, gdal, ogr
gdal.UseExceptions()
import pandas as pd
import numpy as np
import glob
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans

import auxf


class Samples:

    def __init__(self, parameters):
        self.parameters = parameters

    def run(self):
        """
        Runs the random sampling process.
        """
        # Creating new directories:
        pathxcut = auxf.newdirectory(self.parameters.dest + '/temp', 'smoothed_cutMSI')
        pathxcluster = auxf.newdirectory(self.parameters.dest + '/temp', 'kmeans')
        # Creating a buffer -300-meter:
        buffer_300m = self.buffer(self.parameters.roi, self.parameters.dest + '/temp/shapefiles', self.parameters.geocode)
        # Cutting the bands:
        paths = [band for band in glob.glob(os.path.join(self.parameters.dest + '/temp/resampled', '*.tif')) if 'B02' in band or 'B03' in band or 'B04' in band or 'B05' in band or 'watermask' in band]
        for i, j in zip(paths, self.parameters.MSIBAND + ['watermask']):
            auxf.cutbands(self.parameters.dest + '/temp/shapefiles/roi_buffer.shp', i, j, pathxcut)
        # Loading the bands:
        array = auxf.loadarray([band for band in glob.glob(os.path.join(pathxcut, '*.tif'))], self.parameters.MSIBAND + ['watermask'])
        # Generating the clusters:
        self.kmeanscluster(array, pathxcut + '/B02.tif', pathxcluster)
        # Converting raster to shapefile:
        self.rasterToshapefile(pathxcluster + '/kmeans.tif', self.parameters.geocode, self.parameters.dest + '/temp/shapefiles')
        # Creating a grid of points:
        grid_points = self.gridpoints(buffer_300m, self.parameters.geocode, self.parameters.dest + '/temp/shapefiles')
        # Selecting points per class and validation and training points:
        self.randompointsintoclasters(grid_points, self.parameters.dest + '/temp/shapefiles/kmeans.shp', buffer_300m, self.parameters.geocode, self.parameters.dest + '/temp/shapefiles')

    def buffer(self, roi: str, dest: str, crs):
        """
        Generates a buffer of -300 m from reservoir edge.
        """
        geometry = gpd.read_file(roi)
        gdf = geometry.to_crs(crs)  # geotransform.
        buffered = gdf.buffer(-300) # buffer distance: -300-meters
        buffered.to_file(dest + '/roi_buffer.shp')
        return buffered

    def kmeanscluster(self, array: dict, reference: str, dest: str):
        """
        Generates clusters by k-means.
        """
        # Applies the water masker over the images:
        maskedarray = {}
        for i in array:
            if 'watermask' not in i:
                wmasked = array[i] * array['watermask']
                maskedarray[i] = wmasked * np.where(array['B03'] < 0, 0, 1) # used to mask nan pixels inside of waterbody. It is necessary when the image does not cover all the waterbody.
        # Stacks the bands:
        x, y = maskedarray['B02'].shape
        img = np.zeros((x, y, 4))
        img[:, :, 0] = maskedarray['B02']
        img[:, :, 1] = maskedarray['B03']
        img[:, :, 2] = maskedarray['B04']
        img[:, :, 3] = maskedarray['B05']
        # Reduces the array (X, Y, 5) to 2D dimension:
        shape_reduced = (img.shape[0] * img.shape[1], img.shape[2])
        img_reshape = img.reshape(shape_reduced)
        # k-means:
        k = 11 # clusters number # default.
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=150)
        kmeans.fit(img_reshape)
        # Gets the labels and centroids:
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        # Reshapes the labels back into the shape of the original image:
        segmented_image = labels.reshape(img[:, :, 0].shape)
        # Save:
        auxf.export(segmented_image, 'kmeans', reference, dest)
        return None

    def rasterToshapefile(self, raster, crs_string: str, dest: str):
        """
        Converts raster (.TIFF) to shapefile (.shp).
        """
        src_ds = gdal.Open(raster)
        srcband = src_ds.GetRasterBand(1)
        dst_layername = 'data'
        drv = ogr.GetDriverByName("ESRI Shapefile")
        dst_ds = drv.CreateDataSource(dest + '/kmeans.shp')
        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(crs_string)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs=sp_ref)
        fld = ogr.FieldDefn("data", ogr.OFTInteger)
        dst_layer.CreateField(fld)
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex("data")
        gdal.Polygonize(srcband, None, dst_layer, dst_field, [], callback=None)
        return None

    def gridpoints(self, buffer, crs_string, dest: str):
        """
        Generates a grid of points.
        """
        # Polygon:
        polygon = buffer.geometry.iloc[0]
        # Defining a fixed distance among points:
        min_distance = 300 # 1 * pixel_size # default.
        # Calculates the bounding box from polygon:
        min_x, min_y, max_x, max_y = polygon.bounds
        # Creates a grid of points with fixed spacing within the bounding box
        x_values = np.arange(min_x, max_x, min_distance)
        y_values = np.arange(min_y, max_y, min_distance)
        # Creates the dataframe:
        grid_points = [(x, y) for x in x_values for y in y_values]
        geometry = [Point(x, y) for x, y in grid_points]
        gridpoints = gpd.GeoDataFrame(geometry=geometry, crs=crs_string).reset_index(drop=True)
        gridpoints.to_file(dest + '/gridpoint.shp')
        return gridpoints

    def randompointsintoclasters(self, gridpoints, shapefile_clusters, buffer, epsg_string: str, dest: str):
        """
        Classifies the points in each cluster and separates grids of training and validation (50%-by-50%).
        """
        # Filters the geometry and merge:
        shapefile_clusters = gpd.read_file(shapefile_clusters)
        generated_points = []
        for i in range(1, 11): # 0 is equal to values NaN.
            filtergeo = shapefile_clusters.loc[shapefile_clusters["data"] == i]
            polygons_list = []
            for j in filtergeo.geometry:
                polygons_list.append(j)
            geo = gpd.GeoDataFrame(geometry=polygons_list, crs=epsg_string)
            clipped_points = gpd.clip(gridpoints, geo)
            clipped_points.insert(0, 'cluster', i)
            generated_points.append(clipped_points)
        out = pd.concat(generated_points)
        # Selects points to validate and train the models.
        # The number of samples in each reservoir was defined based on water body size.
        # For water body area (km2) > 100 km2 only 25% of total samples are considered in the train + validate of the models.
        # This process prevents a super-adjust of the models to train set. It maintains the proportion among classes.
        xarea = buffer.geometry.iloc[0].area * 0.000001
        if float(xarea) > 100: # in km2
            out = out.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=0.25))
            train = out.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=0.5))
            validation = pd.concat([out, train]).drop_duplicates(keep=False)
        else:
            train = out.groupby('cluster', group_keys=False).apply(lambda x: x.sample(frac=0.5))
            validation = pd.concat([out, train]).drop_duplicates(keep=False)
        # Save as shapefile:
        train.to_file(dest + '/pointsxtrain.shp')
        validation.to_file(dest + '/pointsxvalidation.shp')


# from metadata import Metadata


# path_MSI = r'/Volumes/rsp/s23aqua_validation/S2A_MSIL1C_20210809T131251_N0500_R138_T23KLP_20230116T033833.SAFE'
# path_OLCI = r'/Volumes/rsp/s23aqua_validation/S3A_OL_1_EFR____20210809T121855_20210809T122155_20210810T171811_0179_075_066_3420_LN1_O_NT_002.SEN3'
# roi = r'/Volumes/rsp/s23aqua_validation/ROI/roi_billings_4326.shp'
# dest = r'/Volumes/rsp/s23aqua_validation/out'
#
# a = Metadata(path_MSI, path_OLCI, roi, dest)
# a.run()
#
# b = SAMPLES(a)
# b.run()