# -*- mode: python -*-

import pathlib
from osgeo import gdal
import datetime
import xml.etree.ElementTree as ET


def newdirectory(path: str, name: str) -> str:
    """
    It creates a new directory in the specified path.
    """
    saved_path = path + '/' + name
    pathlib.Path(saved_path).mkdir(parents=True, exist_ok=True)
    return saved_path


def loadarray(path: list, index: list) -> dict:
    """
    Loads the images and returns a dict with arrays.
    """
    output = {}
    for enum, id in enumerate(index):
        dataset = gdal.Open(path[enum])
        output[id] = dataset.ReadAsArray().astype(float)
    return output


def cutbands(path_shapefile: str, path_image: str, index: str, dest: str) -> None:
    """
    Cuts the images.
    """
    kwargs = {'cutlineDSName': 'True', 'dstNodata': 'np.nan', '-to': 'Float32'}
    gdal.Warp(dest + '/' + index + '.tif', path_image,
              cutlineDSName=path_shapefile,
              cropToCutline=True,
              dstNodata=-9999)
    return None


def export(array: float, id: str, reference: str, dest: str) -> None:
    """
    Export the images to dest.
    """
    filename_reference = reference
    filename_out_factor = dest + '/' + id + '.tif'
    dataset_reference = gdal.Open(filename_reference)

    line = dataset_reference.RasterYSize
    column = dataset_reference.RasterXSize
    bands = 1

    # defining drive
    driver = gdal.GetDriverByName('GTiff')
    # copying the bands data type pre-existing
    data_type = gdal.GetDataTypeByName('Float32')
    # create new dataset
    dataset_output = driver.Create(filename_out_factor, column, line, bands, data_type)
    # copying the spatial information pre-existing
    dataset_output.SetGeoTransform(dataset_reference.GetGeoTransform())
    # copying the projection information pre-existing
    dataset_output.SetProjection(dataset_reference.GetProjectionRef())
    # writing array data in band
    dataset_output.GetRasterBand(1).WriteArray(array)
    # solve values
    test = dataset_output.FlushCache()
    # close dataset
    dataset_output = None
    return None


def dict_to_xml(dictionary, parent=None):
    """
    Converts the dict into .xml.
    """
    if parent is None:
        parent = ET.Element('root')

    for key, value in dictionary.items():
        if isinstance(value, dict):
            dict_to_xml(value, ET.SubElement(parent, key))
        else:
            child = ET.SubElement(parent, key)
            child.text = str(value)
    return parent


def meta(meta, dest: str) -> None:
    """
    It writes and exports the metadata as a file .xml.
    """
    current_datetime = datetime.datetime.now()
    image_date = meta.date

    bands = {'Band' + str(i): 'S23AQUA_' + meta.date + '_' + meta.grid + '_' + meta.OLCIBAND[i] for i in range(len(meta.OLCIBAND))}

    metadata_dict = {
        'General_Info': {
            'software': 's23aqua',
            'version': '1',
            'datetime_image': image_date,
            'bandnumber': str(len(meta.OLCIBAND)),
            'bandname': bands
        },
        'OriginalData': {
            'Sentinel-3': meta.olciid,
            'Sentinel-2': meta.msiid
        },
        'InputData': {
            'path_OLCI': meta.path_OLCI,
            'path_MSI': meta.path_MSI,
            'roi': meta.roi
        },
        'Outputdata': {
            'path': meta.dest,
            'pixel_value': 'surface_reflectance',
            'NaN_value': 'nan',
            'file_format': 'TIFF',
            'datetime_processing': current_datetime.isoformat()
        }
    }

    root = ET.Element('root')
    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            dict_to_xml(value, ET.SubElement(root, key))
        else:
            child = ET.SubElement(root, key)
            child.text = str(value)

    tree = ET.ElementTree(root)
    tree.write(dest + '/' + 'MTD.xml', encoding='utf-8', xml_declaration=True)
    return None

