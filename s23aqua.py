# -*- mode: python -*-

import shutil
from tqdm import tqdm

import auxf
from metadata import Metadata
from preprocessing import Preprocessing
from sampling.samples import Samples
from sampling.LUT import LUT
from training import Training
from predict import Predict


class S23aqua:

    def __init__(self, path_MSI: str, path_OLCI: str, roi: str, dest: str):
        self.path_MSI = path_MSI
        self.path_OLCI = path_OLCI
        self.roi = roi
        self.dest = dest

    def run(self):
        """
        Returns the Sentinel-2/3 Synthetic Aquatic Reflectance Bands.
        """
        # Metadata stage:
        meta = Metadata(self.path_MSI, self.path_OLCI, self.roi, self.dest)
        meta.run()
        # New dicts:
        meta.dest = auxf.newdirectory(meta.dest, 'S23AQUA_' + meta.date + '_' + meta.grid)
        tempdir = auxf.newdirectory(meta.dest, 'temp')
        process = [Preprocessing(meta, 'MSI_S2'), Preprocessing(meta, 'OLCI_S3'),
                   Samples(meta), LUT(meta, 'MSI_S2'), LUT(meta, 'OLCI_S3'), Training(meta),
                   Predict(meta)]
        # Progress:
        with tqdm(total=len(process), desc='S23AQUA_' + meta.date + '_' + meta.grid) as pbar:
            for i in process:
                i.run()
                pbar.update(1)
        # Exporting the MTD:
        auxf.meta(meta, meta.dest)
        # Removes the tempdir:
        shutil.rmtree(tempdir)




