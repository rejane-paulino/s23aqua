# -*- mode: python -*-

from s23aqua import S23aqua

# Input Data:
path_MSI = r'/...'
path_OLCI = r'/...'
roi = r'/...'
dest = r'/...'

# Loads the S2/3aqua:
s23aqua = S23aqua(path_MSI, path_OLCI, roi, dest)
s23aqua.run()
