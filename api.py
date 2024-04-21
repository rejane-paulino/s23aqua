# -*- mode: python -*-

from s23aqua import S23aqua

# Input:
path_MSI = r'/...'
path_OLCI = r'/...'
roi = r'/...'
dest = r'/...'

# Loads the S2/3 products:
a = S23aqua(path_MSI, path_OLCI, roi, dest)
a.run()
