# -*- coding: utf-8 -*-
# Filename: main.py
""" Main program for tomographic reconstruction.
"""
#import tomoRecon.tomoRecon
from preprocessing.preprocess import Preprocess
from tomoRecon import tomoRecon
from visualize import image
from dataio.file_types import Tiff

# Input HDF file.
filename = 'filename.h5'

# Pre-process data.
mydata = Preprocess()
mydata.read_hdf5(filename, slices_start=200, slices_end=201)
mydata.normalize()
mydata.median_filter()
mydata.remove_rings()
mydata.optimize_center()

# Reconstruct data.
recon = tomoRecon.tomoRecon(mydata)
recon.run(mydata)

# Save data.
f = Tiff()
f.write(recon.data, filename='/local/dgursoy/GIT/tomopy/data/test_.tiff')

# Visualize data.
image.showSlice(recon.data)
