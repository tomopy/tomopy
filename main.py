# -*- coding: utf-8 -*-
# Filename: main.py
""" Main program for tomographic reconstruction.
"""
#import tomoRecon.tomoRecon
from preprocessing.preprocess import Preprocess
from tomoRecon import tomoRecon
from visualize import image
from dataio.file_types import Tiff
import numpy as np

# Input HDF file.
filename = '/local/dgursoy/data/a/H25_01_4_Nymphicus_hollandicus_GC-10-04_P10_tip/H25_01_4.hdf'

# Pre-process data.
mydata = Preprocess()
mydata.read_hdf5(filename, slices_start=40, slices_end=80)
mydata.normalize()
mydata.remove_rings(wname='db10', sigma=2)
#mydata.median_filter()
#mydata.optimize_center(center_init=1400)
mydata.center = 1404.6484375
mydata.retrieve_phase(pixel_size=0.65e-4, dist=40, energy=22.4, delta_over_mu=1e-8)
mydata.data = np.exp(-mydata.data)


# Reconstruct data.
recon = tomoRecon.tomoRecon(mydata)
recon.run(mydata)

# Save data.
f = Tiff()
f.write(recon.data, file_name='/local/dgursoy/GIT/tomopy/data/test_.tiff')

# Visualize data.
image.show_slice(recon.data)
