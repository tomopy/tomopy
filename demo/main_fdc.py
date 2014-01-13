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
#filename = '/local/data/databank/dataExchange/microCT/Blakely_ALS_2011.h5'
#filename = '/local/data/databank/dataExchange/microCT/PetraIII_ct4_180_new.h5'
#filename = '/local/data/databank/dataExchange/microCT/CAT4B_2_new.h5'
#filename = '/local/data/databank/dataExchange/microCT/Blakely_SLS_2011_new_convert_series_of_images.h5'
#filename = '/local/data/databank/dataExchange/microCT/Hornby_ALS_2011_new_series_of_images.h5'
filename = '/local/data/databank/dataExchange/microCT/Hornby_APS_2011_new_convert_series_of_images.h5'

# Pre-process data.
mydata = Preprocess()
mydata.read_hdf5(filename, slices_start=1200, slices_end=1201)

mydata.normalize()

#mydata.remove_rings(level=12, wname='db10', sigma=2)

mydata.median_filter()
#mydata.optimize_center(center_init=1047)
#mydata.optimize_center()

#mydata.center = 1684
#mydata.center = 2177.00
#mydata.center = 1023.6
#mydata.center = 1047.6
#mydata.center = 1330
mydata.center = 1023.2

##### PetraIII_ct4_180
##mydata.retrieve_phase(pixel_size=1.40e-5, dist=10, energy=15.0, delta_over_mu=1e-8)
##mydata.data = np.exp(-mydata.data)

### CAT4B_2
##mydata.retrieve_phase(pixel_size=1.00e-4, dist=2, energy=64.00, delta_over_mu=1e-8)
##mydata.data = np.exp(-mydata.data)

# Reconstruct data.
recon = tomoRecon.tomoRecon(mydata)
recon.run(mydata)

### Save data.
##f = Tiff()
##f.write(recon.data, file_name='/local/data/databank/PetraIII/rec_ct4.tiff')

# Visualize data.
image.show_slice(recon.data)
