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
filename = '/local/dgursoy/data/360data/ceramic_60keV_50mm_7p5X_360deg_1.hdf'

# Pre-process data.
mydata = Preprocess()
mydata.read_hdf5(filename, slices_start=700, slices_end=701)
#mydata.normalize()
#mydata.median_filter()
mydata.correct_view()
mydata.center = 1824.64355469

#mydata.remove_rings(wname='db20', sigma=3)


# Reconstruct data.
recon = tomoRecon.tomoRecon(mydata)
recon.run(mydata)

# Save data.
f = Tiff()
f.write(recon.data, filename='/local/dgursoy/GIT/tomopy/data/test_.tiff')

# Visualize data.
#image.show_slice(mydata.data[:, 0, :])
