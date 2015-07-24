#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct micro-CT data set.
"""

import tomopy 

# Set path to the micro-CT data to reconstruct.
fname = 'data_dir/sample.h5'

# Select sinogram range to reconstruct.
start = 0
end = 2

# Read APS 32-ID or 2-BM raw data.
proj, flat, dark = tomopy.io.exchange.read_aps_32id(fname, sino=(start, end))

# Set data collection angles as equally spaced between 0-180 degrees.
theta  = tomopy.angles(proj.shape[0], 0, 180)

# Flat-field correction of raw data.
prj = tomopy.normalize(proj, flat, dark)

# Find rotation center.
rot_center = tomopy.find_center(proj, theta, emission=False, ind=0, init=295, tol=0.5)
print "Calculated rotation center: ", rot_center

# Reconstruct object using Gridrec algorithm.
rec = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec', emission=False)

# Mask each reconstructed slice with a circle.
rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

# Set reconstructed images name
rec_name = 'recon_dir/recon'

# Write data as stack of TIFs.
tomopy.io.writer.write_tiff_stack(rec, fname=rec_name)
