#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct a TXM data containing 
a series of useless projections becuase of the presence of an 
environment cell blocking some of the sample views.
"""
 
import tomopy
import numpy as np

if __name__ == '__main__':

    # Set path to the micro-CT data set to reconstruct.
    fname = 'data_dir/sample.h5'

    # Set the [start, end] index of the blocked projections.
    miss_projs = [128, 256]

    # Select sinogram range to reconstruct.
    start = 512
    end = 2048

    # Set number of data chunks for the reconstruction.
    chunks = 64
    num_sino = (end - start) // chunks

    for m in range(chunks):
        sino_start = start + num_sino * m 
        sino_end = start + num_sino * (m + 1)

        # Read APS 32-ID raw data.
        proj, flat, dark = tomopy.read_aps_32id(fname, sino=(sino_start, sino_end))

        # Set data collection angles as equally spaced between 0-180 degrees.
        theta  = tomopy.angles(proj.shape[0])

        # Remove the missing angles from data.
        proj = np.concatenate((proj[0:miss_projs[0], :, :], proj[miss_projs[1] + 1:-1, :, :]), axis=0)
        theta = np.concatenate((theta[0:miss_projs[0]], theta[miss_projs[1] + 1:-1]))

        # Flat-field correction of raw data.
        proj = tomopy.normalize(proj, flat, dark)

        # Reconstruct object using Gridrec algorithm.
        rec = tomopy.recon(proj, theta, center=1024, algorithm='gridrec', emission=False)

        # Write data as stack of TIFs.
        tomopy.write_tiff_stack(rec, fname='recon_dir/recon', start=sino_start)
    
