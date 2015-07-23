#!/usr/bin/env python
"""
TomoPy example to reconstruct a micro-CT data set in chuck of sinogrmas. 
This function is for reconstructing large data set on limited memory computers.
"""

import tomopy
import numpy as np
import matplotlib.pylab as pl

# set the path to the micro-CT data set to reconstruct
fname = 'data_dir/sample.h5'

# Select the sinogram range to reconstruct
start = 740; end = 1700

# Set the number of data chunks for the reconstruction
chunks = 10 
nSino_per_chunk = (end - start)/chunks

for iChunk in range(0,chunks):
    print '\n  -- chunk # %i' % (iChunk+1)
    sino_chunk_start = start + nSino_per_chunk*iChunk 
    sino_chunk_end = start + nSino_per_chunk*(iChunk+1)
    print '\n  --------> [%i, %i]' % (sino_chunk_start, sino_chunk_end)

    if sino_chunk_end > end: 
        break

    # Read the APS 32-ID or 2-BM raw data
    prj, flat, dark = tomopy.io.exchange.read_aps_32id(fname, sino=(sino_chunk_start, sino_chunk_end))

    # Set the data collection angles as equally spaced between 0-180 degrees
    theta  = tomopy.angles(prj.shape[0], ang1=0, ang2=180)

    # Normalize the raw projection data
    prj = tomopy.normalize(prj, flat, dark)

    # Reconstruct using gridrec
    best_center = 1232
    rec = tomopy.recon(prj, theta, center=best_center, algorithm='gridrec', emission=False)

    # Set reconstructed images name
    rec_name = 'rec/sample'
    print rec_name

    # Write data as stack of TIFs.
    tomopy.io.writer.write_tiff_stack(rec, fname=rec_name, start=sino_chunk_start)


