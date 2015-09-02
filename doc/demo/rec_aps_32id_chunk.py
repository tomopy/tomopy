#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct TXM data iteratively
in chuck of sinogrmas. This function is for reconstructing large 
data on limited memory computers.
"""

from __future__ import print_function
import tomopy

if __name__ == '__main__':

    # Set path to the micro-CT data to reconstruct
    fname = 'data_dir/sample.h5'

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
        proj, flat, dark = tomopy.io.exchange.read_aps_32id(fname, sino=(sino_start, sino_end))

        # Set data collection angles as equally spaced between 0-180 degrees.
        theta  = tomopy.angles(proj.shape[0])

        # Flat-field correction of raw data.
        proj = tomopy.normalize(proj, flat, dark)

        # Reconstruct object using Gridrec algorithm.
        rec = tomopy.recon(proj, theta, center=1024, algorithm='gridrec', emission=False)

        # Write data as stack of TIFs.
        tomopy.io.writer.write_tiff_stack(rec, fname='recon_dir/recon', start=sino_start)
