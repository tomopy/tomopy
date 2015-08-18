#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct the Elettra syrmep data as original tiff.
"""

import tomopy

if __name__ == '__main__':

    # Set path to the CT data to reconstruct.
    fname = 'data_dir/'

    proj_start = 1
    proj_end = 1801
    flat_start = 1
    flat_end = 11
    dark_start = 1
    dark_end = 11

    ind_tomo = range(proj_start, proj_end)
    ind_flat = range(flat_start, flat_end)
    ind_dark = range(dark_start, dark_end)

    # Select the sinogram range to reconstruct.
    start = 0
    end = 16

    # Read the Elettra syrmep
    proj, flat, dark = tomopy.read_elettra_syrmep(fname, ind_tomo, ind_flat, ind_dark, sino=(start, end))

    # Set data collection angles as equally spaced between 0-180 degrees.
    theta  = tomopy.angles(proj.shape[0], 0, 180)

    # Flat-field correction of raw data.
    proj = tomopy.normalize(proj, flat, dark)

    # Find rotation center.
    rot_center = tomopy.find_center(proj, theta, emission=False, init=1024, ind=0, tol=0.5)
    print("Center of rotation: ", rot_center)

    # Reconstruct object using Gridrec algorithm.
    rec = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec', emission=False)

    # Mask each reconstructed slice with a circle.
    rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

    # Write data as stack of TIFs.
    tomopy.write_tiff_stack(rec, fname='recon_dir/recon')
