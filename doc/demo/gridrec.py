#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct the tomography data as
with gridrec.
"""
from __future__ import print_function
import tomopy
import dxchange

if __name__ == '__main__':

    # Set path to the micro-CT data to reconstruct.
    fname = '../../../tomopy/data/tooth.h5'

    # Select the sinogram range to reconstruct.
    start = 0
    end = 2

    # Read the APS 2-BM 0r 32-ID raw data.
    proj, flat, dark = dxchange.read_aps_32id(fname, sino=(start, end))

    # Set data collection angles as equally spaced between 0-180 degrees.
    theta = tomopy.angles(proj.shape[0])

    # Set data collection angles as equally spaced between 0-180 degrees.
    proj = tomopy.normalize(proj, flat, dark)

    # Set data collection angles as equally spaced between 0-180 degrees.
    rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)

    proj = tomopy.minus_log(proj)

    # Reconstruct object using Gridrec algorithm.
    recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')

    # Mask each reconstructed slice with a circle.
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

    # Write data as stack of TIFs.
    dxchange.write_tiff_stack(recon, fname='recon_dir/recon')
