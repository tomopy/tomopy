#!/usr/bin/env python

"""
TomoPy example to reconstruct a micro-CT data set.
"""

import tomopy 

def rec_aps_32id_full():

    # set the path to the micro-CT data set ro reconstruct
    fname = 'data_dir/sample.h5'

    # Select the sinogram range to reconstruct
    start = 0; end = 2    

    # Read the APS 32-ID or 2-BM raw data
    prj, flat, dark = tomopy.io.exchange.read_aps_32id(fname, sino=(start, end))

    # Set the data collection angles as equally spaced between 0-180 degrees
    theta  = tomopy.angles(prj.shape[0], ang1=0, ang2=180)

    # Normalize the raw projection data
    prj = tomopy.normalize(prj, flat, dark)

    # Set the aprox rotation axis location.
    # This parameter is the starting angle for auto centering routine
    start_center=295 
    print "Start Center: ", start_center

    # Auto centering
    calc_center = tomopy.find_center(prj, theta, emission=False, ind=0, init=start_center, tol=0.3)
    print "Calculated Center:", calc_center

    # recon using gridrec
    rec = tomopy.recon(prj, theta, center=calc_center, algorithm='gridrec', emission=False)

    # Mask each reconstructed slice with a circle
    rec = tomopy.circ_mask(rec, axis=0, ratio=0.8)

    # to save the reconstructed images uncomment and customize the following line:
    rec_name = 'rec/tooth'

    # Write data as stack of TIFs.
    tomopy.io.writer.write_tiff_stack(rec, fname=rec_name)
    print "Done!  reconstructions at: ", rec_name

if __name__ == "__main__":
    rec_aps_32id_full()
