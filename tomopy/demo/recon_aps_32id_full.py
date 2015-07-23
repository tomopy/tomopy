#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
TomoPy example to reconstruct a micro-CT data set.
"""

import os.path
import tomopy 


def recon_aps_32id_full():
    """

    """

    # set the path to the example file tooth.h5
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    fname = os.path.join(DATA_PATH, 'tooth.h5')
    print fname

    # set the path to the micro-CT data set
    #fname = 'data_dir/sample.h5'

    # Read the APS 32-ID or 2-BM raw data
    start = 0    
    end = 2    
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

def _main():

    recon_aps_32id_full()

if __name__ == "__main__":
    _main()


