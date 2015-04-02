#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tomopy.prep import *
from tomopy.io import *
from tomopy.io.recipes import *
from tomopy.io.data import *
from tomopy.io.phantom import *
from tomopy.recon import *


fname = '/home/oxygen/DGURSOY/Data/APS2BM/xpcdata.h5'
data, white, dark = read_aps2bm(fname)

# data = normalize(data, white, dark, cutoff=0.8) # X
# data = stripe_removal(data, level=None, wname='db5', sigma=2, pad=None)
# data = phase_retrieval(data, psize=1e-4, dist=50, energy=20, alpha=1e-4, pad=True)
# data = circular_roi(data, ratio=1, val=2) 
# roi, center = focus_region(data, xcoord=0, ycoord=0, dia=1000, center=None, pad=True, corr=True)
# data = median_filter(data, size=10)
# data = zinger_removal(data, dif=1000, size=3)
# data = correct_air(data, air=10, ind=None)
# padded = apply_padding(data, npad=None, val=0.)
# downdat = downsample2d(data, level=2)
# downdat = downsample3d(data, level=2)

# data = lena()
theta = np.linspace(0, np.pi, 60)
# sdata = simulate(data, theta)
art(data, theta, delta=34256, beta=123, niter=2)



# write_tiff_stack(sdata, 'tmp/test', axis=1, dtype='float32', overwrite=True)



