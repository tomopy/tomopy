#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tomopy import prep
from tomopy.io import data
import tomopy.multiprocess as mp
import h5py

# arr = phantom.shepp3d((256, 256, 256))
# arr = 3*np.ones((2, 2))
fname = '/home/oxygen/DGURSOY/Data/APS2BM/xpcdata.h5'
arr = data.read_hdf5(fname)

arr = mp.distribute_jobs(
		arr, func=prep.median_filter, 
		args=(10, 0), axis=0)

arr = mp.distribute_jobs(
		arr, func=prep.phase_retrieval, 
		args=(1.2e-4, 60, 18, 1e-3, False), axis=0)

data.write_tiff_stack(arr, 'tmp1/test', axis=1, overwrite=True)



