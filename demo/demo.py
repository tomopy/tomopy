# -*- coding: utf-8 -*-
import tomopy
import ipdb
# Read HDF5 file.
data, white, dark, theta = tomopy.xtomo_reader('demo/data/data.h5',
                                               slices_start=0,
                                               slices_end=16)

# Xtomo object creation and pipeline of methods.
ipdb.set_trace()
d = tomopy.xtomo_dataset(log='debug')
d.dataset(data, white, dark, theta)
d.normalize()
d.correct_drift()
d.phase_retrieval()
d.correct_drift()
d.center = 661.5
d.gridrec()


# Write to stack of TIFFs.
tomopy.xtomo_writer(d.data_recon, '/tmp/test/test_', axis=0)
