# -*- coding: utf-8 -*-
import tomopy

# Read HDF5 files.
data, theta = tomopy.xftomo_reader('demo/2014-2_jakes3/*.h5',
                            file_start=100,
                            file_end=164,
                            files_exclude=[None],
                            slices_start=0,
                            slices_end=16)

# Xtomo object creation and pipeline of methods.
d = tomopy.xftomo_dataset(log='debug')
d.dataset(data, theta)
d.align_projections()
d.center = 661.5
d.gridrec()


# Write to stack of TIFFs.
tomopy.xftomo_writer(d.data_recon, 'tmp/test_', axis=0)
