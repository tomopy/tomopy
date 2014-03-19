# -*- coding: utf-8 -*-
import tomopy

data, white, dark, theta = tomopy.xtomo_reader('demo/data.h5',
                                               slices_start=0,
                                               slices_end=16)

d = tomopy.xtomo_dataset(log='debug')
d.dataset(data, white, dark, theta)
d.normalize()
d.correct_drift()
d.phase_retrieval()
d.correct_drift()
d.center=661.5
d.gridrec()



tomopy.xtomo_writer(d.data_recon, 'tmp/test_', axis=0, overwrite=True)


