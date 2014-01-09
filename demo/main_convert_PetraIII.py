# -*- coding: utf-8 -*-
# Filename: main_convert_PetraIII.py
""" Main program for convert Petra III microCT data into dataExchange.
"""
from preprocessing.preprocess import Preprocess
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry

from dataio.file_types import Tiff
from tomoRecon import tomoRecon
from visualize import image

import matplotlib.pyplot as plt
import numpy as np

import os
import h5py

import re

#def main():

### ct2: pj: from 0 -> 3600; bf from 0 -> 20; df from 0 -> 20
##file_name = '/local/data/databank/PetraIII/ct2/ct2_.tif'
##file_name_dark = '/local/data/databank/PetraIII/ct2/df2b_.tif'
##file_name_white = '/local/data/databank/PetraIII/ct2/bf2b_.tif'
##
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct2_test_angles2.h5'
##
##projections_start = 0
##projections_end = 3601
##white_start = 0
##white_end = 20
##white_step = 1
##dark_start = 0
##dark_end = 20
##dark_step = 1

# ct3: pj: from 0 -> 3601; bf from 20 -> 39; df from 0 -> 19
##file_name = '/local/data/databank/PetraIII/ct3/ct3_.tif'
##file_name_dark = '/local/data/databank/PetraIII/ct3/df_.tif'
##file_name_white = '/local/data/databank/PetraIII/ct3/bf_.tif'
##
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct3_test_angles2.h5'
##
##projections_start = 0
##projections_end = 3601
##white_start = 20
##white_end = 40
##white_step = 1
##dark_start = 0
##dark_end = 20
##dark_step = 1

# ct4: pj: from 0 -> 1199; bf from 1 -> 18; df from 0 -> 19
file_name = '/local/data/databank/PetraIII/ct4/ct4_.tif'
file_name_dark = '/local/data/databank/PetraIII/ct4/df_ct4_.tif'
file_name_white = '/local/data/databank/PetraIII/ct4/bf_ct4_.tif'

hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct4.h5'

projections_start = 0
projections_end = 1201
white_start = 1
white_end = 19
white_step = 1
dark_start = 0
dark_end = 20
dark_step = 1

verbose = True

if verbose: print file_name

z = np.arange(projections_end - projections_start);
if verbose: print z, len(z)
    
# Fabricate theta values
theta = (z * float(180) / (len(z) - 1))
if verbose: print theta

mydata = Preprocess()

mydata.read_tiff(file_name,
                 projections_start,
                 projections_end,
                 file_name_white = file_name_white,
                 white_start = white_start,
                 white_end = white_end,
                 white_step = white_step,
                 file_name_dark = file_name_dark,
                 dark_start = dark_start,
                 dark_end = dark_end,
                 dark_step = dark_step,
                 digits=5,
                 zeros = True
                 )


#Write HDF5 file.

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='w') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'Petra III'}) )

# Create HDF5 subgroup
# /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': 'Petra III'},
                                    date_time={'value': "2011-25-05T19:42:13+0100"},
                                    beamline={'value': "BM05"},
                                    )
)

# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Walter Schroeder"},
                                            role={'value':"Project PI"},
                )
    )

# /measurement/sample
f.add_entry( DataExchangeEntry.sample( name={'value':'ct2'},
                                        description={'value':'freeze-dried plant'},
        )
    )

# Create core HDF5 dataset in exchange group for 180 deep stack
# of x,y images /exchange/data
f.add_entry( DataExchangeEntry.data(data={'value': mydata.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x' }))
f.add_entry( DataExchangeEntry.data(theta={'value': theta, 'units':'degrees'}))
f.add_entry( DataExchangeEntry.data(data_dark={'value': mydata.dark, 'units':'counts', 'axes':'theta_dark:y:x' }))
f.add_entry( DataExchangeEntry.data(data_white={'value': mydata.white, 'units':'counts', 'axes':'theta_white:y:x' }))
f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))

f.close()

###if __name__ == "__main__":
###    main()

