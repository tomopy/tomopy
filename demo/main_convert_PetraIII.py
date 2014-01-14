# -*- coding: utf-8 -*-
# Filename: ALS_Converter.py
""" Main program for convert ALS data into dataExchange.
"""
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry
from dataio.data_convert import Convert

import re

#def main():


# Petra III collects data over 360deg but in this data sets they had problem with the rotary
# stage stop moving . This happened after 180 deg so picking the first 180 deg are good to reconstruct.
# The 3 blocks below load only the good 180 deg

### ct2: pj: from 0 -> 3600; bf from 0 -> 20; df from 0 -> 20
##file_name = '/local/data/databank/PetraIII/ct2/ct2_.tif'
##dark_file_name = '/local/data/databank/PetraIII/ct2/df2b_.tif'
##white_file_name = '/local/data/databank/PetraIII/ct2/bf2b_.tif'
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct2_180.h5'
##sample_name = 'ct2'
##
### ct2: Wheat root
### Sample measured at room temperature
##
##projections_start = 0
##projections_end = 1801
##white_start = 0
##white_end = 20
##white_step = 1
##dark_start = 0
##dark_end = 20
##dark_step = 1

### ct3: pj: from 0 -> 3601; bf from 20 -> 39; df from 0 -> 19
##file_name = '/local/data/databank/PetraIII/ct3/ct3_.tif'
##dark_file_name = '/local/data/databank/PetraIII/ct3/df_.tif'
##white_file_name = '/local/data/databank/PetraIII/ct3/bf_.tif'
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct3_180.h5'
##sample_name = 'ct3'
##
### ct3: Wheat root
### Same sample as ct3 but measured at cryogenic condition
##
##projections_start = 0
##projections_end = 1801
##white_start = 20
##white_end = 40
##white_step = 1
##dark_start = 0
##dark_end = 20
##dark_step = 1

# ct4: pj: from 0 -> 1199; bf from 1 -> 18; df from 0 -> 19
file_name = '/local/data/databank/PetraIII/ct4/ct4_.tif'
dark_file_name = '/local/data/databank/PetraIII/ct4/df_ct4_.tif'
white_file_name = '/local/data/databank/PetraIII/ct4/bf_ct4_.tif'
hdf5_file_name = '/local/data/databank/dataExchange/microCT/PetraIII_ct4_180.h5'
sample_name = 'ct4'

# ct4: Leaf of rice
# Fresh sample measured at cryogenic condition

projections_start = 0
projections_end = 601
white_start = 1
white_end = 19
white_step = 1
dark_start = 0
dark_end = 20
dark_step = 1

##### if testing uncomment
##projections_start = 0
##projections_end = 5
##white_start = 0
##white_end = 5
##white_step = 1
##dark_start = 0
##dark_end = 5
##dark_step = 1


verbose = True

if verbose: print file_name
if verbose: print hdf5_file_name
if verbose: print sample_name


if verbose: print "Dark start, end", dark_start, dark_end
if verbose: print "White start, end", white_start, white_end
if verbose: print "Projections start, end", projections_start, projections_end


mydata = Convert()
# Create minimal hdf5 file
mydata.series_of_images(file_name,
                 hdf5_file_name,
                 projections_start,
                 projections_end,
                 # projections_angle_range=360,
                 white_file_name = white_file_name,
                 white_start = white_start,
                 white_end = white_end,
                 white_step = white_step,
                 dark_file_name = dark_file_name,
                 dark_start = dark_start,
                 dark_end = dark_end,
                 dark_step = dark_step,
                 sample_name = sample_name,
                 digits = 5,
                 zeros = True,
                 # verbose = False
                 )

 
# Add extra metadata if available

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='a') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'Petra III'}) )

# Create HDF5 subgroup
# /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': 'Petra III'},
                                    date_time={'value': "2011-25-05T19:42:13+0100"},
                                    beamline={'value': "P06"},
                                    )
)

# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Walter Schroeder"},
                                            role={'value':"Project PI"},
                )
    )

f.close()
if verbose: print "Done converting ", file_name

###if __name__ == "__main__":
###    main()

