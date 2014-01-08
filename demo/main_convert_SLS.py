# -*- coding: utf-8 -*-
# Filename: SLSConverter.py
""" Main program for convert SLS data into dataExchange.
"""
from preprocessing.preprocess import Preprocess
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry
#from inout import data
from dataio.file_types import Tiff
from tomoRecon import tomoRecon
from visualize import image

import matplotlib.pyplot as plt
import numpy as np

import os
import h5py

import re

#def main():

##file_name = '/local/data/databank/SLS_2011/Blakely_SLS/Blakely.tif'
##sls_log_file = '/local/data/databank/SLS_2011/Blakely_SLS/Blakely.log'
##
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/Blakely_SLS_2011_test_angles.h5'

file_name = '/local/data/databank/SLS_2011/Hornby_SLS/Hornby_b.tif'
sls_log_file = '/local/data/databank/SLS_2011/Hornby_SLS/Hornby.log'

hdf5_file_name = '/local/data/databank/dataExchange/microCT/Hornby_SLS_2011.h5'

verbose = True

if verbose: print file_name
if verbose: print sls_log_file



#Read input SLS data
file = open(sls_log_file, 'r')
if verbose: print '###############################'
for line in file:
    if 'Number of darks' in line:
        NumberOfDarks = re.findall(r'\d+', line)
        if verbose: print 'Number of Darks', NumberOfDarks[0]
    if 'Number of flats' in line:
        NumberOfFlats = re.findall(r'\d+', line)
        if verbose: print 'Number of Flats', NumberOfFlats[0]
    if 'Number of projections' in line:
        NumberOfProjections = re.findall(r'\d+', line)
        if verbose: print 'Number of Projections', NumberOfProjections[0]
    if 'Number of inter-flats' in line:
        NumberOfInterFlats = re.findall(r'\d+', line)
        if verbose: print 'Number of inter-flats', NumberOfInterFlats[0]
    if 'Inner scan flag' in line:
        InnerScanFlag = re.findall(r'\d+', line)
        if verbose: print 'Inner scan flag', InnerScanFlag[0]
    if 'Flat frequency' in line:
        FlatFrequency = re.findall(r'\d+', line)
        if verbose: print 'Flat frequency', FlatFrequency[0]
    if 'Rot Y min' in line:
        RotYmin = re.findall(r'\d+.\d+', line)
        if verbose: print 'Rot Y min', RotYmin[0]
    if 'Rot Y max' in line:
        RotYmax = re.findall(r'\d+.\d+', line)
        if verbose: print 'Rot Y max', RotYmax[0]
    if 'Angular step' in line:
        AngularStep = re.findall(r'\d+.\d+', line)
        if verbose: print 'Angular step', AngularStep[0]
if verbose: print '###############################'
file.close()

dark_start = 1
dark_end = int(NumberOfDarks[0]) + 1
white_start = dark_end
white_end = white_start + int(NumberOfFlats[0])
projection_start = white_end
projection_end = projection_start + int(NumberOfProjections[0])

if verbose: print dark_start, dark_end
if verbose: print white_start, white_end
if verbose: print projection_start, projection_end

dark_start = 1
dark_end = 21
white_start = 21
white_end = 221
projections_start = 221
projections_end = 1662

#dark_end = 4
#white_end = 24
#projections_end = 224

z = np.arange(projections_end - projections_start);
if verbose: print z, len(z)
    
# Fabricate theta values
theta = (z * float(180) / (len(z) - 1))
if verbose: print theta

mydata = Preprocess()
mydata.read_tiff(file_name,
                 projections_start,
                 projections_end,
                 white_start = white_start,
                 white_end = white_end,
                 dark_start = dark_start,
                 dark_end = dark_end
                 )


#Write HDF5 file.

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='w') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'Tomcat'}) )

# Create HDF5 subgroup
# /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': 'Swiss Light Source'},
                                    date_time={'value': "2010-11-08T14:51:56+0100"},
                                    beamline={'value': "Tomcat"},
                                    current={'value': 401.96, 'units': 'mA', 'dataset_opts': {'dtype': 'd'}},
                                    )
)

# Create HDF5 subgroup
# /measurement/instrument/monochromator
f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Multilayer'},
                                            energy={'value': 19.260, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                            mono_stripe={'value': 'Ru/C'},
                                            )
    )

# Create HDF5 subgroup
# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Federica Marone"},
                                            role={'value':"Project PI"},
                                            affiliation={'value':"Swiss Light Source"},
                                            phone={'value':"+41 56 310 5318"},
                                            email={'value':"federica.marone@psi.ch"},

                )
    )

# Create HDF5 subgroup
# /measurement/instrument/detector
f.add_entry( DataExchangeEntry.detector(manufacturer={'value':'CooKe Corporation'},
                                        model={'value': 'pco dimax'},
                                        serial_number={'value': '1234XW2'},
                                        bit_depth={'value': 12, 'dataset_opts':  {'dtype': 'd'}},
                                        x_pixel_size={'value': 6.7e-6, 'dataset_opts':  {'dtype': 'f'}},
                                        y_pixel_size={'value': 6.7e-6, 'dataset_opts':  {'dtype': 'f'}},
                                        x_dimensions={'value': 2048, 'dataset_opts':  {'dtype': 'i'}},
                                        y_dimensions={'value': 2048, 'dataset_opts':  {'dtype': 'i'}},
                                        x_binning={'value': 1, 'dataset_opts':  {'dtype': 'i'}},
                                        y_binning={'value': 1, 'dataset_opts':  {'dtype': 'i'}},
                                        operating_temperature={'value': 270, 'units':'K', 'dataset_opts':  {'dtype': 'f'}},
                                        exposure_time={'value': 170, 'units':'ms', 'dataset_opts':  {'dtype': 'd'}},
                                        frame_rate={'value': 3, 'dataset_opts':  {'dtype': 'i'}},
                                        output_data={'value':'/exchange'}
                                        )
    )

f.add_entry(DataExchangeEntry.objective(magnification={'value':10, 'dataset_opts': {'dtype': 'd'}},
                                    )
    )

f.add_entry(DataExchangeEntry.scintillator(name={'value':'LuAg '},
                                            type={'value':'LuAg'},
                                            scintillating_thickness={'value':20e-6, 'dataset_opts': {'dtype': 'd'}},
        )
    )

# Create HDF5 subgroup
# /measurement/experiment
f.add_entry( DataExchangeEntry.experiment( proposal={'value':"e11218"},
            )
    )

# Create HDF5 subgroup
# /measurement/sample
f.add_entry( DataExchangeEntry.sample( name={'value':'Hornby_b'},
                                        description={'value':'rock sample tested at ALS and APS'},
        )
    )

# Create core HDF5 dataset in exchange group for 180 deep stack
# of x,y images /exchange/data
f.add_entry( DataExchangeEntry.data(data={'value': mydata.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
f.add_entry( DataExchangeEntry.data(theta={'value': theta, 'units':'degrees'}))
f.add_entry( DataExchangeEntry.data(data_dark={'value': mydata.dark, 'units':'counts', 'axes':'theta_dark:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
f.add_entry( DataExchangeEntry.data(data_white={'value': mydata.white, 'units':'counts', 'axes':'theta_white:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))

f.close()

###if __name__ == "__main__":
###    main()

