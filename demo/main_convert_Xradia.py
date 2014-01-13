# -*- coding: utf-8 -*-
# Filename: XradiaConverter.py
""" Main program for convert SLS data into dataExchange.
"""
import dataio.xradia.xradia_xrm as xradia
import dataio.xradia.data_stack_sim as dstack
import dataio.xradia.data_struct as dstruct
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry

import numpy as np
import os

import scipy

from dataio.data_exchange import DataExchangeFile, DataExchangeEntry
from dataio.data_convert import Convert

import re

#def main():

file_name = '/local/data/databank/TXM_26ID/20130731_004_Stripe_Solder_Sample_Tip1_TomoScript_181imgs_p1s_b1.txrm'
white_file_name = '/local/data/databank/TXM_26ID/20130731_001_Background_Reference_20imgs_p5s_b1.xrm'
hdf5_file_name = '/local/data/databank/dataExchange/TXM/20130731_004_Stripe_Solder_Sample_Tip1_nx.h5'
log_file = '/local/data/databank/dataExchange/TXM/Blakely20130731_004_Stripe_Solder_Sample_Tip1.log'

verbose = True

if verbose: print file_name
if verbose: print white_file_name
if verbose: print hdf5_file_name
if verbose: print log_file

mydata = Convert()
# Create minimal hdf5 file
if verbose: print "Reading data ... "
mydata.x_radia(file_name,
               hdf5_file_name = hdf5_file_name,
               white_file_name = white_file_name,
               sample_name = 'Stripe_Solder_Sample_Tip1'
               )
if verbose: print "Done reading data ... "

 
# Add extra metadata if available

if verbose: print "Adding extra metadata ..."
reader = xradia.xrm()
array = dstruct
reader.read_txrm(file_name,array)

# Read angles
n_angles = np.shape(array.exchange.angles)
if verbose: print "Done reading ", n_angles, " angles"
theta = np.zeros(n_angles)
theta = array.exchange.angles[:]


# Save any other available metadata in a log file
f = open(log_file,'w')
f.write('Data creation date: \n')
f.write(str(array.information.file_creation_datetime))
f.write('\n')
f.write('=======================================\n')
f.write('Sample name: \n')
f.write(str(array.information.sample.name))
f.write('\n')
f.write('=======================================\n')
f.write('Experimenter name: \n')
f.write(str(array.information.experimenter.name))
f.write('\n')
f.write('=======================================\n')
f.write('X-ray energy: \n')
f.write(str(array.exchange.energy))
f.write(str(array.exchange.energy_units))
f.write('\n')
f.write('=======================================\n')
f.write('Angles: \n')
f.write(str(array.exchange.angles))
f.write('\n')
f.write('=======================================\n')
f.write('Data axes: \n')
f.write(str(array.exchange.data_axes))
f.write('\n')
f.write('=======================================\n')
f.write('x distance: \n')
f.write(str(array.exchange.x))
f.write('\n')
f.write('=======================================\n')
f.write('x units: \n')
f.write(str(array.exchange.x_units))
f.write('\n')
f.write('=======================================\n')
f.write('y distance: \n')
f.write(str(array.exchange.y))
f.write('\n')
f.write('=======================================\n')
f.write('y units: \n')
f.write(str(array.exchange.y_units))
f.write('\n')
f.close()


# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='a') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'APS-CNM 26-ID'}) )

### Create HDF5 subgroup
### /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': "Advanced Photon Source"},
                                    date_time={'value': "2013-07-31T19:42:13+0100"},
                                    beamline={'value': "26-ID"},
                                    )
)

# Create HDF5 subgroup
# /measurement/instrument/monochromator
f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Unknown'},
                                            energy={'value': float(array.exchange.energy[0]), 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                            mono_stripe={'value': 'Unknown'},
                                            )
    )

# Create HDF5 subgroup
# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Robert Winarski"},
                                            role={'value':"Project PI"},
                )
    )

# Create HDF5 subgroup
# /measurement/sample
f.add_entry( DataExchangeEntry.data(theta={'value': theta, 'units':'degrees'}))

f.close()
if verbose: print "Done converting ", file_name

###if __name__ == "__main__":
###    main()

