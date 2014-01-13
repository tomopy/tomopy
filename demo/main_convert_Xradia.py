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
hdf5_file_name = '/local/data/databank/dataExchange/TXM/20130731_004_Stripe_Solder_Sample_Tip1_new.h5'

verbose = True

if verbose: print file_name
if verbose: print white_file_name
if verbose: print hdf5_file_name



mydata = Convert()
# Create minimal hdf5 file
mydata.x_radia(file_name,
               hdf5_file_name = hdf5_file_name,
               white_file_name = white_file_name
               )

 
# Add extra metadata if available

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='a') 

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
f.close()
if verbose: print "Done converting ", file_name

###if __name__ == "__main__":
###    main()

