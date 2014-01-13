# -*- coding: utf-8 -*-
# Filename: SLSConverter.py
""" Main program for convert SLS data into dataExchange.
"""
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry
from dataio.data_convert import Convert

import re

#def main():

##file_name = '/local/data/databank/ALS_2011/Blakely/blakely_raw/blakelyALS_.tif'
##dark_file_name = '/local/data/databank/ALS_2011/Blakely/blakely_raw/blakelyALSdrk_.tif'
##white_file_name = '/local/data/databank/ALS_2011/Blakely/blakely_raw/blakelyALSbak_.tif'
##log_file = '/local/data/databank/ALS_2011/Blakely/blakely_raw/blakelyALS.sct'
##
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/Blakely_ALS_2011.h5'

file_name = '/local/data/databank/ALS_2011/Hornby/raw/hornbyALS_.tif'
dark_file_name = '/local/data/databank/ALS_2011/Hornby/raw/hornbyALSdrk_.tif'
white_file_name = '/local/data/databank/ALS_2011/Hornby/raw/hornbyALSbak_.tif'
log_file = '/local/data/databank/ALS_2011/Hornby/raw/hornbyALS.sct'

hdf5_file_name = '/local/data/databank/dataExchange/microCT/Hornby_ALS_2011_new_series_of_images.h5'

verbose = True

if verbose: print file_name
if verbose: print log_file
if verbose: print hdf5_file_name



#Read input ALS data
file = open(log_file, 'r')
if verbose: print '###############################'
for line in file:
    if '-scanner' in line:
        Source = re.sub(r'-scanner ', "", line)
        if verbose: print 'Facility', Source
    
    if '-object' in line:
        Sample = re.sub(r'-object ', "", line)
        if verbose: print 'Sample', Sample
        
    if '-senergy' in line:
        Energy = re.findall(r'\d+.\d+', line)
        if verbose: print 'Energy', Energy[0]
        
    if '-scurrent' in line:
        Current = re.findall(r'\d+.\d+', line)
        if verbose: print 'Current', Current[0]

    if '-nangles' in line:
        Angles = re.findall(r'\d+', line)
        if verbose: print 'Angles', Angles[0]

    if '-i0cycle' in line:
        WhiteStep = re.findall(r'\s+\d+', line)
        if verbose: print 'White Step', WhiteStep[0]

if verbose: print '###############################'
file.close()

dark_start = 0
dark_end = 20
dark_step = 1
white_start = 0
white_end = int(Angles[0]) 
white_step = int(WhiteStep[0])
projections_start = 0
projections_end = int(Angles[0])

if verbose: print dark_start, dark_end
if verbose: print white_start, white_end
if verbose: print projections_start, projections_end

### if testing uncomment
##dark_end = 2
##white_end = 361
##projections_end = 2

mydata = Convert()
# Create minimal hdf5 file
mydata.series_of_images(file_name,
                 hdf5_file_name,
                 projections_start,
                 projections_end,
                 white_file_name = white_file_name,
                 white_start = white_start,
                 white_end = white_end,
                 white_step = white_step,
                 dark_file_name = dark_file_name,
                 dark_start = dark_start,
                 dark_end = dark_end,
                 dark_step = dark_step,
                 zeros = False,
                 verbose = False
                 )

 
# Add extra metadata if available

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='a') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'ALS'}) )

# Create HDF5 subgroup
# /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': Source},
                                    date_time={'value': "2011-25-05T19:42:13+0100"},
                                    beamline={'value': "ALS Tomo"},
                                    current={'value': float(Current[0]), 'units': 'mA', 'dataset_opts': {'dtype': 'd'}},
                                    )
)

# Create HDF5 subgroup
# /measurement/instrument/monochromator
f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Unknown'},
                                            energy={'value': float(Energy[0]), 'units': 'eV', 'dataset_opts': {'dtype': 'd'}},
                                            mono_stripe={'value': 'Unknown'},
                                            )
    )

# Create HDF5 subgroup
# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Jane Waruntorn"},
                                            role={'value':"Project PI"},
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

f.close()
if verbose: print "Done converting ", file_name

###if __name__ == "__main__":
###    main()

