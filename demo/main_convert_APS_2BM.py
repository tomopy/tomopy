# -*- coding: utf-8 -*-
# Filename: SLSConverter.py
""" Main program for convert SLS data into dataExchange.
"""
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry
from dataio.data_convert import Convert

import re

#def main():

##file_name = '/local/data/databank/APS_2_BM/Sam18_hornby/raw/Hornby_19keV_10x_.hdf'
##log_file = '/local/data/databank/APS_2_BM/Sam18_hornby/raw/Hornby.log'
##
##hdf5_file_name = '/local/data/databank/dataExchange/microCT/Hornby_APS_2011.h5'

file_name = '/local/data/databank/APS_2_BM/Sam19_blakely/raw/Blakely_19keV_10x_.hdf'
log_file = '/local/data/databank/APS_2_BM/Sam19_blakely/raw/Blakely.log'

hdf5_file_name = '/local/data/databank/dataExchange/microCT/Blakely_APS_2011.h5'

verbose = True

if verbose: print file_name
if verbose: print log_file
if verbose: print hdf5_file_name



#Read input SLS data
file = open(log_file, 'r')
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
projections_start = white_end
projections_end = projections_start + int(NumberOfProjections[0])

if verbose: print dark_start, dark_end
if verbose: print white_start, white_end
if verbose: print projections_start, projections_end

dark_start = 1504
dark_end = 1505
white_start = 1
white_end = 2
projections_start = 2
projections_end = 1503

### if testing uncomment
##dark_start = 1
##dark_end = 3
##white_start = 10
##white_end = 12
##projections_start = 20
##projections_end = 23

if verbose: print dark_start, dark_end
if verbose: print white_start, white_end
if verbose: print projections_start, projections_end

mydata = Convert()
# Create minimal hdf5 file
mydata.series_of_images(file_name,
                 hdf5_file_name,
                 projections_start,
                 projections_end,
                 white_start = white_start,
                 white_end = white_end,
                 dark_start = dark_start,
                 dark_end = dark_end,
                 digits = 5,
                 data_type = 'hdf4',
                 #verbose = False
             )

 
# Add extra metadata if available

# Open DataExchange file
f = DataExchangeFile(hdf5_file_name, mode='a') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'APS 2-BM'}) )

f.add_entry( DataExchangeEntry.source(name={'value': 'Advanced Photon Source'},
                                    date_time={'value': "2012-07-31T21:15:23+0600"},
                                    beamline={'value': "2-BM"},
                                    current={'value': 101.199, 'units': 'mA', 'dataset_opts': {'dtype': 'd'}},
                                    energy={'value': 7.0, 'units':'GeV', 'dataset_opts': {'dtype': 'd'}},
                                    mode={'value':'TOPUP'}
                                    )
)
# Create HDF5 subgroup
# /measurement/instrument/attenuator
f.add_entry( DataExchangeEntry.attenuator(thickness={'value': 1e-3, 'units': 'm', 'dataset_opts': {'dtype': 'd'}},
                                        type={'value': 'Al'}
                                        )
    )

# Create HDF5 subgroup
# Create HDF5 subgroup
# /measurement/instrument/monochromator
f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Multilayer'},
                                            energy={'value': 19.26, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                            energy_error={'value': 1e-3, 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                            mono_stripe={'value': 'Ru/C'},
                                            )
    )


# Create HDF5 subgroup
# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Jane Waruntorn"},
                                            role={'value':"Project PI"},
                                            affiliation={'value':"University of California"},
                                            facility_user_id={'value':"64924"},

                )
    )

f.add_entry(DataExchangeEntry.objective(manufacturer={'value':'Zeiss'},
                                        model={'value':'Plan-NEOFLUAR 1004-072'},
                                        magnification={'value':5, 'dataset_opts': {'dtype': 'd'}},
                                        numerical_aperture={'value':0.5, 'dataset_opts': {'dtype': 'd'}},
                                    )
    )

f.add_entry(DataExchangeEntry.scintillator(manufacturer={'value':'Crytur'},
                                            serial_number={'value':'12'},
                                            name={'value':'LuAg '},
                                            type={'value':'LuAg'},
                                            scintillating_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
                                            substrate_thickness={'value':50e-6, 'dataset_opts': {'dtype': 'd'}},
        )
    )

# Create HDF5 subgroup
# /measurement/experiment
f.add_entry( DataExchangeEntry.experiment( proposal={'value':"GUP-34353"},
                                            activity={'value':"32-IDBC-2013-106491"},
                                            safety={'value':"106491-49734"},
            )
    )


f.close()
if verbose: print "Done converting ", file_name

###if __name__ == "__main__":
###    main()

