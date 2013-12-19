# -*- coding: utf-8 -*-
# Filename: SLSConverter.py
""" Main program for convert SLS data into dataExchange.
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

filename = '/local/data/databank/SLS_2011/Blakely_SLS/Blakely.tif'
SLSlogFile = '/local/data/databank/SLS_2011/Blakely_SLS/Blakely.log'

print filename
print SLSlogFile


#Read input SLS data
file = open(SLSlogFile, 'r')
for line in file:
    if 'Number of darks' in line:
        NumberOfDarks = re.findall(r'\d+', line)
        print 'Number of Darks', NumberOfDarks[0]
    if 'Number of flats' in line:
        NumberOfFlats = re.findall(r'\d+', line)
        print 'Number of Flats', NumberOfFlats[0]
    if 'Number of projections' in line:
        NumberOfProjections = re.findall(r'\d+', line)
        print 'Number of Projections', NumberOfProjections[0]
    if 'Number of inter-flats' in line:
        NumberOfInterFlats = re.findall(r'\d+', line)
        print 'Number of inter-flats', NumberOfInterFlats[0]
    if 'Inner scan flag' in line:
        InnerScanFlag = re.findall(r'\d+', line)
        print 'Inner scan flag', InnerScanFlag[0]
    if 'Flat frequency' in line:
        FlatFrequency = re.findall(r'\d+', line)
        print 'Flat frequency', FlatFrequency[0]
    if 'Rot Y min' in line:
        RotYmin = re.findall(r'\d+.\d+', line)
        print 'Rot Y min', RotYmin[0]
    if 'Rot Y max' in line:
        RotYmax = re.findall(r'\d+.\d+', line)
        print 'Rot Y max', RotYmax[0]
    if 'Angular step' in line:
        AngularStep = re.findall(r'\d+.\d+', line)
        print 'Angular step', AngularStep[0]
file.close()

darkStart = 1
darkEnd = int(NumberOfDarks[0]) + 1
whiteStart = darkEnd
whiteEnd = whiteStart + int(NumberOfFlats[0])
projectionStart = whiteEnd
projectionEnd = projectionStart + int(NumberOfProjections[0])

print darkStart, darkEnd
print whiteStart, whiteEnd
print projectionStart, projectionEnd

dark_start = 1
dark_end = 21
white_start = 21
white_end = 221
projections_start = 221
projections_end = 1662

white_end = 42
projections_end = 242

mydata = Preprocess()

mydata.read_tiff(filename, projections_start, projections_end, white_start = white_start, white_end = white_end, dark_start= dark_start, dark_end = dark_end)


#Write HDF5 file.

HDF5 = '/local/data/databank/tt5.h5'
# Open DataExchange file
f = DataExchangeFile(HDF5, mode='w') 


# Create core HDF5 dataset in exchange group for 180 deep stack
# of x,y images /exchange/data
f.add_entry( DataExchangeEntry.data(data={'value': mydata.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x',
                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
)
f.add_entry( DataExchangeEntry.data(data_dark={'value': mydata.dark, 'units':'counts', 'axes':'theta_dark:y:x',
                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
)
f.add_entry( DataExchangeEntry.data(data_white={'value': mydata.white, 'units':'counts', 'axes':'theta_white:y:x',
                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
)
f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))

f.close()

#if __name__ == "__main__":
#    main()

