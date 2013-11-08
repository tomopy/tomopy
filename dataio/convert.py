# -*- coding: utf-8 -*-
# Filename: convert.py
""" Module for converting data into dataExchange.
"""
from data_exchange import DataExchangeFile, DataExchangeEntry
import matplotlib.pyplot as plt
from dataio import data
from dataio import tiff
from dataio import hdf5

import numpy as np

import os
import h5py

import re

def APS2BM(dataFolder, baseName):

        #dataFolder = '/local/data/Harrison_Aug_2013/A16/raw'
        dataExchangeFolder = dataFolder + '/' + 'dataExchange'

        darkStart = 724
        darkEnd = 725
        whiteStart = 1
        whiteEnd = 2
#        projectionStart = 2
#        projectionEnd = 722
        projectionStart = 2
        projectionEnd = 20

        print darkStart, darkEnd
        print whiteStart, whiteEnd
        print projectionStart, projectionEnd

        print 'loading data'
        data = hdf5.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = projectionStart, inputEnd = projectionEnd) 
        print 'loading dark'
        data_dark = hdf5.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = darkStart, inputEnd = darkEnd)
        print 'loading white'
        data_white = hdf5.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = whiteStart, inputEnd = whiteEnd)


        print data.shape
        print data_dark.shape
        print data_white.shape

        # Create new folders.
        HDF5File = dataExchangeFolder + '/' + baseName + '.' + 'h5'
        print HDF5File
        dirPath = os.path.dirname(HDF5File)
        if not os.path.exists(dirPath):
                os.makedirs(dirPath)

        # Write HDF5 file.

        print 'Assembling HDF5 file: ' + os.path.realpath(HDF5File)
        # Open DataExchange file
        f = DataExchangeFile(HDF5File, mode='w') 
        
    
        # Create core HDF5 dataset in exchange group for 180 deep stack
        # of x,y images /exchange/data
        f.add_entry( DataExchangeEntry.data(data={'value': data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x',
                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
        f.add_entry( DataExchangeEntry.data(data_dark={'value': data_dark, 'units':'counts', 'axes':'theta_dark:y:x',
                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.add_entry( DataExchangeEntry.data(data_white={'value': data_white, 'units':'counts', 'axes':'theta_white:y:x',
                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.close()
        return HDF5File

def SLSTomcat(inputFile):

        print inputFile
        if inputFile.endswith('tif') or \
           inputFile.endswith('tiff'):
                dataFile = inputFile.split('.')[-2]
                dataExtension = inputFile.split('.')[-1]

        SLSlogFile = dataFile + '.log'
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

##        projectionStart = 221
##        projectionEnd = 322
##        whiteStart = 22
##        whiteEnd = 33
##        darkStart = 1
##        darkEnd = 21
##
##        print darkStart, darkEnd
##        print whiteStart, whiteEnd
##        print projectionStart, projectionEnd

        HDF5File = dataFile + '.' + 'h5'
        print HDF5File
        dirPath = os.path.dirname(HDF5File)
        if not os.path.exists(dirPath):
                os.makedirs(dirPath)

        data = tiff.TIFF2HDF5(inputFile = inputFile,
                              inputStart = projectionStart,
                              inputEnd = projectionEnd,
                              whiteFile = inputFile,
                              whiteStart = whiteStart,
                              whiteEnd = whiteEnd,
                              darkFile = inputFile,
                              darkStart = darkStart,
                              darkEnd = darkEnd,
                              outputFile = HDF5File)
#        data_dark = tiff.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = darkStart, inputEnd = darkEnd)
#        data_white = tiff.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = whiteStart, inputEnd = whiteEnd)


#        print data.shape
#        print data_dark.shape
#        print data_white.shape
        # Create new folders.
#        HDF5File = dataExchangeFolder + '/' + baseName + '.' + 'h5'
#        print HDF5File
#        dirPath = os.path.dirname(HDF5File)
#        if not os.path.exists(dirPath):
#                os.makedirs(dirPath)

        # Write HDF5 file.

#        print 'Assembling HDF5 file: ' + os.path.realpath(HDF5File)
        # Open DataExchange file
#       f = DataExchangeFile(HDF5File, mode='w') 
        
    
        # Create core HDF5 dataset in exchange group for 180 deep stack
        # of x,y images /exchange/data
#        f.add_entry( DataExchangeEntry.data(data={'value': data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x',
#                                                   'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
#        f.add_entry( DataExchangeEntry.data(data_dark={'value': data_dark, 'units':'counts', 'axes':'theta_dark:y:x',
#                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.add_entry( DataExchangeEntry.data(data_white={'value': data_white, 'units':'counts', 'axes':'theta_white:y:x',
#                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.close()
        return HDF5File

def PetraIII(inputFile,
             whiteFile = None,
             darkFile = None,
             digits = 5):

        print inputFile
        if inputFile.endswith('tif') or \
           inputFile.endswith('tiff'):
                dataFile = inputFile.split('.')[-2]
                dataExtension = inputFile.split('.')[-1]

        logFile = dataFile + '.log'
        print logFile

        #Read input log data
        file = open(logFile, 'r')
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

# this is to force values for testing
        projectionStart = 0
        projectionEnd = 1800
        whiteStart = 0
        whiteEnd = 20
        darkStart = 0
        darkEnd = 20
# this is to force values for testing

        print darkStart, darkEnd
        print whiteStart, whiteEnd
        print projectionStart, projectionEnd

        HDF5File = dataFile + '.' + 'h5'
        print HDF5File
        dirPath = os.path.dirname(HDF5File)
        if not os.path.exists(dirPath):
                os.makedirs(dirPath)
        print inputFile
        print whiteFile
        print darkFile
        data = tiff.TIFF2HDF5(inputFile = inputFile,
                              inputStart = projectionStart,
                              inputEnd = projectionEnd,
                              whiteFile = whiteFile,
                              whiteStart = whiteStart,
                              whiteEnd = whiteEnd,
                              darkFile = darkFile,
                              darkStart = darkStart,
                              darkEnd = darkEnd,
                              outputFile = HDF5File,
                              digits = digits)
#        data_dark = tiff.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = darkStart, inputEnd = darkEnd)
#        data_white = tiff.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = whiteStart, inputEnd = whiteEnd)


#        print data.shape
#        print data_dark.shape
#        print data_white.shape
        # Create new folders.
#        HDF5File = dataExchangeFolder + '/' + baseName + '.' + 'h5'
#        print HDF5File
#        dirPath = os.path.dirname(HDF5File)
#        if not os.path.exists(dirPath):
#                os.makedirs(dirPath)

        # Write HDF5 file.

#        print 'Assembling HDF5 file: ' + os.path.realpath(HDF5File)
        # Open DataExchange file
#       f = DataExchangeFile(HDF5File, mode='w') 
        
    
        # Create core HDF5 dataset in exchange group for 180 deep stack
        # of x,y images /exchange/data
#        f.add_entry( DataExchangeEntry.data(data={'value': data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x',
#                                                   'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
#        f.add_entry( DataExchangeEntry.data(data_dark={'value': data_dark, 'units':'counts', 'axes':'theta_dark:y:x',
#                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.add_entry( DataExchangeEntry.data(data_white={'value': data_white, 'units':'counts', 'axes':'theta_white:y:x',
#                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
#        )
#        f.close()
        return HDF5File

def APS26ID(dataFolder, baseName):


        dataExchangeFolder = dataFolder + '/' + 'dataExchange'

        # for test
        darkStart = 1
        darkEnd = 21
        whiteStart = 21
        whiteEnd = 41
        projectionStart = 221
        projectionEnd = 241

        print darkStart, darkEnd
        print whiteStart, whiteEnd
        print projectionStart, projectionEnd

        data = bin.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = projectionStart, inputEnd = projectionEnd)
        data_dark = bin.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = darkStart, inputEnd = darkEnd)
        data_white = bin.convert2stack(dataFolder = dataFolder, baseName = baseName, inputStart = whiteStart, inputEnd = whiteEnd)


        print data.shape
        print data_dark.shape
        print data_white.shape
        # Create new folders.
        HDF5File = dataExchangeFolder + '/' + baseName + '.' + 'h5'
        print HDF5File
        dirPath = os.path.dirname(HDF5File)
        if not os.path.exists(dirPath):
                os.makedirs(dirPath)

        # Write HDF5 file.

        print 'Assembling HDF5 file: ' + os.path.realpath(HDF5File)
        # Open DataExchange file
        f = DataExchangeFile(HDF5File, mode='w') 
        
    
        # Create core HDF5 dataset in exchange group for 180 deep stack
        # of x,y images /exchange/data
        f.add_entry( DataExchangeEntry.data(data={'value': data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x',
                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
        f.add_entry( DataExchangeEntry.data(data_dark={'value': data_dark, 'units':'counts', 'axes':'theta_dark:y:x',
                                                    'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.add_entry( DataExchangeEntry.data(data_white={'value': data_white, 'units':'counts', 'axes':'theta_white:y:x',
                                            'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} })
        )
        f.close()
        return HDF5File

