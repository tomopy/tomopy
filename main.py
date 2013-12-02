# -*- coding: utf-8 -*-
# Filename: main.py
""" Main program for tomographic reconstruction.
"""
#import tomoRecon.tomoRecon
from preprocessing.preprocess import Preprocess

# Read input HDF file.
filename = '/local/dgursoy/data/Harrison_Aus_2013/A01_.h5'
mydata = Preprocess()
mydata.read_hdf5(filename, slicesStart=200, slicesEnd=201)
