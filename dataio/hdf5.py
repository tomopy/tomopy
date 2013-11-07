# -*- coding: utf-8 -*-
# Filename: hdf5.py
""" Module for the basic functions to work with HDF data files.
"""
import h5py
import os


def read(inputFile,
         arrayName=None,
         projectionsStart=None,
         projectionsEnd=None,
         projectionsStep=None,
         slicesStart=None,
         slicesEnd=None,
         slicesStep=None,
         pixelsStart=None,
         pixelsEnd=None,
         pixelsStep=None):
    """ Read 3-D data from exchange group of hdf file.

    Opens ``inputFile`` and reads the contents
    of the array specified by ``arrayName`` in
    the specified group of the HDF file.

    Parameters
    ----------
    inputFile : str
        Input HDF file.

    arrayName : str
        Name of the array to be read at exchange group.

    projectionsStart, projectionsEnd, projectionsStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    slicesStart, slicesEnd, slicesStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixelsStart, pixelsEnd, pixelsStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    Returns
    -------
    out : ndarray
        Returns the data as a matrix.
    """
    # Read data from file.
    f = h5py.File(inputFile, 'r')
    hdfdata = f[arrayName]

    # Select desired slices from whole data.
    numProjections, numSlices, numPixels = hdfdata.shape
    if projectionsStart is None:
        projectionsStart = 0
    if projectionsEnd is None:
        projectionsEnd = numProjections
    if projectionsStep is None:
        projectionsStep = 1
    if slicesStart is None:
        slicesStart = 0
    if slicesEnd is None:
        slicesEnd = numSlices
    if slicesStep is None:
        slicesStep = 1
    if pixelsStart is None:
        pixelsStart = 0
    if pixelsEnd is None:
        pixelsEnd = numPixels
    if pixelsStep is None:
        pixelsStep = 1

    # Construct dataset from desired slices.
    dataset = hdfdata[projectionsStart:projectionsEnd:projectionsStep,
                      slicesStart:slicesEnd:slicesStep,
                      pixelsStart:pixelsEnd:pixelsStep]
    f.close()
    return dataset


def write(dataset, outputFile='./data/recon.h5'):
    """ Write data to HDF file.

    Parameters
    -----------
    dataset : ndarray
        Input values.

    outputFile : str, optional
        Name of the output HDF file that contains the dataset.

    arrayName : str, optional
        Name of the group that the data will be put
        under the exchange group of the HDF file.
    """
    # Enforce HDF data format if different.
    if not outputFile.endswith('h5'):
        outputFile = outputFile.split(".")[-2] + 'h5'

    # Create new folders.
    dirPath = os.path.dirname(outputFile)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Write data
    f = h5py.File(outputFile, 'w')
    f.create_dataset('implements', data='exchange')
    exchangeGrp = f.create_group("exchange")
    exchangeGrp.create_dataset('recon', data=dataset)
    f.close()
