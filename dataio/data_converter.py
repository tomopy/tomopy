# -*- coding: utf-8 -*-
# Filename: data_converter.py
import numpy as np
import os
import h5py
from dataio.file_types import tiff, hdf4

def tiff_to_hdf5(inputFile,
                 inputStart,
                 inputEnd,
                 slicesStart=None,
                 slicesEnd=None,
                 slicesStep=None,
                 pixelsStart=None,
                 pixelsEnd=None,
                 pixelsStep=None,
                 digits=4,
                 zeros=True,
                 dtype='uint16',
                 outputFile='myfile.h5',
                 whiteFile=None,
                 whiteStart=None,
                 whiteEnd=None,
                 darkFile=None,
                 darkStart=None,
                 darkEnd=None):
    """ Converts a stack of projection 16-bit TIFF files
    in a folder to a single HDF5 file. The dataset is
    constructed using the projection data, white field
    and dark field images.

    Parameters
    ----------
    inputFile : str
        Name of the generic input file name
        for all the TIFF files to be assembled.

    inputStart, inputEnd : scalar
        Determines the portion of the TIFF images
        to be used for assembling the HDF file.

    slicesStart, slicesEnd, slicesStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixelsStart, pixelsEnd, pixelsStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    digits : scalar, optional
        Number of digits used for file indexing.
        For example if 4: test_XXXX.tiff

    zeros : bool, optional
        If ``True`` assumes all indexing uses four digits
        (0001, 0002, ..., 9999). If ``False`` omits zeros in
        indexing (1, 2, ..., 9999)

    dtype : str, optional
        Corresponding Numpy data type of the TIFF file.

    outputFile : str
        Name of the output HDF file.

    whiteFile : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    whiteStart, whiteEnd : scalar, optional
        Determines the portion of the white
        field TIFF images to be used for
        assembling HDF file.

    darkFile : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    darkStart, darkEnd : scalar, optional
        Determines the portion of the dark
        field TIFF images to be used for
        assembling HDF file.
    """
    # Create new folders.
    dirPath = os.path.dirname(outputFile)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Prepare HDF5 file.
    print 'Assembling HDF5 file: ' + os.path.realpath(outputFile)
    f = h5py.File(outputFile, 'w')
    f.create_dataset('implements', data='exchange')
    exchangeGrp = f.create_group("exchange")

    # Read projection TIFF files in the given folder.
    inputData = read_stack(inputFile,
                           inputStart,
                           inputEnd,
                           slicesStart=slicesStart,
                           slicesEnd=slicesEnd,
                           slicesStep=slicesStep,
                           pixelsStart=pixelsStart,
                           pixelsEnd=pixelsEnd,
                           pixelsStep=pixelsStep,
                           dtype=dtype,
                           digits=digits,
                           zeros=zeros)
    exchangeGrp.create_dataset('data', data=inputData, dtype=dtype)

    # Read white-field TIFF files in the given folder.
    if not whiteFile == None:
        whiteData = read_stack(whiteFile,
                               whiteStart,
                               whiteEnd,
                               slicesStart=slicesStart,
                               slicesEnd=slicesEnd,
                               slicesStep=slicesStep,
                               pixelsStart=pixelsStart,
                               pixelsEnd=pixelsEnd,
                               pixelsStep=pixelsStep,
                               dtype=dtype,
                               digits=digits,
                               zeros=zeros)
        exchangeGrp.create_dataset('data_white', data=whiteData, dtype=dtype)

    # Read dark-field TIFF files in the given folder.
    if not darkFile == None:
        darkData = read_stack(darkFile,
                              darkStart,
                              darkEnd,
                              slicesStart=slicesStart,
                              slicesEnd=slicesEnd,
                              slicesStep=slicesStep,
                              pixelsStart=pixelsStart,
                              pixelsEnd=pixelsEnd,
                              pixelsStep=pixelsStep,
                              dtype=dtype,
                              digits=digits,
                              zeros=zeros)
        exchangeGrp.create_dataset('data_dark', data=darkData, dtype=dtype)
    f.close()

def hdf4_to_hdf5(inputFile,
                 inputStart,
                 inputEnd,
                 slicesStart=None,
                 slicesEnd=None,
                 slicesStep=None,
                 pixelsStart=None,
                 pixelsEnd=None,
                 pixelsStep=None,
                 digits=4,
                 zeros=True,
                 dtype='uint16',
                 arrayName=None,
                 outputFile='myfile.h5',
                 whiteFile=None,
                 whiteStart=None,
                 whiteEnd=None,
                 darkFile=None,
                 darkStart=None,
                 darkEnd=None):
    """ Converts a stack of projection 16-bit HDF files
    in a folder to a single HDF5 file. The dataset is
    constructed using the projection data, white field
    and dark field images.

    Parameters
    ----------
    inputFile : str
        Name of the generic input file name
        for all the TIFF files to be assembled.

    inputStart, inputEnd : scalar
        Determines the portion of the TIFF images
        to be used for assembling the HDF file.

    slicesStart, slicesEnd, slicesStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixelsStart, pixelsEnd, pixelsStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    digits : scalar, optional
        Number of digits used for file indexing.
        For example if 4: test_XXXX.tiff

    zeros : bool, optional
        If ``True`` assumes all indexing uses four digits
        (0001, 0002, ..., 9999). If ``False`` omits zeros in
        indexing (1, 2, ..., 9999)

    dtype : str, optional
        Corresponding Numpy data type of the TIFF file.

    hdftype : scalar, optional
        Type of HDF files to be read (4:HDF4, 5:HDF5)

    outputFile : str
        Name of the output HDF file.

    whiteFile : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    whiteStart, whiteEnd : scalar, optional
        Determines the portion of the white
        field TIFF images to be used for
        assembling HDF file.

    darkFile : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    darkStart, darkEnd : scalar, optional
        Determines the portion of the dark
        field TIFF images to be used for
        assembling HDF file.
    """
    # Create new folders.
    dirPath = os.path.dirname(outputFile)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Prepare HDF5 file.
    print 'Assembling HDF5 file: ' + os.path.realpath(outputFile)
    f = h5py.File(outputFile, 'w')
    f.create_dataset('implements', data='exchange')
    exchangeGrp = f.create_group("exchange")

    # Update HDF5 file in chunks.
    chunkSize = 20
    ind1 = np.int(np.floor(np.float(inputStart) / chunkSize))
    ind2 = np.int(np.ceil(np.float(inputEnd) / chunkSize))
    for m in range(ind2-ind1):
        indStart = (m * chunkSize) + inputStart
        indEnd = indStart + chunkSize
        if indEnd > inputEnd:
            indEnd = inputEnd

        # Read projection files in the given folder.
        inputData = read_stack(inputFile,
                               inputStart=indStart,
                               inputEnd=indEnd,
                               slicesStart=slicesStart,
                               slicesEnd=slicesEnd,
                               slicesStep=slicesStep,
                               pixelsStart=pixelsStart,
                               pixelsEnd=pixelsEnd,
                               pixelsStep=pixelsStep,
                               digits=digits,
                               zeros=zeros,
                               arrayName=arrayName)

        # Update HDF5 file.
        if m == 0:
            dset = exchangeGrp.create_dataset('data',
                                              (inputEnd-inputStart,
                                               inputData.shape[1],
                                               inputData.shape[2]),
                                              dtype=dtype)
        dset[(indStart-inputStart):(indEnd-inputStart), :, :] = inputData

    # Read white-field TIFF files in the given folder.
    if not whiteFile == None:
        whiteData = read_stack(whiteFile,
                               whiteStart,
                               whiteEnd,
                               digits=digits,
                               zeros=zeros,
                               arrayName=arrayName)
        exchangeGrp.create_dataset('data_white', data=whiteData, dtype=dtype)


    # Read dark-field TIFF files in the given folder.
    if not darkFile == None:
        darkData = read_stack(darkFile,
                              darkStart,
                              darkEnd,
                              digits=digits,
                              zeros=zeros,
                              arrayName=arrayName)
        exchangeGrp.create_dataset('data_dark', data=darkData, dtype=dtype)
    f.close()

def read_stack(inputFile,
               inputStart,
               inputEnd,
               slicesStart=None,
               slicesEnd=None,
               slicesStep=None,
               pixelsStart=None,
               pixelsEnd=None,
               pixelsStep=None,
               digits=4,
               zeros=True,
               dtype='uint16',
               arrayName=None):
    """Read a stack of files in a folder.

    Parameters
    ----------
    inputFile : str
        Name of the input file.

    inputStart, inputEnd : scalar
        Determines the portion of the images
        to be used for assembling the HDF file.

    slicesStart, slicesEnd, slicesStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixelsStart, pixelsEnd, pixelsStep : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    digits : scalar, optional
        Number of digits used for file indexing.
        For example if 4: test_XXXX.tiff

    zeros : bool, optional
        If ``True`` assumes all indexing uses four digits
        (0001, 0002, ..., 9999). If ``False`` omits zeros in
        indexing (1, 2, ..., 9999)

    dtype : str, optional
        Corresponding Numpy data type of the file.

    Returns
    -------
    inputData : ndarray
        Output 2-D matrix as numpy array.

    .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
    """
    # Split the string with the delimeter '.'
    dataFile = inputFile.split('.')[-2]
    dataExtension = inputFile.split('.')[-1]

    fileIndex = ["" for x in range(digits)]
    for m in range(digits):
        if zeros is True:
            fileIndex[m] = '0' * (digits - m - 1)

        elif zeros is False:
            fileIndex[m] = ''

    ind = range(inputStart, inputEnd)
    for m in range(len(ind)):
        for n in range(digits):
            if ind[m] < np.power(10, n + 1):
                fileName = dataFile + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                break

        if os.path.isfile(fileName):
            print 'Reading file: ' + os.path.realpath(fileName)
            if dataExtension == 'tiff' or dataExtension == 'tif':
                f = tiff()
                tmpdata = f.read(fileName, dtype=dtype,
                                slicesStart=slicesStart,
                                slicesEnd=slicesEnd,
                                slicesStep=slicesStep,
                                pixelsStart=pixelsStart,
                                pixelsEnd=pixelsEnd,
                                pixelsStep=pixelsStep)
            elif dataExtension == 'hdf':
                f = hdf4()
                tmpdata = f.read(fileName,
                                slicesStart=slicesStart,
                                slicesEnd=slicesEnd,
                                slicesStep=slicesStep,
                                pixelsStart=pixelsStart,
                                pixelsEnd=pixelsEnd,
                                pixelsStep=pixelsStep,
                                arrayName=arrayName)
        if m == 0: # Get resolution once.
            inputData = np.empty((inputEnd-inputStart,
                                  tmpdata.shape[0],
                                  tmpdata.shape[1]),
                                 dtype=dtype)
        inputData[m, :, :] = tmpdata
    return inputData
