# -*- coding: utf-8 -*-
# file_name: data_converter.py
import numpy as np
import os
import h5py
from dataio.file_types import Tiff, Hdf4

def tiff_to_hdf5(input_file,
                 input_start,
                 input_end,
                 slices_start=None,
                 slices_end=None,
                 slices_step=None,
                 pixels_start=None,
                 pixels_end=None,
                 pixels_step=None,
                 digits=4,
                 zeros=True,
                 dtype='uint16',
                 output_file='myfile.h5',
                 white_file=None,
                 white_start=None,
                 white_end=None,
                 dark_file=None,
                 dark_start=None,
                 dark_end=None):
    """ Converts a stack of projection 16-bit TIFF files
    in a folder to a single HDF5 file. The dataset is
    constructed using the projection data, white field
    and dark field images.

    Parameters
    ----------
    input_file : str
        Name of the generic input file name
        for all the TIFF files to be assembled.

    input_start, input_end : scalar
        Determines the portion of the TIFF images
        to be used for assembling the HDF file.

    slices_start, slices_end, slices_step : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixels_start, pixels_end, pixels_step : scalar, optional
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

    output_file : str
        Name of the output HDF file.

    white_file : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    white_start, white_end : scalar, optional
        Determines the portion of the white
        field TIFF images to be used for
        assembling HDF file.

    dark_file : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    dark_start, dark_end : scalar, optional
        Determines the portion of the dark
        field TIFF images to be used for
        assembling HDF file.
    """
    # Create new folders.
    dirPath = os.path.dirname(output_file)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Prepare HDF5 file.
    print 'Assembling HDF5 file: ' + os.path.realpath(output_file)
    f = h5py.File(output_file, 'w')
    f.create_dataset('implements', data='exchange')
    exchangeGrp = f.create_group("exchange")

    # Read projection TIFF files in the given folder.
    input_data = read_stack(input_file,
                           input_start,
                           input_end,
                           slices_start=slices_start,
                           slices_end=slices_end,
                           slices_step=slices_step,
                           pixels_start=pixels_start,
                           pixels_end=pixels_end,
                           pixels_step=pixels_step,
                           dtype=dtype,
                           digits=digits,
                           zeros=zeros)
    exchangeGrp.create_dataset('data', data=input_data, dtype=dtype)

    # Read white-field TIFF files in the given folder.
    if not white_file == None:
        whiteData = read_stack(white_file,
                               white_start,
                               white_end,
                               slices_start=slices_start,
                               slices_end=slices_end,
                               slices_step=slices_step,
                               pixels_start=pixels_start,
                               pixels_end=pixels_end,
                               pixels_step=pixels_step,
                               dtype=dtype,
                               digits=digits,
                               zeros=zeros)
        exchangeGrp.create_dataset('data_white', data=whiteData, dtype=dtype)

    # Read dark-field TIFF files in the given folder.
    if not dark_file == None:
        darkData = read_stack(dark_file,
                              dark_start,
                              dark_end,
                              slices_start=slices_start,
                              slices_end=slices_end,
                              slices_step=slices_step,
                              pixels_start=pixels_start,
                              pixels_end=pixels_end,
                              pixels_step=pixels_step,
                              dtype=dtype,
                              digits=digits,
                              zeros=zeros)
        exchangeGrp.create_dataset('data_dark', data=darkData, dtype=dtype)
    f.close()

def hdf4_to_hdf5(input_file,
                 input_start,
                 input_end,
                 slices_start=None,
                 slices_end=None,
                 slices_step=None,
                 pixels_start=None,
                 pixels_end=None,
                 pixels_step=None,
                 digits=4,
                 zeros=True,
                 dtype='uint16',
                 array_name=None,
                 output_file='myfile.h5',
                 white_file=None,
                 white_start=None,
                 white_end=None,
                 dark_file=None,
                 dark_start=None,
                 dark_end=None):
    """ Converts a stack of projection 16-bit HDF files
    in a folder to a single HDF5 file. The dataset is
    constructed using the projection data, white field
    and dark field images.

    Parameters
    ----------
    input_file : str
        Name of the generic input file name
        for all the TIFF files to be assembled.

    input_start, input_end : scalar
        Determines the portion of the TIFF images
        to be used for assembling the HDF file.

    slices_start, slices_end, slices_step : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixels_start, pixels_end, pixels_step : scalar, optional
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

    output_file : str
        Name of the output HDF file.

    white_file : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    white_start, white_end : scalar, optional
        Determines the portion of the white
        field TIFF images to be used for
        assembling HDF file.

    dark_file : str, optional
        Name of the generic input file name
        for all the white field
        TIFF files to be assembled.

    dark_start, dark_end : scalar, optional
        Determines the portion of the dark
        field TIFF images to be used for
        assembling HDF file.
    """
    # Create new folders.
    dirPath = os.path.dirname(output_file)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    # Prepare HDF5 file.
    print 'Assembling HDF5 file: ' + os.path.realpath(output_file)
    f = h5py.File(output_file, 'w')
    f.create_dataset('implements', data='exchange')
    exchangeGrp = f.create_group("exchange")

    # Update HDF5 file in chunks.
    chunkSize = 20
    ind1 = np.int(np.floor(np.float(input_start) / chunkSize))
    ind2 = np.int(np.ceil(np.float(input_end) / chunkSize))
    for m in range(ind2-ind1):
        indStart = (m * chunkSize) + input_start
        indEnd = indStart + chunkSize
        if indEnd > input_end:
            indEnd = input_end

        # Read projection files in the given folder.
        input_data = read_stack(input_file,
                               input_start=indStart,
                               input_end=indEnd,
                               slices_start=slices_start,
                               slices_end=slices_end,
                               slices_step=slices_step,
                               pixels_start=pixels_start,
                               pixels_end=pixels_end,
                               pixels_step=pixels_step,
                               digits=digits,
                               zeros=zeros,
                               array_name=array_name)

        # Update HDF5 file.
        if m == 0:
            dset = exchangeGrp.create_dataset('data',
                                              (input_end-input_start,
                                               input_data.shape[1],
                                               input_data.shape[2]),
                                              dtype=dtype)
        dset[(indStart-input_start):(indEnd-input_start), :, :] = input_data

    # Read white-field TIFF files in the given folder.
    if not white_file == None:
        whiteData = read_stack(white_file,
                               white_start,
                               white_end,
                               digits=digits,
                               zeros=zeros,
                               array_name=array_name)
        exchangeGrp.create_dataset('data_white', data=whiteData, dtype=dtype)


    # Read dark-field TIFF files in the given folder.
    if not dark_file == None:
        darkData = read_stack(dark_file,
                              dark_start,
                              dark_end,
                              digits=digits,
                              zeros=zeros,
                              array_name=array_name)
        exchangeGrp.create_dataset('data_dark', data=darkData, dtype=dtype)
    f.close()

def read_stack(input_file,
               input_start,
               input_end,
               slices_start=None,
               slices_end=None,
               slices_step=None,
               pixels_start=None,
               pixels_end=None,
               pixels_step=None,
               digits=4,
               zeros=True,
               dtype='uint16',
               array_name=None):
    """Read a stack of files in a folder.

    Parameters
    ----------
    input_file : str
        Name of the input file.

    input_start, input_end : scalar
        Determines the portion of the images
        to be used for assembling the HDF file.

    slices_start, slices_end, slices_step : scalar, optional
        Values of the start, end and step of the
        slicing for the whole ndarray.

    pixels_start, pixels_end, pixels_step : scalar, optional
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
    input_data : ndarray
        Output 2-D matrix as numpy array.

    .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
    """
    # Split the string with the delimeter '.'
    data_file = input_file.split('.')[-3] + '.' + input_file.split('.')[-2]
    data_extension = input_file.split('.')[-1]

    file_index = ["" for x in range(digits)]
    for m in range(digits):
        if zeros is True:
            file_index[m] = '0' * (digits - m - 1)

        elif zeros is False:
            file_index[m] = ''

    ind = range(input_start, input_end)
    for m in range(len(ind)):
        for n in range(digits):
            if ind[m] < np.power(10, n + 1):
                file_name = data_file + file_index[n] + str(ind[m]) + '.' + data_extension
                break

        if os.path.isfile(file_name):
            print 'Reading file: ' + os.path.realpath(file_name)
            if data_extension == 'tiff' or data_extension == 'tif':
                f = Tiff()
                tmpdata = f.read(file_name, dtype=dtype,
                                slices_start=slices_start,
                                slices_end=slices_end,
                                slices_step=slices_step,
                                pixels_start=pixels_start,
                                pixels_end=pixels_end,
                                pixels_step=pixels_step)
            elif data_extension == 'hdf':
                f = Hdf4()
                tmpdata = f.read(file_name,
                                slices_start=slices_start,
                                slices_end=slices_end,
                                slices_step=slices_step,
                                pixels_start=pixels_start,
                                pixels_end=pixels_end,
                                pixels_step=pixels_step,
                                array_name=array_name)
        if m == 0: # Get resolution once.
            input_data = np.empty((input_end-input_start,
                                  tmpdata.shape[0],
                                  tmpdata.shape[1]),
                                 dtype=dtype)
        input_data[m, :, :] = tmpdata
    return input_data
