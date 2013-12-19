# -*- coding: utf-8 -*-
# file_name: data_read.py
from file_types import Hdf5
from file_types import Tiff
import numpy as np
import os

class Dataset():
    def __init__(self, data=None, white=None, dark=None,
                 center=None, angles=None):
        self.data = data
        self.white = white
        self.dark = dark
        self.center = center
        self.angles = angles

    def read_tiff(self, file_name,
                      projections_start=None,
                      projections_end=None,
                      projections_step=None,
                      slices_start=None,
                      slices_end=None,
                      slices_step=None,
                      pixels_start=None,
                      pixels_end=None,
                      pixels_step=None,
                      white_start=None,
                      white_end=None,
                      dark_start=None,
                      dark_end=None,
                      digits=4,
                      zeros=True,
                      dtype='uint16'):
        """Read a stack of TIFF files in a folder.

        Parameters
        ----------
        inputFile : str
            Name of the input TIFF file.

        projections_start, projections_end, projections_step : scalar, optional
            Values of the start, end and step of the projections to
            be used for slicing for the whole ndarray.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the slices to
            be used for slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the pixels to
            be used for slicing for the whole ndarray.

        white_start, white_end : scalar, optional
            Values of the start, end and step of the
            slicing for the whole white field shots.

        dark_start, dark_end : scalar, optional
            Values of the start, end and step of the
            slicing for the whole dark field shots.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        Returns
        -------
        inputData : ndarray
            Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """

        if file_name.endswith('tif') or \
           file_name.endswith('tiff'):
            dataFile = file_name.split('.')[-2]
            dataExtension = file_name.split('.')[-1]

        fileIndex = ["" for x in range(digits)]

        for m in range(digits):           
            if zeros is True:
               fileIndex[m] = '0' * (digits - m - 1)

            elif zeros is False:
               fileIndex[m] = ''

        ind = range(projections_start, projections_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    fileName = dataFile + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                    break

            if os.path.isfile(fileName):
                print 'Reading projection file: ' + os.path.realpath(fileName)
                f = Tiff()
                tmpdata = f.read(fileName,
                                    x_start=slices_start,
                                    x_end=slices_end,
                                    x_step=slices_step,
                                    dtype=dtype)
                if m == 0: # Get resolution once.
                    inputData = np.empty((projections_end-projections_start,
                                        tmpdata.shape[0],
                                        tmpdata.shape[1]),
                                        dtype=dtype)
                inputData[m, :, :] = tmpdata
        self.data = inputData
        
        ind = range(white_start, white_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    fileName = dataFile + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                    break

            if os.path.isfile(fileName):
                print 'Reading white file: ' + os.path.realpath(fileName)
                f = Tiff()
                tmpdata = f.read(fileName,
                                    x_start=slices_start,
                                    x_end=slices_end,
                                    x_step=slices_step,
                                    dtype=dtype)
                if m == 0: # Get resolution once.
                    inputData = np.empty((white_end-white_start,
                                        tmpdata.shape[0],
                                        tmpdata.shape[1]),
                                        dtype=dtype)
                inputData[m, :, :] = tmpdata
        self.white = inputData

        ind = range(dark_start, dark_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    fileName = dataFile + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                    break

            if os.path.isfile(fileName):
                print 'Reading dark file: ' + os.path.realpath(fileName)
                f = Tiff()
                tmpdata = f.read(fileName,
                                    x_start=slices_start,
                                    x_end=slices_end,
                                    x_step=slices_step,
                                    dtype=dtype)
                if m == 0: # Get resolution once.
                    inputData = np.empty((dark_end-dark_start,
                                        tmpdata.shape[0],
                                        tmpdata.shape[1]),
                                        dtype=dtype)
                inputData[m, :, :] = tmpdata
        self.dark = inputData

     
    def read_hdf5(self, file_name,
                  projections_start=None,
                  projections_end=None,
                  projections_step=None,
                  slices_start=None,
                  slices_end=None,
                  slices_step=None,
                  pixels_start=None,
                  pixels_end=None,
                  pixels_step=None,
                  white_start=None,
                  white_end=None,
                  dark_start=None,
                  dark_end=None,
                  dtype='float32'):
        """ Read Data Exchange HDF5 file.

        Parameters
        ----------
        file_name : str
            Input file.

        projections_start, projections_end, projections_step : scalar, optional
            Values of the start, end and step of the projections to
            be used for slicing for the whole ndarray.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the slices to
            be used for slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the pixels to
            be used for slicing for the whole ndarray.

        white_start, white_end : scalar, optional
            Values of the start, end and step of the
            slicing for the whole white field shots.

        dark_start, dark_end : scalar, optional
            Values of the start, end and step of the
            slicing for the whole dark field shots.

        dtype : str, optional
            Desired output data type.
        """
        print "Reading data..."
        self.file_name = file_name

        # Initialize f to null.
        f = None

        # Get the file_name in lower case.
        lFn = file_name.lower()

        # Split the string with the delimeter '.'
        end = lFn.split('.')

        # If the string has an extension.
        if len(end) > 1:
            # Check.
            if end[1] == 'h5' or end[1] == 'hdf':
                f = Hdf5()

        # If f != None the call read on it.
        if not f == None:
            # Read data from exchange group.
            self.data = f.read(file_name,
                                array_name='exchange/data',
                                x_start=projections_start,
                                x_end=projections_end,
                                x_step=projections_step,
                                y_start=slices_start,
                                y_end=slices_end,
                                y_step=slices_step,
                                z_start=pixels_start,
                                z_end=pixels_end,
                                z_step=pixels_step).astype(dtype)

            # Read white field data from exchange group.
            self.white = f.read(file_name,
                                array_name='exchange/data_white',
                                x_start=white_start,
                                x_end=white_end,
                                y_start=slices_start,
                                y_end=slices_end,
                                y_step=slices_step,
                                z_start=pixels_start,
                                z_end=pixels_end,
                                z_step=pixels_step).astype(dtype)

            # Read dark field data from exchange group.
            self.dark = f.read(file_name,
                                array_name='exchange/data_dark',
                                x_start=dark_start,
                                x_end=dark_end,
                                y_start=slices_start,
                                y_end=slices_end,
                                y_step=slices_step,
                                z_start=pixels_start,
                                z_end=pixels_end,
                                z_step=pixels_step).astype(dtype)

            # Assign the rotation center.
            self.center = self.data.shape[2] / 2
        else:
            print 'Unsupported file.'
