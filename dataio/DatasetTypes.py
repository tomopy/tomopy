# -*- coding: utf-8 -*-
# Filename: DataTypes.py
import h5py
import os
import numpy as np
from scipy import misc
import PIL.Image as Image
from pyhdf import SD
from DatasetFileInterface import DatasetFileInterface


class hdf5(DatasetFileInterface):
    def read(self, fileName,
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
        """ Read 3-D tomographic data from hdf5 file.

        Opens ``fileName`` and reads the contents
        of the array specified by ``arrayName`` in
        the specified group of the HDF file.

        Parameters
        ----------
        fileName : str
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
        f = h5py.File(fileName, 'r')
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

    def write(self, fileName, arrayName):
        """ Write data to hdf5 file.

        Parameters
        -----------
        dataset : ndarray
            Input values.

        fileName : str
            Name of the output HDF file.

        arrayName : str, optional
            Name of the group that the data will be put
            under the exchange group of the HDF file.
        """
        # Create new folders.
        dirPath = os.path.dirname(fileName)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Write data
        f = h5py.File(fileName, 'w')
        f.create_dataset('implements', data='exchange')
        exchangeGrp = f.create_group("exchange")
        exchangeGrp.create_dataset(arrayName, data=self.dataset)
        f.close()


class hdf4(DatasetFileInterface):
    def read(self, fileName,
             arrayName=None,
             slicesStart=None,
             slicesEnd=None,
             slicesStep=None,
             pixelsStart=None,
             pixelsEnd=None,
             pixelsStep=None):
        """ Read 2-D tomographic data from hdf4 file.

        Opens ``fileName`` and reads the contents
        of the array specified by ``arrayName`` in
        the specified group of the HDF file.

        Parameters
        ----------
        fileName : str
            Input HDF file.

        arrayName : str
            Name of the array to be read at exchange group.

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
        f = SD.SD(fileName)
        sds = f.select(arrayName)
        hdfdata = sds.get()
        hdfdata = hdfdata.reshape(hdfdata.shape[1],
                                  hdfdata.shape[0])

        # Select desired slices from whole data.
        numSlices, numPixels = hdfdata.shape
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
        dataset = hdfdata[slicesStart:slicesEnd:slicesStep,
                          pixelsStart:pixelsEnd:pixelsStep]
        return dataset

    def write(self):
        pass


class tiff(DatasetFileInterface):
    def read(self, fileName, dtype='uint16',
             slicesStart=None,
             slicesEnd=None,
             slicesStep=None,
             pixelsStart=None,
             pixelsEnd=None,
             pixelsStep=None):
        """Read TIFF files.

        Parameters
        ----------
        fileName : str
            Name of the input TIFF file.

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        slicesStart, slicesEnd, slicesStep : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixelsStart, pixelsEnd, pixelsStep : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        Returns
        -------
        out : ndarray
            Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        im = Image.open(fileName)
        out = np.fromstring(im.tostring(), dtype).reshape(tuple(list(im.size[::-1])))

        # Select desired slices from whole data.
        numSlices, numPixels = out.shape
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
        return out[slicesStart:slicesEnd:slicesStep,
                   pixelsStart:pixelsEnd:pixelsStep]


    def write(self, dataset,
              fileName,
              slicesStart=None,
              slicesEnd=None,
              digits=5):
        """ Write reconstructed slices to a stack
        of 2-D 32-bit TIFF images.

        Parameters
        -----------
        dataset : ndarray
            Reconstructed values as a 3-D ndarray.

        fileName : str
            Generic name for all TIFF images. Index will
            be added to the end of the name.

        slicesStart : scalar, optional
            First index of the data on first dimension
            of the array.

        slicesEnd : scalar, optional
            Last index of the data on first dimension
            of the array.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff
        """
        # Create new folders.
        dirPath = os.path.dirname(fileName)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Remove TIFF extension.
        if fileName.endswith('tiff'):
            outputFile = fileName.split(".")[-2]

        # Select desired slices from whole data.
        numX, numY, numZ = dataset.shape
        if slicesStart is None:
            slicesStart = 0
        if slicesEnd is None:
            slicesEnd = slicesStart+numX

        # Write data.
        fileIndex = ["" for x in range(digits)]
        for m in range(digits):
            fileIndex[m] = '0' * (digits - m - 1)
        ind = range(slicesStart, slicesEnd)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    fileName = outputFile + fileIndex[n] + str(ind[m]) + '.tiff'
                    break
            img = misc.toimage(dataset[m, :, :])
            #img = misc.toimage(dataset[m, :, :], mode='F')
            img.save(fileName)
