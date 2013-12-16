# -*- coding: utf-8 -*-
# file_name: file_types.py
import h5py
import os
import numpy as np
from scipy import misc
import PIL.Image as Image
from pyhdf import SD
from file_interface import FileInterface

class Hdf5(FileInterface):
    def read(self, file_name,
             array_name=None,
             projections_start=None,
             projections_end=None,
             projections_step=None,
             slices_start=None,
             slices_end=None,
             slices_step=None,
             pixels_start=None,
             pixels_end=None,
             pixels_step=None):
        """ Read 3-D tomographic data from hdf5 file.

        Opens ``file_name`` and reads the contents
        of the array specified by ``array_name`` in
        the specified group of the HDF file.

        Parameters
        ----------
        file_name : str
            Input HDF file.

        array_name : str
            Name of the array to be read at exchange group.

        projections_start, projections_end, projections_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        Returns
        -------
        out : ndarray
            Returns the data as a matrix.
        """
        # Read data from file.
        f = h5py.File(file_name, 'r')
        hdfdata = f[array_name]

        # Select desired slices from whole data.
        numProjections, num_slices, num_pixels = hdfdata.shape
        if projections_start is None:
            projections_start = 0
        if projections_end is None:
            projections_end = numProjections
        if projections_step is None:
            projections_step = 1
        if slices_start is None:
            slices_start = 0
        if slices_end is None:
            slices_end = num_slices
        if slices_step is None:
            slices_step = 1
        if pixels_start is None:
            pixels_start = 0
        if pixels_end is None:
            pixels_end = num_pixels
        if pixels_step is None:
            pixels_step = 1

        # Construct dataset from desired slices.
        dataset = hdfdata[projections_start:projections_end:projections_step,
                          slices_start:slices_end:slices_step,
                          pixels_start:pixels_end:pixels_step]
        f.close()
        return dataset

    def write(self, dataset, file_name, array_name):
        """ Write data to hdf5 file.

        Parameters
        -----------
        dataset : ndarray
            Input values.

        file_name : str
            Name of the output HDF file.

        array_name : str, optional
            Name of the group that the data will be put
            under the exchange group of the HDF file.
        """
        # Create new folders.
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Write data
        f = h5py.File(file_name, 'w')
        f.create_dataset('implements', data='exchange')
        exchange_group = f.create_group("exchange")
        exchange_group.create_dataset(array_name, data=dataset)
        f.close()


class Hdf4(FileInterface):
    def read(self, file_name,
             array_name=None,
             slices_start=None,
             slices_end=None,
             slices_step=None,
             pixels_start=None,
             pixels_end=None,
             pixels_step=None):
        """ Read 2-D tomographic data from hdf4 file.

        Opens ``file_name`` and reads the contents
        of the array specified by ``array_name`` in
        the specified group of the HDF file.

        Parameters
        ----------
        file_name : str
            Input HDF file.

        array_name : str
            Name of the array to be read at exchange group.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        Returns
        -------
        out : ndarray
            Returns the data as a matrix.
        """
        # Read data from file.
        f = SD.SD(file_name)
        sds = f.select(array_name)
        hdfdata = sds.get()
        hdfdata = hdfdata.reshape(hdfdata.shape[1],
                                  hdfdata.shape[0])

        # Select desired slices from whole data.
        num_slices, num_pixels = hdfdata.shape
        if slices_start is None:
            slices_start = 0
        if slices_end is None:
            slices_end = num_slices
        if slices_step is None:
            slices_step = 1
        if pixels_start is None:
            pixels_start = 0
        if pixels_end is None:
            pixels_end = num_pixels
        if pixels_step is None:
            pixels_step = 1

        # Construct dataset from desired slices.
        dataset = hdfdata[slices_start:slices_end:slices_step,
                          pixels_start:pixels_end:pixels_step]
        return dataset

    def write(self):
        pass


class Tiff(FileInterface):
    def read(self, file_name, dtype='uint16',
             slices_start=None,
             slices_end=None,
             slices_step=None,
             pixels_start=None,
             pixels_end=None,
             pixels_step=None):
        """Read TIFF files.

        Parameters
        ----------
        file_name : str
            Name of the input TIFF file.

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        Returns
        -------
        out : ndarray
            Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        im = Image.open(file_name)
        out = np.fromstring(im.tostring(), dtype).reshape(tuple(list(im.size[::-1])))

        # Select desired slices from whole data.
        num_slices, num_pixels = out.shape
        if slices_start is None:
            slices_start = 0
        if slices_end is None:
            slices_end = num_slices
        if slices_step is None:
            slices_step = 1
        if pixels_start is None:
            pixels_start = 0
        if pixels_end is None:
            pixels_end = num_pixels
        if pixels_step is None:
            pixels_step = 1
        return out[slices_start:slices_end:slices_step,
                   pixels_start:pixels_end:pixels_step]

    def write(self, dataset,
              file_name,
              slices_start=None,
              slices_end=None,
              digits=5):
        """ Write reconstructed slices to a stack
        of 2-D 32-bit TIFF images.

        Parameters
        -----------
        dataset : ndarray
            Reconstructed values as a 3-D ndarray.

        file_name : str
            Generic name for all TIFF images. Index will
            be added to the end of the name.

        slices_start : scalar, optional
            First index of the data on first dimension
            of the array.

        slices_end : scalar, optional
            Last index of the data on first dimension
            of the array.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff
        """
        # Create new folders.
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Remove TIFF extension.
        if file_name.endswith('tiff'):
            output_file = file_name.split(".")[-2]

        # Select desired slices from whole data.
        numx, numy, numz = dataset.shape
        if slices_start is None:
            slices_start = 0
        if slices_end is None:
            slices_end = slices_start+numx

        # Write data.
        file_index = ["" for x in range(digits)]
        for m in range(digits):
            file_index[m] = '0' * (digits - m - 1)
        ind = range(slices_start, slices_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    file_name = output_file + file_index[n] + str(ind[m]) + '.tiff'
                    break
            img = misc.toimage(dataset[m, :, :])
            #img = misc.toimage(dataset[m, :, :], mode='F')
            img.save(file_name)
