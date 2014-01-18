# -*- coding: utf-8 -*-
# file_name: file_types.py
import h5py
import os
import numpy as np
from scipy import misc
import PIL.Image as Image
from pyhdf import SD
from file_interface import FileInterface

import dataio.xradia.xradia_xrm as xradia
import dataio.xradia.data_struct as dstruct
import dataio.data_spe as spe


class Hdf5(FileInterface):
    def read(self, file_name,
             array_name=None,
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None,
             z_start=None,
             z_end=None,
             z_step=None):
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

        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        y_start, y_end, y_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        z_start, z_end, z_step : scalar, optional
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

        # Select desired y from whole data.
        num_x, num_y, num_z = hdfdata.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = num_z
        if z_step is None:
            z_step = 1

        # Construct dataset from desired y.
        dataset = hdfdata[x_start:x_end:x_step,
                          y_start:y_end:y_step,
                          z_start:z_end:z_step]
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
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None):
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

        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        y_start, y_end, y_step : scalar, optional
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
        hdfdata = hdfdata.reshape(hdfdata.shape[0],
                                  hdfdata.shape[1])

        # Select desired x from whole data.
        num_x, num_y = hdfdata.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1

        # Construct dataset from desired x.
        dataset = hdfdata[x_start:x_end:x_step,
                          y_start:y_end:y_step]
        return dataset

    def write(self):
        pass


class Tiff(FileInterface):
    def read(self, file_name, dtype='uint16',
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None):
        """Read TIFF files.

        Parameters
        ----------
        file_name : str
            Name of the input TIFF file.

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        y_start, y_end, y_step : scalar, optional
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

        # Select desired x from whole data.
        num_x, num_y = out.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1
        return out[x_start:x_end:x_step,
                   y_start:y_end:y_step]

    def write(self, dataset,
              file_name,
              x_start=None,
              x_end=None,
              digits=5):
        """ Write reconstructed x to a stack
        of 2-D 32-bit TIFF images.

        Parameters
        -----------
        dataset : ndarray
            Reconstructed values as a 3-D ndarray.

        file_name : str
            Generic name for all TIFF images. Index will
            be added to the end of the name.

        x_start : scalar, optional
            First index of the data on first dimension
            of the array.

        x_end : scalar, optional
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

        # Select desired x from whole data.
        num_x, num_y, num_z = dataset.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = x_start+num_x

        # Write data.
        file_index = ["" for x in range(digits)]
        for m in range(digits):
            file_index[m] = '0' * (digits - m - 1)
        ind = range(x_start, x_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    file_name = output_file + file_index[n] + str(ind[m]) + '.tiff'
                    break
            img = misc.toimage(dataset[m, :, :])
            #img = misc.toimage(dataset[m, :, :], mode='F')
            img.save(file_name)

class Txrm(FileInterface):
    def read(self, file_name,
             array_name='Image',
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None,
             z_start=None,
             z_end=None,
             z_step=None
             ):
        """ Read 3-D tomographic data from a txrm file and the background/reference image for an xrm files.

        Opens ``file_name`` and copy into an array its content;
                this is can be a series/scan of tomographic projections (if file_name extension is ``txrm``) or
                a series of backgroud/reference images if the file_name extension is ``xrm``
        
        Parameters
        ----------
        file_name : str
            Input txrm or xrm file.
            
        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        y_start, y_end, y_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        z_start, z_end, z_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        Returns
        -------
        out : array
            Returns the data as a matrix.
        """
        verbose = True
        imgname = array_name
        reader = xradia.xrm()
        array = dstruct

        # Read data from file.
        if file_name.endswith('txrm'):
            if verbose: print "reading projections ... "
            reader.read_txrm(file_name,array)
            num_x, num_y, num_z = np.shape(array.exchange.data)
            if verbose:
                print "done reading ", num_z, " projections images of (", num_x,"x", num_y, ") pixels"

        # Select desired y from whole data.
        # num_x, num_y, num_z = hdfdata.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = num_z
        if z_step is None:
            z_step = 1

        # Construct dataset from desired y.
        dataset = array.exchange.data[x_start:x_end:x_step,
                          y_start:y_end:y_step,
                          z_start:z_end:z_step]
        return dataset

    def write(self):
        pass

class Xrm(FileInterface):
    def read(self, file_name,
             array_name='Image',
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None,
             z_start=None,
             z_end=None,
             z_step=None
             ):
        """ Read 3-D tomographic data from a txrm file and the background/reference image for an xrm files.

        Opens ``file_name`` and copy into an array its content;
                this is can be a series/scan of tomographic projections (if file_name extension is ``txrm``) or
                a series of backgroud/reference images if the file_name extension is ``xrm``
        
        Parameters
        ----------
        file_name : str
            Input txrm or xrm file.
            
        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        y_start, y_end, y_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        z_start, z_end, z_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        Returns
        -------
        out : array
            Returns the data as a matrix.
        """
        verbose = True
        imgname = array_name
        reader = xradia.xrm()
        array = dstruct

        # Read data from file.
                    
        if file_name.endswith('xrm'):
            if verbose: print "reading reference images ... "
            reader.read_xrm(file_name,array)
            num_x, num_y, num_z = np.shape(array.exchange.data)
            if verbose:
                print "done reading ", num_z, " reference images of (", num_x,"x", num_y, ") pixels"
                
        # Select desired y from whole data.
        # num_x, num_y, num_z = hdfdata.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = num_z
        if z_step is None:
            z_step = 1

        # Construct dataset from desired y.
        dataset = array.exchange.data[x_start:x_end:x_step,
                          y_start:y_end:y_step,
                          z_start:z_end:z_step]
        return dataset

    def write(self):
        pass

class Spe(FileInterface):
    def read(self, file_name,
             #array_name='Image',
             x_start=None,
             x_end=None,
             x_step=None,
             y_start=None,
             y_end=None,
             y_step=None,
             z_start=None,
             z_end=None,
             z_step=None
             ):
        """ Read 3-D tomographic data from a spe file and the background/reference image for an xrm files.

        Opens ``file_name`` and copy into an array its content;
                this is can be a series/scan of tomographic projections (if file_name extension is ``txrm``) or
                a series of backgroud/reference images if the file_name extension is ``xrm``
        
        Parameters
        ----------
        file_name : str
            Input txrm or xrm file.
            
        x_start, x_end, x_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        y_start, y_end, y_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        z_start, z_end, z_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole array.

        Returns
        -------
        out : array
            Returns the data as a matrix.
        """
        verbose = True
        #imgname = array_name
        spe_data = spe.PrincetonSPEFile(file_name)
        #array = dstruct
        if verbose: print file_name
        if verbose: print spe_data
        # Read data from file.
        if file_name.endswith('SPE'):
            if verbose: print "reading data ... "
            array = spe_data.getData()
            #reader.openFile(file_name)
            num_x, num_y, num_z = np.shape(array)
            if verbose:
                print "done reading ", num_z, " images of (", num_x,"x", num_y, ") pixels"

        # Select desired y from whole data.
        # num_x, num_y, num_z = hdfdata.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            x_end = num_x
        if x_step is None:
            x_step = 1
        if y_start is None:
            y_start = 0
        if y_end is None:
            y_end = num_y
        if y_step is None:
            y_step = 1
        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = num_z
        if z_step is None:
            z_step = 1

        # Construct dataset from desired y.
        dataset = array[x_start:x_end:x_step,
                          y_start:y_end:y_step,
                          z_start:z_end:z_step]
        return dataset

    def write(self):
        pass
