# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
from scipy import misc
from reader import Dataset
import logging
logger = logging.getLogger("tomopy")

        
def write_hdf5(TomoObj, output_file):
    """ Write data to hdf5 file.

    Parameters
    -----------
    output_file : str, optional
        Name of the output file.
    """
    if TomoObj.FLAG_DATA:
        TomoObj.output_file =  os.path.abspath(output_file)
    
        # check if file exists.
        if os.path.isfile(TomoObj.output_file):
            logger.error("another file exists at location")
    
        # check folder's read permissions.
        dir_path = os.path.dirname(TomoObj.output_file)
        write_access = os.access(dir_path, os.W_OK)
        if not write_access:
            logger.error("permission denied to write at location")
        
        # Create new folders.
        dir_path = os.path.dirname(TomoObj.output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        # Write data
        f = h5py.File(TomoObj.output_file, 'w')
        f.create_dataset('implements', data='exchange')
        exchange_group = f.create_group("processed")
        exchange_group.create_dataset('data', data=TomoObj.data_recon)
        f.close()
    
def write_tiff(TomoObj, output_file, x_start=None, x_end=None, digits=5):
    """ Write data to a stack of tiff files.

    Parameters
    -----------
    output_file : str, optional
        Name of the output file.

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
    if TomoObj.FLAG_DATA:
        TomoObj.output_file =  os.path.abspath(output_file)
    
        # check if file exists.
        if os.path.isfile(TomoObj.output_file):
            logger.error("another file exists at location")
    
        # check folder's read permissions.
        dir_path = os.path.dirname(TomoObj.output_file)
        write_access = os.access(dir_path, os.W_OK)
        if not write_access:
            logger.error("permission denied to write at location")
        
        # Create new folders.
        dir_path = os.path.dirname(TomoObj.output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
        # Remove TIFF extension.
        if TomoObj.output_file.endswith('tif'):
            output_file = TomoObj.output_file.split(".")[-2]
    
        # Select desired x from whole data.
        num_x, num_y, num_z = TomoObj.data_recon.shape
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
            img = misc.toimage(TomoObj.data_recon[m, :, :])
            img.save(file_name)

        
setattr(Dataset, 'write_hdf5', write_hdf5)
setattr(Dataset, 'write_tiff', write_tiff)