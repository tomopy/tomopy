# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
from scipy import misc
from reader import Dataset
import logging
logger = logging.getLogger("tomopy")

        
def recon_to_hdf5(TomoObj, output_file=None):
    """ 
    Write reconstructed data to hdf5 file.

    Parameters
    ----------
    output_file : str, optional
        Name of the output file.
        
    Notes
    -----
    If file exists, saves it with a modified name.
    
    If output location is not specified, the data is
    saved inside ``recon`` folder where the input data
    resides. The name of the reconstructed files will 
    be initialized with ``recon``
    """
    if TomoObj.FLAG_DATA_RECON:
        if output_file == None:
            dir_path = os.path.dirname(TomoObj.file_name)
            base_name = os.path.basename(TomoObj.file_name).split(".")[-2]
            output_file = dir_path + "/recon/recon_" + base_name + ".h5"
            logger.warning("generate output file name [ok]")
        output_file =  os.path.abspath(output_file)
        
        # check folder's read permissions.
        dir_path = os.path.dirname(output_file)
        write_access = os.access(dir_path, os.W_OK)
        if write_access:
            logger.debug("save folder directory permissions [ok]")
        else:
            logger.error("save folder directory permissions [failed]")
        
        # Create new folders.
        dir_path = os.path.dirname(output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.debug("new folders generated [ok]")
                
        # Remove HDF5 extension if there is.
        if (output_file.endswith('h5') or
            output_file.endswith('hdf5') or
            output_file.endswith('hdf')):
            file_body = output_file.split(".")[-2]
        else:
            file_body = output_file
        file_name = file_body + '.h5'
    
        # check if file exists.
        if os.path.isfile(file_name):
            logger.warning("saving path check [failed]")
            # genarate new file name.
            ind = 1
            FLAG_SAVE = False
            while not FLAG_SAVE:
                new_file_name = file_body + '-' + str(ind) + '.h5'
                if not os.path.isfile(new_file_name):
                    _export_to_hdf5(new_file_name, TomoObj.data_recon, TomoObj.provenance)
                    FLAG_SAVE = True
                    file_name = new_file_name
                else:
                    ind += 1
            logger.warning("saved as %s [ok]", file_name)
        else:
            _export_to_hdf5(file_name, TomoObj.data_recon, TomoObj.provenance)
            logger.debug("saved as %s [ok]", file_name)
        TomoObj.output_file = output_file
        logger.info("save data at %s [ok]", dir_path)
    else:
        logger.warning("save data [bypassed]")

def recon_to_tiff(TomoObj, output_file=None, x_start=None, x_end=None, digits=5, axis=0):
    """ 
    Write reconstructed data to a stack of tif files.

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
        
    axis : scalar, optional
        Imaages is read along that axis.
    
    Notes
    -----
    If file exists, saves it with a modified name.
    
    If output location is not specified, the data is
    saved inside ``recon`` folder where the input data
    resides. The name of the reconstructed files will
    be initialized with ``recon``
    """
    if TomoObj.FLAG_DATA_RECON:
        if output_file == None:
            dir_path = os.path.dirname(TomoObj.file_name)
            base_name = os.path.basename(TomoObj.file_name).split(".")[-2]
            output_file = dir_path + "/recon/recon_" + base_name + "_"
            logger.warning("generate output file name [ok]")
        output_file =  os.path.abspath(output_file)
        
        # Remove TIFF extension if there is.
        if (output_file.endswith('tif') or
            output_file.endswith('tiff')) :
                output_file = output_file.split(".")[-2]
    
        # check folder's read permissions.
        dir_path = os.path.dirname(output_file)
        write_access = os.access(dir_path, os.W_OK)
        if write_access:
            logger.debug("save folder directory permissions [ok]")
        else:
            logger.error("save folder directory permissions [failed]")
        
        # Create new folders.
        dir_path = os.path.dirname(output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.debug("new folders generated [ok]")

        # Select desired x from whole data.
        num_x, num_y, num_z = TomoObj.data_recon.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            if axis == 0:
                x_end = x_start+num_x
            elif axis == 1:
                x_end = x_start+num_y
            elif axis == 2:
                x_end = x_start+num_z
    
        # Write data.
        file_index = ["" for x in range(digits)]
        for m in range(digits):
            file_index[m] = '0' * (digits - m - 1)
        ind = range(x_start, x_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    file_body = output_file + file_index[n] + str(ind[m])
                    file_name = file_body + '.tif'
                    break
            if axis == 0:
                img = misc.toimage(TomoObj.data_recon[m, :, :])
            elif axis == 1:
                img = misc.toimage(TomoObj.data_recon[:, m, :])
            elif axis == 2:
                img = misc.toimage(TomoObj.data_recon[:, :, m])
            # check if file exists.
            if os.path.isfile(file_name):
                logger.warning("saving path check [failed]")
                # genarate new file name.
                indq = 1
                FLAG_SAVE = False
                while not FLAG_SAVE:
                    new_file_body = file_body + '-' + str(indq)
                    new_file_name = new_file_body + '.tif'
                    if not os.path.isfile(new_file_name):
                        img.save(new_file_name)
                        FLAG_SAVE = True
                        file_name = new_file_name
                    else:
                        indq += 1
                logger.warning("saved as %s [ok]", new_file_name)
            else:
                img.save(file_name)
                logger.debug("saved as %s [ok]", file_name)
            TomoObj.output_file = file_name
        logger.info("save data at %s [ok]", dir_path)
    else:
        logger.warning("save data [bypassed]")

def data_to_hdf5(TomoObj, output_file=None):
    """
    Write raw data to hdf5 file.
    
    Parameters
    ----------
    output_file : str, optional
        Name of the output file.
        
    Notes
    -----
    If file exists, saves it with a modified name.
    
    If output location is not specified, the data is
    saved inside ``data`` folder where the input data
    resides. The name of the reconstructed files will
    be initialized with ``data``
    """
    if TomoObj.FLAG_DATA:
        if output_file == None:
            dir_path = os.path.dirname(TomoObj.file_name)
            base_name = os.path.basename(TomoObj.file_name).split(".")[-2]
            output_file = dir_path + "/data/data_" + base_name + ".h5"
            logger.warning("generate output file name [ok]")
        output_file =  os.path.abspath(output_file)
        
        # check folder's read permissions.
        dir_path = os.path.dirname(output_file)
        write_access = os.access(dir_path, os.W_OK)
        if write_access:
            logger.debug("save folder directory permissions [ok]")
        else:
            logger.error("save folder directory permissions [failed]")
        
        # Create new folders.
        dir_path = os.path.dirname(output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.debug("new folders generated [ok]")
        
        # Remove HDF5 extension if there is.
        if (output_file.endswith('h5') or
            output_file.endswith('hdf5') or
            output_file.endswith('hdf')):
            file_body = output_file.split(".")[-2]
        else:
            file_body = output_file
        file_name = file_body + '.h5'
        
        # check if file exists.
        if os.path.isfile(file_name):
            logger.warning("saving path check [failed]")
            # genarate new file name.
            ind = 1
            FLAG_SAVE = False
            while not FLAG_SAVE:
                new_file_name = file_body + '-' + str(ind) + '.h5'
                if not os.path.isfile(new_file_name):
                    _export_to_hdf5(new_file_name, TomoObj.data, TomoObj.provenance)
                    FLAG_SAVE = True
                    file_name = new_file_name
                else:
                    ind += 1
            logger.warning("saved as %s [ok]", file_name)
        else:
            _export_to_hdf5(file_name, TomoObj.data, TomoObj.provenance)
            logger.debug("saved as %s [ok]", file_name)
        TomoObj.output_file = output_file
        logger.info("save data at %s [ok]", dir_path)
    else:
        logger.warning("save data [bypassed]")

def data_to_tiff(TomoObj, output_file=None, x_start=None, x_end=None, digits=5, axis=1):
    """
    Write raw data to a stack of tif files.
    
    Parameters
    -----------
    output_file : str, optional
        Name of the output file.
    
    x_start : scalar, optional
        First index of the data on first dimension
        of the array.
    
    x_end : scalar, optional
        Last index of the data on first dimension
        of t]he array.
    
    digits : scalar, optional
        Number of digits used for file indexing.
        For example if 4: test_XXXX.tiff
        
    axis : scalar, optional
        Imaages is read along that axis.
        
    Notes
    -----
    If file exists, saves it with a modified name.
    
    If output location is not specified, the data is
    saved inside ``data`` folder where the input data
    resides. The name of the reconstructed files will
    be initialized with ``data``
    """
    if TomoObj.FLAG_DATA:
        if output_file == None:
            dir_path = os.path.dirname(TomoObj.file_name)
            base_name = os.path.basename(TomoObj.file_name).split(".")[-2]
            output_file = dir_path + "/data/data_" + base_name + "_"
            logger.warning("generate output file name [ok]")
        output_file =  os.path.abspath(output_file)
        
        # Remove TIFF extension if there is.
        if (output_file.endswith('tif') or
            output_file.endswith('tiff')) :
            output_file = output_file.split(".")[-2]
            
        # check folder's read permissions.
        dir_path = os.path.dirname(output_file)
        write_access = os.access(dir_path, os.W_OK)
        if write_access:
            logger.debug("save folder directory permissions [ok]")
        else:
            logger.error("save folder directory permissions [failed]")
        
        # Create new folders.
        dir_path = os.path.dirname(output_file)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.debug("new folders generated [ok]")
        
        # Select desired x from whole data.
        num_x, num_y, num_z = TomoObj.data.shape
        if x_start is None:
            x_start = 0
        if x_end is None:
            if axis == 0:
                x_end = x_start+num_x
            elif axis == 1:
                x_end = x_start+num_y
            elif axis == 2:
                x_end = x_start+num_z

        # Write data.
        file_index = ["" for x in range(digits)]
        for m in range(digits):
            file_index[m] = '0' * (digits - m - 1)
        ind = range(x_start, x_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    file_body = output_file + file_index[n] + str(ind[m])
                    file_name = file_body + '.tif'
                    break
            if axis == 0:
                img = misc.toimage(TomoObj.data[m, :, :])
            elif axis == 1:
                img = misc.toimage(TomoObj.data[:, m, :])
            elif axis == 2:
                img = misc.toimage(TomoObj.data[:, :, m])
            # check if file exists.
            if os.path.isfile(file_name):
                logger.warning("saving path check [failed]")
                # genarate new file name.
                indq = 1
                FLAG_SAVE = False
                while not FLAG_SAVE:
                    new_file_body = file_body + '-' + str(indq)
                    new_file_name = new_file_body + '.tif'
                    if not os.path.isfile(new_file_name):
                        img.save(new_file_name)
                        FLAG_SAVE = True
                        file_name = new_file_name
                    else:
                        indq += 1
                logger.warning("saved as %s [ok]", new_file_name)
            else:
                img.save(file_name)
                logger.debug("saved as %s [ok]", file_name)
            TomoObj.output_file = file_name
        logger.info("save data at %s [ok]", dir_path)
    else:
        logger.warning("save data [bypassed]")

def _export_to_hdf5(file_name, data, provenance):
    f = h5py.File(file_name, 'w')
    f.create_dataset('implements', data='exchange')
    exchange_group = f.create_group("processed")
    exchange_group.create_dataset('data', data=data)
    provenance_group = f.create_group("provenance")
    for key, value in provenance.iteritems():
        provenance_group.create_dataset(key, data=str(value))
    f.close()

setattr(Dataset, 'recon_to_hdf5', recon_to_hdf5)
setattr(Dataset, 'recon_to_tiff', recon_to_tiff)
setattr(Dataset, 'data_to_hdf5', data_to_hdf5)
setattr(Dataset, 'data_to_tiff', data_to_tiff)


