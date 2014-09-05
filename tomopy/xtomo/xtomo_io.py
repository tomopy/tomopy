# -*- coding: utf-8 -*-
import numpy as np
import warnings
from skimage import io as skimage_io 
import shutil
import os
import h5py


def xtomo_reader(file_name,
                 exchange_rank = 0,
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
                 dark_end=None):
    """
    Read Data Exchange HDF5 file.
    
    Parameters
    ----------
    file_name : str
        Input file.

    exchange_rank : scalar, optional
        exchange rank is added to /exchange to point tomopy to the data to recontruct.
        if rank is not set then the data are raw from the detector and are located under
        exchange = "/exchange/...", if we want to process data that are the result of
        some intemedite processing step then exchange_rank = 1 will direct tomopy 
        to process "/exchange1/...", 

    projections_start, projections_end, projections_step : scalar, optional
        Values of the start, end and step of the projections to
        be used for slicing for the whole data.
    
    slices_start, slices_end, slices_step : scalar, optional
        Values of the start, end and step of the slices to
        be used for slicing for the whole data.
    
    pixels_start, pixels_end, pixels_step : scalar, optional
        Values of the start, end and step of the pixels to
        be used for slicing for the whole data.
    
    white_start, white_end : scalar, optional
        Values of the start and end of the
        slicing for the whole white field shots.
    
    dark_start, dark_end : scalar, optional
        Values of the start and end of the
        slicing for the whole dark field shots.
        
    Examples
    --------
    - Import data, white-field, dark-field and projection angles
      from HDF5 file:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>>
        >>> # Image data
        >>> import pylab as plt
        >>> plt.figure()
        >>> plt.imshow(data[:, 0, :])
        >>> plt.show()
    
    - Import only 4th slice from HDF5 file:

        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile,  slices_start=4, slices_end=5)
        >>> 
        >>> # Image data
        >>> import pylab as plt
        >>> plt.figure()
        >>> plt.imshow(data[:, 0, :])
        >>> plt.show()
    """

    # Start working on checks and stuff.
    file_name = os.path.abspath(file_name)

    # Start reading data.
    f = h5py.File(file_name, "r")

    if exchange_rank > 0:
        exchange_base = 'exchange{:d}'.format(exchange_rank)
        hdfdata = f['/'.join([exchange_base, "data"])]
    else:
        hdfdata = f["exchange/data"]

    num_x, num_y, num_z = hdfdata.shape
    if projections_end is None:
        projections_end = num_x
    if slices_end is None:
        slices_end = num_y
    if pixels_end is None:
        pixels_end = num_z
    data = hdfdata[projections_start:projections_end:projections_step,
                   slices_start:slices_end:slices_step,
                   pixels_start:pixels_end:pixels_step]
    data = np.nan_to_num(data)
    
    try:
        # Now read white fields.
        if exchange_rank > 0:
            exchange_base = 'exchange{:d}'.format(exchange_rank)
            hdfdata = f['/'.join([exchange_base, "data_white"])]
        else:
            hdfdata = f["exchange/data_white"]
        if white_end is None:
            white_end = num_x
        data_white = hdfdata[white_start:white_end,
                             slices_start:slices_end:slices_step,
                             pixels_start:pixels_end:pixels_step]
        data_white = np.nan_to_num(data_white)
    except KeyError:
        data_white = None
        
    try:
        # Now read dark fields. 
        if exchange_rank > 0:
            exchange_base = 'exchange{:d}'.format(exchange_rank)
            hdfdata = f['/'.join([exchange_base, "data_dark"])]
        else:
            hdfdata = f["exchange/data_dark"]
        if dark_end is None:
            dark_end = num_x
        data_dark = hdfdata[dark_start:dark_end,
                            slices_start:slices_end:slices_step,
                            pixels_start:pixels_end:pixels_step]
        data_dark = np.nan_to_num(data_dark)
    except KeyError:
        data_dark = None
    
    try:
        # Read projection angles.
        if exchange_rank > 0:
            exchange_base = 'exchange{:d}'.format(exchange_rank)
            hdfdata = f['/'.join([exchange_base, "theta"])]
        else:
            hdfdata = f["exchange/theta"]
        theta = hdfdata[projections_start:projections_end:projections_step]
        theta = np.nan_to_num(theta)
    except KeyError:
        theta = None
    
    f.close()
    
    return data, data_white, data_dark, theta



def xtomo_writer(data, output_file=None, x_start=0,
                 digits=5, axis=0, overwrite=False, delete=False,
                 dtype='float32', data_min=None, data_max=None):
    """ 
    Write 3-D data to a stack of tif files.

    Parameters
    -----------
    output_file : str, optional
        Name of the output file.

    x_start : scalar, optional
        First index of the data on first dimension
        of the array.

    digits : scalar, optional
        Number of digits used for file indexing.
        For example if 4: test_XXXX.tiff
        
    axis : scalar, optional
        Imaages is read along that axis.
        
    overwrite: bool, optional
        if overwrite=True the existing files in the
        reconstruction folder will be overwritten
        with the new ones.
        
    delete: bool, optional
        if delete=True the reconstruction
        folder and its contents will be deleted.
        
    dtype : bool, optional
        Export data type precision.
        
    data_min, data_max : scalar, optional
        User defined minimum and maximum values
        in the data that will be used to scale 
        the dataset when saving.
    
    Notes
    -----
    If file exists, saves it with a modified name.
    
    If output location is not specified, the data is
    saved inside ``recon`` folder where the input data
    resides. The name of the reconstructed files will
    be initialized with ``recon``
    
    Examples
    --------
    - Save sinogram data:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>> 
        >>> # Save data
        >>> output_file='tmp/slice_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        
    - Save first 16 projections:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, projections_start=0, projections_end=16)
        >>> 
        >>> # Save data
        >>> output_file='tmp/projection_'
        >>> tomopy.xtomo_writer(data, output_file, axis=0)
        >>> print "Images are succesfully saved at " + output_file + '...'
        
    - Save reconstructed slices:
        
        >>> import tomopy
        >>> 
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile)
        >>> 
        >>> # Perform reconstruction
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>> d.center = 661.5
        >>> d.gridrec()
        >>> 
        >>> # Save data
        >>> output_file='tmp/reconstruction_'
        >>> tomopy.xtomo_writer(d.data_recon, output_file, axis=0)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """
    if output_file == None:
        output_file = "tmp/img_" 
    output_file =  os.path.abspath(output_file)
    dir_path = os.path.dirname(output_file)
        
    # Find max min of data for scaling
    if data_max is None:
        data_max = np.max(data)
    if data_min is None:
        data_min = np.min(data)
        
    if data_max < np.max(data):
        data[data>data_max] = data_max
    if data_min > np.min(data):
        data[data<data_min] = data_min
    
    # Remove TIFF extension if there is.
    if (output_file.endswith('tif') or
        output_file.endswith('tiff')) :
            output_file = output_file.split(".")[-2]
  
    if delete:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            
    # Create new folders.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Select desired x from whole data.
    num_x, num_y, num_z = data.shape
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

        # check if file exists.
        if not overwrite:
            if os.path.isfile(file_name):
                # generate new file unique name.
                indq = 1
                FLAG_SAVE = False
                while not FLAG_SAVE:
                    new_file_body = file_body + '-' + str(indq)
                    new_file_name = new_file_body + '.tif'
                    if not os.path.isfile(new_file_name):
                        FLAG_SAVE = True
                        file_name = new_file_name
                    else:
                        indq += 1

        if axis == 0:
            arr = data[m, :, :]
        elif axis == 1:
            arr = data[:, m, :]
        elif axis == 2:
            arr = data[:, :, m]

        if dtype is 'uint8':
            arr = ((arr*1.0 - data_min)/(data_max-data_min)*255).astype('uint8')
        elif dtype is 'uint16':
            arr = ((arr*1.0 - data_min)/(data_max-data_min)*65535).astype('uint16')
        elif dtype is 'float32':
            arr = arr.astype('float32')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage_io.imsave(file_name, arr, plugin='tifffile')

