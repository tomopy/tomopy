"""Import X-ray fluorescence tomo datasets.

This module provides a unified interface to reading data into the
format required by tomopy.

XRF data vary widely between facility and beamline. This module
standardises the reading of that data. At a minimum it must
return the data to be reconstructed as a 4-D array [channel,
projection, slice, pixel].

:Author: David J. Vine <http://www.djvine.com>

:Organization: Argonne National Laboratory

:Version: 2015.01.15

Requires
--------
Numpy

Examples
--------

"""
from __future__ import print_function
import numpy as np
import h5py
import ipdb
import sys
import os
import warnings
import skimage.io

def import_aps_2ide(
        filename_pattern=None,
        f_start=None,
        f_end=None,
        f_exclude=None,
        slices_start=None,
        slices_end=None,
        angle_pv='2xfm:m53.VAL',
        fit_method=('XRF_fits', 'XRF_roi_plus', 'XRF_roi'),
        recon_channels=None,
        include_scalers=False):
    """
    Import data acquired at APS beamline 2-ID-E

    Assumes tomo data are acquired with 1 projection/file.

    :Parameters:
        filename_pattern (string) : tomo data filenames with an integer format
            placeholder for specific files.
                e.g. '/home/david/python/tomopy/demo/data/2xfm_{:04d}.h5'
        f_start (int) : Substituted into `filename_pattern` to give the
            first filname of the tomo dataset.
        f_end (int) : Substituted into `filename_pattern` to give the
            last filename of tomo dataset.
        f_exclude (List(int)) : Any files between `f_start` and `f_end`
            which should excluded from dataset returned to tomopy.
        slices_start (int) : specify the range of slices to include int the
            dataset passed to tomopy.
        slices_end (int) : see slices_start.
        angle_pv (string) : Specifies the PV which contains the angle for
            a given projection.
        fit_method (List(string)) : Return a dataset using this fitting method.
            Since the data may not have been fitted yet this parameter is a list
            of fitting methods in order of preference.
            Available options: 'XRF_fits', 'XRF_roi_plus', 'XRF_roi'
        recon_channels (List(str)): XRF elements to include in the dataset
            returned to tomopy.
            if None: defaults to all channels.
        include_scalers (bool) : whether to include scalers in the dataset returned
            to tomopy.

    :Returns:
        numpy.ndarray : 4-D array of [channel, projection, slice, pixel].
        numpy.ndarray : 1-D array of projection angles
        List(string) : a list of channel names
    """
    # Validate input
    if not (filename_pattern or f_start or f_end):
        print('Error: must specify a filename pattern, f_start and f_end.')
        sys.exit(1)


    file_numbers = [n for n in range(f_start, f_end+1) if n not in f_exclude]

    # Validate files and ensure all dims are identical
    invalid_filenumbers = []
    file_dims = []
    for file_number in file_numbers:
        try:
            f_handle = h5py.File(filename_pattern.format(file_number), 'r')
            file_dims.append(f_handle['MAPS/scalers'].shape)
            f_handle.close()
        except IOError:
            invalid_filenumbers.append(file_number)
            print('ERROR: File does not exist.\n\t{:s}'.format(
                filename_pattern.format(file_number)))
    if len(invalid_filenumbers) > 0:
        sys.exit(1)
    if not file_dims.count(file_dims[0]) == len(file_dims):
        print('ERROR: not all projections have the same dimensions')
        print(file_dims)
        sys.exit(1)

    # Validate fit method
    with h5py.File(filename_pattern.format(f_start), 'r') as f_handle:
        if not isinstance(fit_method, list):
            fit_method = list(fit_method)
        for m_idx in range(len(fit_method)):
            method = fit_method[m_idx]
            if method in f_handle['MAPS'].keys():
                dims = f_handle[os.path.join('MAPS', method)].shape[1:]
                break
            else:
                method = None
        if not method:
            print('ERROR: Fitting method not in dataset')
            sys.exit(1)

    # Validate slice range
    if not slices_start:
        slices_start = 0
    else:
        if slices_start<0 or slices_start>dims[0]:
            print('ERROR: slices_start must be in the range 0 <= slices_start < {:d}'.format(dims[0]))
            sys.exit(1)
    if not slices_end:
        slices_end = dims[0]
    else:
        if slices_end<0 or slices_end>dims[0]:
            print('ERROR: slices_end must be in the range 0 <= slices_end < {:d}'.format(dims[0]))
            sys.exit(1)
    # Validate angle PV
    with h5py.File(filename_pattern.format(f_start), 'r') as f_handle:
        try:
            angpv_idx = list(f_handle['MAPS/extra_pvs'].value[0]).index(angle_pv)
        except ValueError:
            print('ERROR: Could not find angle_pv.\nAvailable choices: {:s}'.format(
                f_handle['MAPS/extra_pvs'].value[0].__repr__()))
            sys.exit(1)

    # Validate channels
    with h5py.File(filename_pattern.format(f_start), 'r') as f_handle:
        xf_channels = list(f_handle['MAPS/channel_names'])
        sc_channels = list(f_handle['MAPS/scaler_names'])
        channels = list(f_handle['MAPS/channel_names'].value)
        if include_scalers:
            channels += [sc for sc in f_handle['MAPS/scaler_names'].value \
                if sc.endswith('_ic') or sc.endswith('_cfg') or sc.endswith('_norm') or \
                sc == 'phase']
        if recon_channels: # User has specified which channels to reconstruct.
            # Verify user specified channels exist
            for channel_name in recon_channels:
                if channel_name not in channels:
                    print('ERROR: channel {:s} does not exist in dataset.\nAvailable channels: \
                            {:s}'.format(channel_name, channels.__repr__()))
                    sys.exit(1)
            channels = recon_channels

    # Read in data
    print('Reading in {:d} files.'.format(len(file_numbers)))
    data = np.zeros((len(channels), len(file_numbers), slices_end-slices_start, dims[1]))
    theta = np.zeros(len(file_numbers))
    method = 'MAPS/'+method
    for i, file_number in enumerate(file_numbers):
        with h5py.File(filename_pattern.format(file_number), 'r') as f_handle:
            for j, channel in enumerate(channels):
                if channel in xf_channels:
                    ch_idx = xf_channels.index(channel)
                    h5_str = method
                else:
                    ch_idx = sc_channels.index(channel)
                    h5_str = '/MAPS/scalers'
                data[j, i, :, :] = f_handle[h5_str][ch_idx, slices_start:slices_end, :]
                data[j, i, :, :] = np.where(np.isfinite(data[j, i, :, :]), data[j, i, :, :], 0)
                if i == 0:
                    theta[j] = float(f_handle['MAPS/extra_pvs'][1, angpv_idx])

    print('Reading in files. Done.')
    return data, theta, channels

def xftomo_writer(data, output_file=None, x_start=0,
                 digits=5, axis=0, delete=False,
                 dtype='float32', data_min=None, data_max=None,
                 channel=None, channel_names=None):
    """
    Write a stack of 3-D data to a stack of stack of tif files.

    Parameters
    -----------
    output_file : str, optional
        Name of the output file. Must contain two
        format fields e.g. "{:}" that will be passed the channel name or
        number and the projection number.

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

    channel: int
        Which channel to save. If None, write all.

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
    ipdb.set_trace()
    if output_file is None:
        output_file = "tmp/img_{:d}_{:d}.tif"
    output_file = os.path.abspath(output_file)
    dir_path = os.path.dirname(output_file)

    # Find max min of data for scaling
    if data_max is None:
        data_max = np.max(data)
    if data_min is None:
        data_min = np.min(data)

    if data_max < np.max(data):
        data[data > data_max] = data_max
    if data_min > np.min(data):
        data[data < data_min] = data_min

    if delete:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

    # Create new folders.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Select desired x from whole data.
    num_channels, num_x, num_y, num_z = data.shape
    if axis == 0:
        x_end = x_start + num_x
    elif axis == 1:
        x_end = x_start + num_y
    elif axis == 2:
        x_end = x_start + num_z

    # Write data.
    if channel:
        channels = [channel]
    else:
        channels = range(data.shape[0])
        ind = range(x_start, x_end)
        for channel in channels:
            for m in range(len(ind)):

                if axis == 0:
                    arr = data[channel, m, :, :]
                elif axis == 1:
                    arr = data[channel, :, m, :]
                elif axis == 2:
                    arr = data[channel, :, :, m]

                if dtype is 'uint8':
                    arr = ((arr * 1.0 - data_min) / (data_max - data_min) * 255).astype(
                        'uint8')
                elif dtype is 'uint16':
                    arr = (
                        (arr * 1.0 - data_min) / (data_max - data_min) * 65535).astype(
                        'uint16')
                elif dtype is 'float32':
                    arr = arr.astype('float32')

                if channel_names:
                    channel_name = self.channel_names[channel]
                else:
                    channel_name = channel
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    skimage.io.imsave(output_file.format(channel_name, m), arr, plugin='tifffile')

if __name__ == '__main__':
    data, theta, channel_names = import_aps_2ide(
        filename_pattern='/home/david/python/tomopy/demo/data/tomo/2xfm_{:04d}.h5',
        f_start=100,
        f_end=165,
        f_exclude=[140, 141, 142, 143, 145],
        angle_pv='2xfm:m53.VAL',
        fit_method=['XRF_fits', 'XRF_roi_plus', 'XRF_roi'],
        recon_channels=None
        )
    import matplotlib.pylab as pyl
    pyl.ion()
    from matplotlib import *
    pyl.matshow(data[0,0,:,:])
    ipdb.set_trace()

