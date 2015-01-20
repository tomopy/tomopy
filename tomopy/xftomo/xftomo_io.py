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

def import_aps_2ide(
        filename_pattern=None,
        f_start=None,
        f_end=None,
        f_exclude=None,
        slices_start=None,
        slices_end=None,
        angle_pv='2xfm:m53.VAL',
        fit_method=('XRF_fits', 'XRF_roi_plus', 'XRF_roi'),
        recon_channels=None):
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
        if slices_start<0 or slices_start>dim[0]:
            print('ERROR: slices_start must be in the range 0 <= slices_start < {:d}'.format(dim[0]))
            sys.exit(1)
    if not slices_end:
        slices_end = dim[0]
    else:
        if slices_end<0 or slices_end>dim[0]:
            print('ERROR: slices_end must be in the range 0 <= slices_end < {:d}'.format(dim[0]))
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
        channels = list(f_handle['MAPS/channel_names'].value) + \
                [sc for sc in f_handle['MAPS/scaler_names'].value \
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


    return data, theta, channels

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

