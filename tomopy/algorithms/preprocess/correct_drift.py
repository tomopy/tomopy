# -*- coding: utf-8 -*-
import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
if os.name == 'nt':
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'lib/libprep.pyd'))
    libprep = ctypes.CDLL(libpath)
else:
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'lib/libprep.so'))
    libprep = ctypes.CDLL(libpath)

# --------------------------------------------------------------------


def correct_drift(data, air_pixels):
    """
    Corrects for drifts in the sinogram.

    It normalizes sinogram such that the left and
    the right boundaries are set to one and
    all intermediate values between the boundaries
    are normalized linearly. It can be used if white
    field is absent.

    Parameters
    ----------
    data : ndarray, float32
        3-D tomographic data with dimensions:
        [projections, slices, pixels]

    air_pixels : scalar, int32
        number of pixels at each boundaries that
        the white field will be approximated
        for normalization.

    Returns
    -------
    output : ndarray
        Normalized data.

    Examples
    --------
    - Auto-normalize and correct for drifts in sinogram:

        >>> import tomopy
        >>>
        >>> # Load data
        >>> myfile = 'demo/data.h5'
        >>> data, white, dark, theta = tomopy.xtomo_reader(myfile, slices_start=0, slices_end=1)
        >>>
        >>> # Save data before correction
        >>> output_file='tmp/before_correction_'
        >>> tomopy.xtomo_writer(data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
        >>>
        >>> # Construct tomo object
        >>> d = tomopy.xtomo_dataset(log='error')
        >>> d.dataset(data, white, dark, theta)
        >>>
        >>> # Perform correction
        >>> d.correct_drift()
        >>>
        >>> # Save data after correction
        >>> output_file='tmp/after_correction_'
        >>> tomopy.xtomo_writer(d.data, output_file, axis=1)
        >>> print "Images are succesfully saved at " + output_file + '...'
    """

    num_projections = np.array(data.shape[0], dtype='int32')
    num_slices = np.array(data.shape[1], dtype='int32')
    num_pixels = np.array(data.shape[2], dtype='int32')

    if air_pixels <= 0:
        return data

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)

    libprep.correct_drift.restype = ctypes.POINTER(ctypes.c_void_p)
    libprep.correct_drift(data.ctypes.data_as(c_float_p),
                          ctypes.c_int(num_projections),
                          ctypes.c_int(num_slices),
                          ctypes.c_int(num_pixels),
                          ctypes.c_int(air_pixels))
    return data
