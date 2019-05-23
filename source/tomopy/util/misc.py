#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module for internal utility functions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import warnings
import tifffile
import os
from .. import fft_impl


logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['deprecated', 'write_tiff', 'fft2', 'ifft2', 'fft', 'ifft']


def deprecated(func, msg=None):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    def new_func(*args, **kwargs):
        warnings.simplefilter('once', DeprecationWarning)
        warnings.warn(
            "Deprecated function {}.".format(func.__name__), category=DeprecationWarning)
        return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


def write_tiff(data, fname='tmp/data', digit=None, ext='tiff'):
    """
    Write image data to a tiff file.

    Overwrite existing data and infer data-type from the data.

    Parameters
    ----------
    data : ndarray
        Array data to be saved.
    fname : str
        File name to which the data is saved. ``.tiff`` extension
        will be appended if it doesn't already have one.
    digit : int
        Append this number to fname using a folder e.g. {fname}/{digit}.{ext}
    """
    # Add the extension and digit.
    if digit is not None:
        fname = os.path.join(fname, str(digit))
    if not str(fname).endswith(ext):
        fname = ".".join([fname, ext])
    # Convert to absolute path.
    fname = os.path.abspath(fname)
    # Create the directory if it doesn't exist.
    dname = os.path.dirname(os.path.abspath(fname))
    if not os.path.exists(dname):
        os.makedirs(dname)
    # Save the file.
    tifffile.imsave(fname, data)


if fft_impl == 'mkl_fft':
    import mkl_fft
    def fft(x, n=None, axis=-1, overwrite_input=False, extra_info=None):
        return mkl_fft.fft(x, n=n, axis=axis, overwrite_x=overwrite_input)


    def ifft(x, n=None, axis=-1, overwrite_input=False, extra_info=None):
        return mkl_fft.ifft(x, n=n, axis=axis, overwrite_x=overwrite_input)


    def fft2(x, s=None, axes=(-2,-1), overwrite_input=False, extra_info=None):
        return mkl_fft.fft2(x, shape=s, axes=axes, overwrite_x=overwrite_input)


    def ifft2(x, s=None, axes=(-2,-1), overwrite_input=False, extra_info=None):
        return mkl_fft.ifft2(x, shape=s, axes=axes, overwrite_x=overwrite_input)

else:
    import numpy as np
    def fft(x, n=None, axis=-1, overwrite_input=False, extra_info=None):
        return np.fft.fft(x, n=n, axis=axis)


    def ifft(x, n=None, axis=-1, overwrite_input=False, extra_info=None):
        return np.fft.ifft(x, n=n, axis=axis)


    def fft2(x, s=None, axes=(-2,-1), overwrite_input=False, extra_info=None):
        return np.fft.fft2(x, s=s, axes=axes)


    def ifft2(x, s=None, axes=(-2,-1), overwrite_input=False, extra_info=None):
        return np.fft.ifft2(x, s=s, axes=axes)
