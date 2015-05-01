#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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
Module for data I/O.
"""

from __future__ import absolute_import, division, print_function

from tomopy.util import *
from skimage import io as sio
import warnings
import numpy as np
import os
import h5py
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Writer']


class Writer():

    """
    Class for writing data to various data formats.

    Attributes
    ----------
    data : ndarray
        Input data.
    fname : str
        Output file name.
    dtype : str, optional
        The desired data-type for saved data.
    dmin, dmax : float, optional
        Minimum and maximum values in data for scaling before saving.
    overwrite: bool, optional
        if True, the existing files in the reconstruction folder will be
        overwritten with the new ones.
    """

    def __init__(
            self, data, fname='tmp/data.tiff', dtype='float32',
            dmin=None, dmax=None, overwrite=False):

        self.fname = os.path.abspath(fname)
        self.dname = os.path.dirname(fname)
        self.data = data
        self.dtype = dtype
        self.dmax = dmax
        self.dmin = dmin
        self.overwrite = overwrite

        self._range(self.dmin, self.dmax)
        if self.dtype is 'uint8':
            self.data = as_uint8(data)
        elif self.dtype is 'uint16':
            self.data = as_uint16(data)
        elif self.dtype is 'float32':
            self.data = as_float32(data)

        if not os.path.exists(self.dname):
            os.makedirs(self.dname)

        if overwrite is False:
            if os.path.isfile(self.fname):
                self._suggest_new_fname(digit=1)

    def hdf5(self, gname="exchange"):
        """
        Write data to hdf5 file in a specific group.

        Parameters
        ----------
        gname : str, optional
            Path to the group inside hdf5 file where data will be written.
        """
        f = h5py.File(self.fname, 'w')
        ds = f.create_dataset('implements', data="exchange")
        exchangeGrp = f.create_group(gname)
        ds = exchangeGrp.create_dataset('data', data=self.data)
        f.close()

    def tiff(self, stack=True, axis=0, digit=5, start=0):
        """
        Write data to tiff file.

        Parameters
        ----------
        stack : bool, optional
            If True, write 2D images to a stack of files.
        axis : int, optional
            Axis along which stacking is performed.
        start : int, optional
            First index of file in stack for saving.
        digit : int, optional
            Number of digits in indexing stacked files.
        """
        if stack:
            body = self.fname.split(".")[-2]
            ext = '.' + self.fname.split(".")[-1]

            nx, ny, nz = self.data.shape
            if axis == 0:
                end = start + nx
            elif axis == 1:
                end = start + ny
            elif axis == 2:
                end = start + nz

            for m in range(start, end):
                self.fname = body + '_' + '{0:0={1}d}'.format(m, digit) + ext

                if self.overwrite is False:
                    if os.path.isfile(self.fname):
                        self._suggest_new_fname(digit=1)

                if axis == 0:
                    arr = self.data[m - start, :, :]
                elif axis == 1:
                    arr = self.data[:, m - start, :]
                elif axis == 2:
                    arr = self.data[:, :, m - start]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sio.imsave(self.fname, arr, plugin='tifffile')

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sio.imsave(self.fname, self.data, plugin='tifffile')

    def _range(self, dmin=None, dmax=None):
        """
        Change dynamic range of values in data.

        Parameters
        ----------
        dmin, dmax : float, optional
            Mininum and maximum values to rescale data.
        """
        if dmax is None:
            dmax = np.max(self.data)
        if dmin is None:
            dmin = np.min(self.data)
        if dmax < np.max(self.data):
            self.data[self.data > dmax] = dmax
        if dmin > np.min(self.data):
            self.data[self.data < dmin] = dmin

    def _suggest_new_fname(self, digit):
        """
        Suggest new string with an attached (or increased) value indexing
        at the end of a given string.

        For example if "myfile.tiff" exist, it will return "myfile-1.tiff".

        Returns
        -------
        str
            Indexed new string.
        """
        body = self.fname.split(".")[-2]
        ext = '.' + self.fname.split(".")[-1]
        indq = 1
        _flag = False
        while not _flag:
            _body = body + '-' + '{0:0={1}d}'.format(indq, digit)
            if not os.path.isfile(_body + ext):
                _flag = True
                fname = _body
            else:
                indq += 1
        self.fname = fname + ext
