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

from __future__ import absolute_import, division, print_function

from tomopy.io.data import *
from tomopy.io.data import (
    _add_index_to_string, _suggest_new_fname,
    _as_uint8, _as_uint16, _as_float32)
import numpy as np
import os
import shutil
import h5py
from nose.tools import assert_equals


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def test_as_uint8():
    arr = np.arange(5, dtype='float32')
    out = _as_uint8(arr)
    assert_equals(out.dtype, 'uint8')
    out = _as_uint8(arr, dmin=1.)
    assert_equals((out == [0, 0, 85, 170, 255]).all(), True)
    out = _as_uint8(arr, dmax=3.)
    assert_equals((out == [0, 85, 170, 255, 255]).all(), True)


def test_as_uint16():
    arr = np.arange(5, dtype='float32')
    out = _as_uint16(arr)
    assert_equals(out.dtype, 'uint16')
    out = _as_uint16(arr, dmin=1.)
    assert_equals((out == [0, 0, 21845, 43690, 65535]).all(), True)
    out = _as_uint16(arr, dmax=3.)
    assert_equals((out == [0, 21845, 43690, 65535, 65535]).all(), True)


def test_remove_neg():
    arr = np.arange(-2, 2, dtype='float32')
    out = remove_neg(arr)
    assert_equals(out[out < 0].size, 0)


def test_remove_nan():
    arr = np.array([np.nan, 1.5, 2., np.nan, 1.], dtype='float')
    out = remove_nan(arr)
    assert_equals(np.isnan(out).sum(), 0)


def test_read_hdf5():
    fname = os.path.join('tomopy', 'data', 'lena.h5')
    out = read_hdf5(fname, '/exchange/data')
    assert_equals(out.shape, (1, 512, 512))


def test_write_hdf5():
    dest = os.path.join('test', 'tmp')
    fname = os.path.join(dest, 'tmp')
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    arr = np.ones((3, 3, 3), dtype='float32')
    write_hdf5(arr, fname)
    f = h5py.File(fname + '.h5', "r")
    assert_equals(f['/exchange/data'][:].shape, (3, 3, 3))
    assert_equals(f['/exchange/data'][:].dtype, 'float32')
    shutil.rmtree(dest)


def test__add_index_to_string():
    out = _add_index_to_string(string='test', ind=12, digit=5)
    assert_equals(out, 'test_00012')


def test__suggest_new_fname():
    out = _suggest_new_fname('lena.tiff')
    assert_equals(out, 'lena-1.tiff')


def test_write_tiff_stack():
    dest = os.path.join('test', 'tmp')
    fname = os.path.join(dest, 'tmp')
    arr = np.ones((1, 2, 3), dtype='float32')
    write_tiff_stack(arr, fname=fname, axis=0, digit=4, dtype='uint8')
    assert_equals(os.path.isfile(fname + '_0000.tiff'), True)
    shutil.rmtree(dest)
    write_tiff_stack(arr, fname=fname, axis=1, digit=5, dtype='uint16')
    assert_equals(os.path.isfile(fname + '_00000.tiff'), True)
    assert_equals(os.path.isfile(fname + '_00001.tiff'), True)
    shutil.rmtree(dest)
    write_tiff_stack(arr, fname=fname, axis=2, id=6, dtype='float32')
    assert_equals(os.path.isfile(fname + '_00006.tiff'), True)
    assert_equals(os.path.isfile(fname + '_00007.tiff'), True)
    assert_equals(os.path.isfile(fname + '_00008.tiff'), True)
    shutil.rmtree(dest)
    write_tiff_stack(arr, fname=fname, axis=2)
    shutil.rmtree(dest)


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
