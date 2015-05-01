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

from tomopy.io.writer import *
import numpy as np
import h5py
import os
import shutil
from nose.tools import assert_equals


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def test_Writer_hdf5():
    dest = os.path.join('test', 'tmp')
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)
    fname = os.path.join(dest, 'test.h5')
    gname = os.path.join('exchange', 'data')
    arr = np.ones((3, 3, 3), dtype='float32')
    Writer(arr, os.path.join(dest, 'test.h5')).hdf5()
    f = h5py.File(os.path.join(dest, 'test.h5'), "r")
    assert_equals(
        f[gname][:].shape,
        (3, 3, 3))
    assert_equals(
        f[gname][:].dtype,
        'float32')
    f.close()
    shutil.rmtree(dest)


def test_Writer_tiff():
    dest = os.path.join('test', 'tmp')
    bname = os.path.join(dest, 'test')
    fname = os.path.join(dest, 'test.tiff')
    arr = np.ones((1, 2, 3), dtype='float32')
    Writer(arr, fname, dtype='uint8').tiff(axis=0, digit=4)
    assert_equals(
        os.path.isfile(bname + '_0000.tiff'),
        True)
    shutil.rmtree(dest)
    Writer(arr, fname, dtype='uint16').tiff(axis=1, digit=5)
    assert_equals(
        os.path.isfile(bname + '_00000.tiff'),
        True)
    assert_equals(
        os.path.isfile(bname + '_00001.tiff'),
        True)
    shutil.rmtree(dest)
    Writer(arr, fname, dtype='float32').tiff(axis=2, start=9)
    assert_equals(
        os.path.isfile(bname + '_00009.tiff'),
        True)
    assert_equals(
        os.path.isfile(bname + '_00010.tiff'),
        True)
    assert_equals(
        os.path.isfile(bname + '_00011.tiff'),
        True)
    shutil.rmtree(dest)


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
