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

from tomopy.misc.corr import *
import numpy as np
from nose.tools import assert_equals
from numpy.testing import assert_array_almost_equal


def synthetic_data():
    """
    Generate a synthetic data.
    """
    data = np.array(
        [[[29., 85., 39., 45., 53.],
          [24., 53., 12., 89., 12.],
          [14., 52., 25., 52., 41.],
          [24., 64., 12., 89., 15.]],
         [[25., 74., 63., 98., 43.],
          [63., 27., 43., 68., 15.],
          [24., 64., 12., 99., 35.],
          [12., 53., 74., 13., 41.]],
         [[13., 65., 33., 12., 39.],
          [71., 33., 87., 16., 78.],
          [42., 97., 77., 11., 41.],
          [90., 12., 32., 63., 14.]]], dtype='float32')
    return data


def test_median_filter():
    data = synthetic_data()

    # Test filtering on axis 0
    result = np.array(
        [[[29., 39., 45., 45., 53.],
          [29., 29., 52., 41., 45.],
          [24., 24., 52., 25., 41.],
          [24., 24., 52., 25., 41.]],
         [[27., 63., 68., 63., 43.],
          [27., 43., 64., 43., 43.],
          [27., 43., 53., 41., 35.],
          [24., 53., 53., 41., 41.]],
         [[33., 33., 33., 33., 39.],
          [42., 65., 33., 39., 39.],
          [71., 71., 33., 41., 41.],
          [90., 42., 32., 32., 14.]]], dtype='float32')
    assert_array_almost_equal(
        median_filter(data, axis=0), result)

    # Test filtering on axis 1
    result = np.array(
        [[[29., 39., 63., 45., 53.],
          [27., 27., 53., 15., 15.],
          [24., 25., 52., 41., 41.],
          [24., 24., 64., 15., 15.]],
         [[29., 39., 63., 43., 43.],
          [53., 43., 43., 43., 16.],
          [42., 42., 52., 41., 41.],
          [24., 32., 53., 32., 15.]],
         [[25., 33., 63., 39., 39.],
          [63., 63., 33., 68., 68.],
          [42., 64., 77., 41., 41.],
          [53., 32., 32., 32., 14.]]], dtype='float32')
    assert_array_almost_equal(
        median_filter(data, axis=1), result)

    # Test filtering on axis 2
    result = np.array(
        [[[29., 74., 39., 68., 43.],
          [24., 53., 25., 68., 41.],
          [24., 53., 12., 89., 15.],
          [24., 64., 12., 89., 35.]],
         [[25., 65., 39., 45., 43.],
          [25., 64., 39., 52., 41.],
          [24., 53., 32., 63., 35.],
          [24., 53., 32., 63., 35.]],
         [[25., 65., 43., 16., 39.],
          [42., 65., 63., 16., 41.],
          [63., 33., 74., 16., 41.],
          [42., 53., 32., 63., 35.]]], dtype='float32')
    assert_array_almost_equal(
        median_filter(data, axis=2), result)


def test_remove_neg():
    arr = np.arange(-2, 2, dtype='float32')
    out = remove_neg(arr)
    assert_equals(out[out < 0].size, 0)


def test_remove_nan():
    arr = np.array([np.nan, 1.5, 2., np.nan, 1.], dtype='float')
    out = remove_nan(arr)
    assert_equals(np.isnan(out).sum(), 0)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
