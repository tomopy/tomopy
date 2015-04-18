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

from tomopy.prep import *
import numpy as np
from nose.tools import assert_equals
from numpy.testing import assert_array_almost_equal


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


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


def test_normalize():
    data = synthetic_data()

    # Synthetic white field data
    white = np.array(
        [[[52., 53., 51., 56., 55.],
          [51., 53., 50., 51., 51.],
          [52., 50., 55., 51., 50.],
          [51., 52., 50., 49., 51.]],
         [[51., 52., 49., 50., 49.],
          [50., 48., 52., 53., 54.],
          [52., 51., 50., 51., 51.],
          [51., 53., 49., 53., 50.]]], dtype='float32')

    # Synthetic dark field data
    dark = np.array(
        [[[1., 3., 0., 4., 2.],
          [1., 0., 5., 0., 3.],
          [0., 0., 0., 0., 1.],
          [0., 2., 1., 0., 0.]]], dtype='float32')

    # Expected result
    result = np.array(
        [[[0.5545, 1.6566, 0.78,   0.8367, 1.02],
          [0.4646, 1.0495, 0.1522, 1.7115, 0.1818],
          [0.2692, 1.0297, 0.4762, 1.0196, 0.8081],
          [0.4706, 1.2277, 0.2268, 1.7451, 0.297]],
         [[0.4752, 1.4343, 1.26,   1.9184, 0.82],
          [1.2525, 0.5347, 0.8261, 1.3077, 0.2424],
          [0.4615, 1.2673, 0.2286, 1.9412, 0.6869],
          [0.2353, 1.0099, 1.5052, 0.2549, 0.8119]],
         [[0.2376, 1.2525, 0.66,   0.1633, 0.74],
          [1.4141, 0.6535, 1.7826, 0.3077, 1.5152],
          [0.8077, 1.9208, 1.4667, 0.2157, 0.8081],
          [1.7647, 0.198,  0.6392, 1.2353, 0.2772]]], dtype='float32')
    assert_array_almost_equal(
        normalize(data, white, dark),
        result, decimal=4)


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


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
