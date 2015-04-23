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


def test_circular_roi():
    assert_equals(
        circular_roi(np.ones((10, 12, 14))).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(circular_roi(np.ones((10, 12, 14)))).sum(),
        0)


def test_correct_air():
    assert_equals(
        correct_air(np.ones((10, 12, 14)), air=1).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(correct_air(np.ones((10, 12, 14)), air=1)).sum(),
        0)


def test_focus_region():
    assert_equals(
        focus_region(np.ones((4, 6, 8)), dia=5).shape,
        (4, 6, 5))
    assert_equals(
        np.isnan(focus_region(np.ones((4, 6, 8)), dia=5)).sum(),
        0)


def test_normalize():
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
        [[[0.5545, 1.0000, 0.7800, 0.8367, 1.0000],
          [0.4646, 1.0000, 0.1522, 1.0000, 0.1818],
          [0.2692, 1.0000, 0.4762, 1.0000, 0.8081],
          [0.4706, 1.0000, 0.2268, 1.0000, 0.2970]],
         [[0.4752, 1.0000, 1.0000, 1.0000, 0.8200],
          [1.0000, 0.5347, 0.8261, 1.0000, 0.2424],
          [0.4615, 1.0000, 0.2286, 1.0000, 0.6869],
          [0.2353, 1.0000, 1.0000, 0.2549, 0.8119]],
         [[0.2376, 1.0000, 0.6600, 0.1633, 0.7400],
          [1.0000, 0.6535, 1.0000, 0.3077, 1.0000],
          [0.8077, 1.0000, 1.0000, 0.2157, 0.8081],
          [1.0000, 0.1980, 0.6392, 1.0000, 0.2772]]], dtype='float32')
    assert_array_almost_equal(
        normalize(synthetic_data(), white, dark, cutoff=1.),
        result, decimal=4)


def test_remove_stripe1():
    assert_equals(
        remove_stripe1(np.ones((10, 12, 14))).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(remove_stripe1(np.ones((10, 12, 14)))).sum(),
        0)


def test_remove_stripe2():
    assert_equals(
        remove_stripe1(np.ones((10, 12, 14))).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(remove_stripe1(np.ones((10, 12, 14)))).sum(),
        0)


def test_remove_zinger():
    assert_equals(
        remove_zinger(np.ones((10, 12, 14)), dif=10).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(remove_zinger(np.ones((10, 12, 14)), dif=10)).sum(),
        0)


def test_retrieve_phase():
    assert_equals(
        retrieve_phase(np.ones((10, 12, 14)), pad=True).shape,
        (10, 12, 14))
    assert_equals(
        np.isnan(retrieve_phase(np.ones((10, 12, 14)), pad=True)).sum(),
        0)
    assert_equals(
        np.isnan(retrieve_phase(np.ones((10, 12, 14)), pad=False)).sum(),
        0)


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
