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

from tomopy.recon import *
import numpy as np
import os
import shutil
from nose.tools import assert_equals
from numpy.testing import assert_array_almost_equal


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def synthetic_tomo():
    """
    Return synthetic tomographic data of a uniform square object.

    Returns
    -------
    ndarray
        3D tomographic data.
    array
        Corresponding projection angles.
    """
    theta = [0.0000, 0.2618, 0.5236, 0.7854]
    tomo = [[[0.0000, 4.0000, 4.0000, 4.0000, 4.0000, 0.0000],
             [0.0000, 4.0000, 4.0000, 4.0000, 4.0000, 0.0000]],
            [[0.0000, 3.8208, 4.1444, 4.1414, 3.8166, 0.0000],
             [0.0000, 3.8208, 4.1444, 4.1414, 3.8166, 0.0000]],
            [[0.5737, 2.8844, 4.6220, 4.6211, 2.8167, 0.5039],
             [0.5737, 2.8844, 4.6220, 4.6211, 2.8167, 0.5039]],
            [[0.7071, 2.6517, 4.5962, 4.5962, 2.6517, 0.7071],
             [0.7071, 2.6517, 4.5962, 4.5962, 2.6517, 0.7071]]]
    return tomo, theta


def test_art():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        art(tomo, theta, num_iter=4),
        [[[0.2617, 0.6431, 0.5990, 0.4459, 0.4092, 0.0604],
          [0.1397, 0.7947, 0.6867, 0.5436, 0.5093, -0.1896],
          [0.0234, 0.8696, 0.6586, 0.7138, 0.7703, -0.2372],
          [-0.1156, 0.7719, 0.7209, 0.6811, 0.9113, 0.0287],
          [-0.2638, 0.5761, 0.6183, 0.7735, 0.8224, 0.1457],
          [0.0328, 0.3805, 0.5393, 0.6634, 0.6821, 0.3600]],
         [[0.2617, 0.6431, 0.5990, 0.4459, 0.4092, 0.0604],
          [0.1397, 0.7947, 0.6867, 0.5436, 0.5093, -0.1896],
          [0.0234, 0.8696, 0.6586, 0.7138, 0.7703, -0.2372],
          [-0.1156, 0.7719, 0.7209, 0.6811, 0.9113, 0.0287],
          [-0.2638, 0.5761, 0.6183, 0.7735, 0.8224, 0.1457],
          [0.0328, 0.3805, 0.5393, 0.6634, 0.6821, 0.3600]]],
        decimal=4)


def test_bart():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        bart(tomo, theta, num_iter=4),
        [[[0.4168, 0.7387, 0.6132, 0.5395, 0.2556, -0.3265],
          [0.2361, 0.7709, 0.6897, 0.5733, 0.5526, -0.1612],
          [0.0657, 0.7390, 0.6953, 0.7003, 0.6447, -0.1046],
          [-0.0695, 0.6693, 0.6939, 0.7013, 0.7602, 0.0581],
          [-0.1411, 0.5820, 0.6072, 0.7044, 0.7752, 0.2645],
          [-0.3514, 0.2355, 0.5748, 0.6512, 0.7456, 0.4288]],
         [[0.4168, 0.7387, 0.6132, 0.5395, 0.2556, -0.3265],
          [0.2361, 0.7709, 0.6897, 0.5733, 0.5526, -0.1612],
          [0.0657, 0.7390, 0.6953, 0.7003, 0.6447, -0.1046],
          [-0.0695, 0.6693, 0.6939, 0.7013, 0.7602, 0.0581],
          [-0.1411, 0.5820, 0.6072, 0.7044, 0.7752, 0.2645],
          [-0.3514, 0.2355, 0.5748, 0.6512, 0.7456, 0.4288]]],
        decimal=4)


def test_gridrec():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        gridrec(tomo, theta),
        [[[0.7080, 3.6484, 2.5953, 3.3888, 0.1789, -46.9959],
          [-0.1904, 1.2394, 1.0513, 1.6344, 1.6720, -14.3921],
          [-0.3596, 0.5610, 0.5913, 1.0264, 1.8171, -5.2543],
          [-0.3409, 0.2287, 0.3354, 0.6430, 1.5613, -0.9358],
          [-0.2456, 0.0566, 0.1640, 0.3655, 1.1149, 0.9159],
          [-0.1406, -0.0126, 0.0555, 0.1727, 0.6465, 1.2073]],
         [[0.7080, 3.6484, 2.5953, 3.3888, 0.1789, -46.9959],
          [-0.1904, 1.2394, 1.0513, 1.6344, 1.6720, -14.3921],
          [-0.3596, 0.5610, 0.5913, 1.0264, 1.8171, -5.2543],
          [-0.3409, 0.2287, 0.3354, 0.6430, 1.5613, -0.9358],
          [-0.2456, 0.0566, 0.1640, 0.3655, 1.1149, 0.9159],
          [-0.1406, -0.0126, 0.0555, 0.1727, 0.6465, 1.2073]]],
        decimal=4)


def test_mlem():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        mlem(tomo, theta, num_iter=4),
        [[[0.2037, 0.7597, 0.6294, 0.5189, 0.1222, 0.0000],
          [0.0888, 0.8133, 0.6948, 0.5972, 0.5054, 0.0020],
          [0.0298, 0.8011, 0.7010, 0.6993, 0.7008, 0.0066],
          [0.0113, 0.7138, 0.7070, 0.7065, 0.8199, 0.0300],
          [0.0039, 0.5617, 0.6259, 0.7143, 0.8281, 0.0952],
          [0.0000, 0.1199, 0.5649, 0.6613, 0.7784, 0.2078]],
         [[0.2037, 0.7597, 0.6294, 0.5189, 0.1222, 0.0000],
          [0.0888, 0.8133, 0.6948, 0.5972, 0.5054, 0.0020],
          [0.0298, 0.8011, 0.7010, 0.6993, 0.7008, 0.0066],
          [0.0113, 0.7138, 0.7070, 0.7065, 0.8199, 0.0300],
          [0.0039, 0.5617, 0.6259, 0.7143, 0.8281, 0.0952],
          [0.0000, 0.1199, 0.5649, 0.6613, 0.7784, 0.2078]]],
        decimal=4)


def test_osem():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        osem(tomo, theta, num_iter=4),
        [[[0.2037, 0.7597, 0.6294, 0.5189, 0.1222, 0.0000],
          [0.0888, 0.8133, 0.6948, 0.5972, 0.5054, 0.0020],
          [0.0298, 0.8011, 0.7010, 0.6993, 0.7008, 0.0066],
          [0.0113, 0.7138, 0.7070, 0.7065, 0.8199, 0.0300],
          [0.0039, 0.5617, 0.6259, 0.7143, 0.8281, 0.0952],
          [0.0000, 0.1199, 0.5649, 0.6613, 0.7784, 0.2078]],
         [[0.2037, 0.7597, 0.6294, 0.5189, 0.1222, 0.0000],
          [0.0888, 0.8133, 0.6948, 0.5972, 0.5054, 0.0020],
          [0.0298, 0.8011, 0.7010, 0.6993, 0.7008, 0.0066],
          [0.0113, 0.7138, 0.7070, 0.7065, 0.8199, 0.0300],
          [0.0039, 0.5617, 0.6259, 0.7143, 0.8281, 0.0952],
          [0.0000, 0.1199, 0.5649, 0.6613, 0.7784, 0.2078]]],
        decimal=4)


def test_ospml_hybrid():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        ospml_hybrid(tomo, theta, num_iter=4),
        [[[0.4601, 0.5979, 0.6191, 0.5004, 0.2396, 0.0840],
          [0.3756, 0.5970, 0.6404, 0.5526, 0.3562, 0.1122],
          [0.2750, 0.5612, 0.6476, 0.6245, 0.4771, 0.1795],
          [0.1915, 0.4891, 0.6321, 0.6506, 0.5596, 0.2724],
          [0.1219, 0.3744, 0.5681, 0.6518, 0.6044, 0.3827],
          [0.0885, 0.2469, 0.5210, 0.6355, 0.6105, 0.4681]],
         [[0.4601, 0.5979, 0.6191, 0.5004, 0.2396, 0.0840],
          [0.3756, 0.5970, 0.6404, 0.5526, 0.3562, 0.1122],
          [0.2750, 0.5612, 0.6476, 0.6245, 0.4771, 0.1795],
          [0.1915, 0.4891, 0.6321, 0.6506, 0.5596, 0.2724],
          [0.1219, 0.3744, 0.5681, 0.6518, 0.6044, 0.3827],
          [0.0885, 0.2469, 0.5210, 0.6355, 0.6105, 0.4681]]],
        decimal=4)


def test_ospml_quad():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        ospml_quad(tomo, theta, num_iter=4),
        [[[0.4712, 0.5889, 0.6148, 0.4916, 0.2473, 0.0926],
          [0.3921, 0.5801, 0.6347, 0.5416, 0.3438, 0.1265],
          [0.2964, 0.5342, 0.6399, 0.6132, 0.4504, 0.1983],
          [0.2109, 0.4619, 0.6211, 0.6426, 0.5321, 0.2927],
          [0.1368, 0.3596, 0.5558, 0.6460, 0.5866, 0.3985],
          [0.0975, 0.2554, 0.5098, 0.6305, 0.6006, 0.4793]],
         [[0.4712, 0.5889, 0.6148, 0.4916, 0.2473, 0.0926],
          [0.3921, 0.5801, 0.6347, 0.5416, 0.3438, 0.1265],
          [0.2964, 0.5342, 0.6399, 0.6132, 0.4504, 0.1983],
          [0.2109, 0.4619, 0.6211, 0.6426, 0.5321, 0.2927],
          [0.1368, 0.3596, 0.5558, 0.6460, 0.5866, 0.3985],
          [0.0975, 0.2554, 0.5098, 0.6305, 0.6006, 0.4793]]],
        decimal=4)


def test_pml_hybrid():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        pml_hybrid(tomo, theta, num_iter=4),
        [[[0.3031, 0.6773, 0.6339, 0.5229, 0.1637, 0.0000],
          [0.1547, 0.6909, 0.6765, 0.5914, 0.4646, 0.0031],
          [0.0552, 0.6862, 0.6656, 0.6791, 0.6247, 0.0104],
          [0.0179, 0.6336, 0.6904, 0.6704, 0.6818, 0.0536],
          [0.0060, 0.4998, 0.6077, 0.6880, 0.7021, 0.1636],
          [0.0000, 0.1619, 0.5569, 0.6553, 0.6940, 0.3067]],
         [[0.3031, 0.6773, 0.6339, 0.5229, 0.1637, 0.0000],
          [0.1547, 0.6909, 0.6765, 0.5914, 0.4646, 0.0031],
          [0.0552, 0.6862, 0.6656, 0.6791, 0.6247, 0.0104],
          [0.0179, 0.6336, 0.6904, 0.6704, 0.6818, 0.0536],
          [0.0060, 0.4998, 0.6077, 0.6880, 0.7021, 0.1636],
          [0.0000, 0.1619, 0.5569, 0.6553, 0.6940, 0.3067]]],
        decimal=4)


def test_pml_quad():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        pml_quad(tomo, theta, num_iter=4),
        [[[0.3175, 0.6655, 0.6336, 0.5204, 0.1711, -0.0000],
          [0.1691, 0.6753, 0.6756, 0.5876, 0.4500, 0.0035],
          [0.0636, 0.6636, 0.6654, 0.6766, 0.6017, 0.0120],
          [0.0205, 0.6102, 0.6879, 0.6702, 0.6584, 0.0613],
          [0.0067, 0.4814, 0.6032, 0.6872, 0.6854, 0.1788],
          [-0.0000, 0.1699, 0.5525, 0.6548, 0.6814, 0.3219]],
         [[0.3175, 0.6655, 0.6336, 0.5204, 0.1711, -0.0000],
          [0.1691, 0.6753, 0.6756, 0.5876, 0.4500, 0.0035],
          [0.0636, 0.6636, 0.6654, 0.6766, 0.6017, 0.0120],
          [0.0205, 0.6102, 0.6879, 0.6702, 0.6584, 0.0613],
          [0.0067, 0.4814, 0.6032, 0.6872, 0.6854, 0.1788],
          [-0.0000, 0.1699, 0.5525, 0.6548, 0.6814, 0.3219]]],
        decimal=4)


def test_sirt():
    tomo, theta = synthetic_tomo()
    assert_array_almost_equal(
        sirt(tomo, theta, num_iter=4),
        [[[0.4168, 0.7387, 0.6132, 0.5395, 0.2556, -0.3265],
          [0.2361, 0.7709, 0.6897, 0.5733, 0.5526, -0.1612],
          [0.0657, 0.7390, 0.6953, 0.7003, 0.6447, -0.1046],
          [-0.0695, 0.6693, 0.6939, 0.7013, 0.7602, 0.0581],
          [-0.1411, 0.5820, 0.6072, 0.7044, 0.7752, 0.2645],
          [-0.3514, 0.2355, 0.5748, 0.6512, 0.7456, 0.4288]],
         [[0.4168, 0.7387, 0.6132, 0.5395, 0.2556, -0.3265],
          [0.2361, 0.7709, 0.6897, 0.5733, 0.5526, -0.1612],
          [0.0657, 0.7390, 0.6953, 0.7003, 0.6447, -0.1046],
          [-0.0695, 0.6693, 0.6939, 0.7013, 0.7602, 0.0581],
          [-0.1411, 0.5820, 0.6072, 0.7044, 0.7752, 0.2645],
          [-0.3514, 0.2355, 0.5748, 0.6512, 0.7456, 0.4288]]],
        decimal=4)


def test_write_center():
    tomo, theta = synthetic_tomo()
    dpath = os.path.join('test', 'tmp')
    write_center(tomo, theta, dpath, center=[3, 5, 0.5])
    assert_equals(os.path.isfile(os.path.join(dpath, '3.00.tiff')), True)
    assert_equals(os.path.isfile(os.path.join(dpath, '3.50.tiff')), True)
    assert_equals(os.path.isfile(os.path.join(dpath, '4.00.tiff')), True)
    assert_equals(os.path.isfile(os.path.join(dpath, '4.50.tiff')), True)
    shutil.rmtree(dpath)


if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
