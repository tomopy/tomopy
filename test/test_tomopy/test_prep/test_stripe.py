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

import numpy as np
import unittest
from ..util import read_file
from numpy.testing import assert_allclose
import tomopy.prep.stripe as srm

__author__ = "Doga Gursoy, Nghia Vo"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class StripeRemovalTestCase(unittest.TestCase):

    def setUp(self):
        self.eps = 10 ** (-6)
        self.size = 64
        self.mat = np.random.rand(self.size, self.size)
        (self.b, self.e) = (30, 31)
        self.mat[:, self.b:self.e] = 0.0

    def test_remove_stripe_fw(self):
        assert_allclose(
            srm.remove_stripe_fw(read_file('proj.npy')),
            read_file('remove_stripe_fw.npy'), rtol=1e-2)

    def test_remove_stripe_ti(self):
        assert_allclose(
            srm.remove_stripe_ti(read_file('proj.npy')),
            read_file('remove_stripe_ti.npy'), rtol=1e-2)

    def test_remove_stripe_based_sorting(self):
        mat_corr = srm.remove_stripe_based_sorting(
            np.expand_dims(self.mat, 1), 3, dim=1)[:, 0, :]
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_filtering(self):
        mat_corr = srm.remove_stripe_based_filtering(
            np.expand_dims(self.mat, 1), 3, 3, dim=1)[:, 0, :]
        num = np.mean(mat_corr[:, self.b:self.e])
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_fitting(self):
        mat = np.random.rand(self.size, self.size)
        mat[:, self.b:self.e] = 1.0
        mat_corr = srm.remove_stripe_based_fitting(
            np.expand_dims(mat, 1), 1, (5, 20))[:, 0, :]
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 1.0)
        self.assertTrue(num > self.eps)

    def test_detect_stripe(self):
        lis = np.random.rand(self.size)
        lis_off = np.linspace(0, 1, len(lis))
        lis = lis + lis_off
        lis[self.b:self.e] = 6.0
        lis_bin = srm._detect_stripe(lis, 1.5)
        pos = np.where(lis_bin == 1.0)
        self.assertTrue(len(pos) > 0 and pos[0] == self.b)

    def test_remove_large_stripe(self):
        mat = np.random.rand(self.size, self.size)
        lis_off = np.linspace(0, 1, self.size)
        mat_off = np.tile(lis_off, (self.size, 1))
        mat[:, self.b:self.e] = 6.0
        mat_corr = srm.remove_large_stripe(
            np.expand_dims(mat, 1), 1.5, 5)[:, 0, :]
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_remove_dead_stripe(self):
        mat = np.random.rand(self.size, self.size)
        lis_off = np.linspace(0, 1, self.size)
        mat[:, self.b:self.e] = 6.0
        mat_corr = srm.remove_dead_stripe(
            np.expand_dims(mat, 1), 1.5, 5)[:, 0, :]
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_remove_all_stripe(self):
        mat = np.random.rand(self.size, self.size)
        lis_off = np.linspace(0, 1, self.size)
        mat_off = np.tile(lis_off, (self.size, 1))
        mat[:, self.b:self.e] = 6.0
        mat_corr = srm.remove_all_stripe(
            np.expand_dims(mat, 1), 1.5, 5, 3)[:, 0, :]
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_remove_stripe_based_interpolation(self):
        mat = np.random.rand(self.size, self.size)
        lis_off = np.linspace(0, 1, self.size)
        mat[:, self.b:self.e] = 6.0
        mat_corr = srm.remove_stripe_based_interpolation(
            np.expand_dims(mat, 1), 1.5, 5)[:, 0, :]
        num = np.abs(np.mean(mat_corr[:, self.b:self.e]) - 6.0)
        self.assertTrue(num > self.eps)

    def test_stripe_detection(self):
        assert_allclose(
            srm.stripes_detect3d(read_file('test_stripe_data.npy'),
                                 size=10,
                                 radius=1),
            read_file('stripes_detect3d.npy'), rtol=1e-6)

    def test_stripe_mask(self):
        assert_allclose(
            srm.stripes_mask3d(read_file('stripes_detect3d.npy'),
                              threshold=0.6,
                              min_stripe_length = 10,
                              min_stripe_depth  = 0,
                              min_stripe_width = 5,
                              sensitivity_perc=85.0),
            read_file('stripes_mask3d.npy'), rtol=1e-6)
        
