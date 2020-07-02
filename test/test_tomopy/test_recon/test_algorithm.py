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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import unittest
from ..util import read_file
from tomopy.recon.algorithm import recon
from numpy.testing import assert_allclose
import numpy as np

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class ReconstructionAlgorithmTestCase(unittest.TestCase):
    def setUp(self):
        self.prj = read_file('proj.npy')
        self.ang = read_file('angle.npy').astype('float32')

    def test_art(self):
        os.environ["TOMOPY_USE_C_ART"] = "1"
        assert_allclose(
            recon(self.prj, self.ang, algorithm='art', num_iter=4),
            read_file('art.npy'), rtol=1e-2)

    def test_bart(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='bart', num_iter=4),
            read_file('bart.npy'), rtol=1e-2)

    def test_fbp(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='fbp'),
            read_file('fbp.npy'), rtol=1e-2)

    def test_gridrec_custom(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='none'),
            recon(
                self.prj, self.ang, algorithm='gridrec', filter_name='custom',
                filter_par=np.ones(self.prj.shape[-1], dtype=np.float32)))

    def test_gridrec(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='none'),
            read_file('gridrec_none.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='shepp'),
            read_file('gridrec_shepp.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='cosine'),
            read_file('gridrec_cosine.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='hann'),
            read_file('gridrec_hann.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='hamming'),
            read_file('gridrec_hamming.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='ramlak'),
            read_file('gridrec_ramlak.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec', filter_name='parzen'),
            read_file('gridrec_parzen.npy'), rtol=1e-2)
        assert_allclose(
            recon(self.prj, self.ang, algorithm='gridrec',
                  filter_name='butterworth'),
            read_file('gridrec_butterworth.npy'), rtol=1e-2)

    def test_mlem(self):
        result = recon(self.prj, self.ang, algorithm='mlem', num_iter=4)
        assert_allclose(result, read_file('mlem.npy'), rtol=1e-2)

    def test_mlem_accel(self):
        result = recon(self.prj, self.ang, algorithm='mlem', num_iter=4,
                       accelerated=True, device='cpu')
        assert_allclose(result, read_file('mlem_accel.npy'), rtol=1e-2)

    @unittest.skipUnless("CUDA_VERSION" in os.environ, "CUDA_VERSION not set.")
    def test_mlem_gpu(self):
        result = recon(self.prj, self.ang, algorithm='mlem', num_iter=4,
                       accelerated=True, device='gpu')
        assert_allclose(result, read_file('mlem_accel_gpu.npy'), rtol=1e-2)

    def test_osem(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='osem', num_iter=4),
            read_file('osem.npy'), rtol=1e-2)

    def test_ospml_hybrid(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='ospml_hybrid', num_iter=4),
            read_file('ospml_hybrid.npy'), rtol=1e-2)

    def test_ospml_quad(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='ospml_quad', num_iter=4),
            read_file('ospml_quad.npy'), rtol=1e-2)

    def test_pml_hybrid(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='pml_hybrid', num_iter=4),
            read_file('pml_hybrid.npy'), rtol=1e-2)

    def test_pml_quad(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='pml_quad', num_iter=4),
            read_file('pml_quad.npy'), rtol=1e-2)

    def test_sirt(self):
        result = recon(self.prj, self.ang, algorithm='sirt', num_iter=4)
        assert_allclose(result, read_file('sirt.npy'), rtol=1e-2)

    def test_sirt_accel(self):
        result = recon(self.prj, self.ang, algorithm='sirt',
                       num_iter=4, accelerated=True, device='cpu')
        assert_allclose(result, read_file('sirt_accel.npy'), rtol=1e-2)

    @unittest.skipUnless("CUDA_VERSION" in os.environ, "CUDA_VERSION not set.")
    def test_sirt_gpu(self):
        result = recon(self.prj, self.ang, algorithm='sirt',
                       num_iter=4, accelerated=True, device='gpu')
        assert_allclose(result, read_file('sirt_accel_gpu.npy'), rtol=1e-2)

    def test_tv(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='tv', num_iter=4),
            read_file('tv.npy'), rtol=1e-2)

    def test_grad(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='grad', num_iter=4),
            read_file('grad.npy'), rtol=1e-2)

    def test_tikh(self):
        assert_allclose(
            recon(self.prj, self.ang, algorithm='tikh', num_iter=4),
            read_file('tikh.npy'), rtol=1e-2)
