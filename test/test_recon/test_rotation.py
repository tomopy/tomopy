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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import unittest
from ..util import read_file
from tomopy.recon.rotation import write_center, find_center, find_center_vo, find_center_pc
import numpy as np
from scipy.ndimage.interpolation import shift as image_shift
import os.path
import shutil
from nose.tools import assert_equals
from numpy.testing import assert_allclose

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class CenterFindingTestCase(unittest.TestCase):
    def test_write_center(self):
        dpath = os.path.join('test', 'tmp')
        cen_range = (5, 7, 0.5)
        cen = np.arange(*cen_range)
        write_center(
            read_file('proj.npy'),
            read_file('angle.npy'),
            dpath, cen_range=cen_range)
        for m in range(cen.size):
            assert_equals(
                os.path.isfile(
                    os.path.join(
                        os.path.join('test', 'tmp'),
                        str('{0:.2f}'.format(cen[m]) + '.tiff'))), True)
        shutil.rmtree(dpath)

    def test_find_center(self):
        sim = read_file('sinogram.npy')
        ang = np.linspace(0, np.pi, sim.shape[0])
        cen = find_center(sim, ang)
        assert_allclose(cen, 45.28, rtol=1e-2)

    def test_find_center_vo(self):
        sim = read_file('sinogram.npy')
        cen = find_center_vo(sim)
        assert_allclose(cen, 45.28, rtol=1e-2)

    def test_find_center_vo_with_downsampling(self):
        sim = read_file('sinogram.npy')
        np.pad(
            sim, ((1000, 1000), (0, 0), (1000, 1000)),
            mode="constant", constant_values=0)
        cen = find_center_vo(sim)
        assert_allclose(cen, 45.28, rtol=1e-2)

    def test_find_center_pc(self):
        proj_0 = read_file('projection.npy')
        proj_180 = image_shift(np.fliplr(proj_0), (0, 18.75), mode='reflect')
        cen = find_center_pc(proj_0, proj_180)
        assert_allclose(cen, 73.375, rtol=0.25)
