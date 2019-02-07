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

import unittest
from tomopy.misc.phantom import baboon, barbara, cameraman, checkerboard, \
    lena, peppers, shepp2d, shepp3d
from numpy.testing import assert_array_equal as assert_equals

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestDistributeJobs(unittest.TestCase):
    def test_baboon(self):
        assert_equals(baboon().dtype, 'float32')
        assert_equals(baboon().shape, (1, 512, 512))
        assert_equals(baboon(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(baboon(size=64).shape, (1, 64, 64))

    def test_barbara(self):
        assert_equals(barbara().dtype, 'float32')
        assert_equals(barbara().shape, (1, 512, 512))
        assert_equals(barbara(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(barbara(size=64).shape, (1, 64, 64))

    def test_cameraman(self):
        assert_equals(cameraman().dtype, 'float32')
        assert_equals(cameraman().shape, (1, 512, 512))
        assert_equals(cameraman(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(cameraman(size=64).shape, (1, 64, 64))

    def test_checkerboard(self):
        assert_equals(checkerboard().dtype, 'float32')
        assert_equals(checkerboard().shape, (1, 512, 512))
        assert_equals(checkerboard(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(checkerboard(size=64).shape, (1, 64, 64))

    def test_lena(self):
        assert_equals(lena().dtype, 'float32')
        assert_equals(lena().shape, (1, 512, 512))
        assert_equals(lena(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(lena(size=64).shape, (1, 64, 64))

    def test_peppers(self):
        assert_equals(peppers().dtype, 'float32')
        assert_equals(peppers().shape, (1, 512, 512))
        assert_equals(peppers(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(peppers(size=64).shape, (1, 64, 64))

    def test_shepp2d(self):
        assert_equals(shepp2d().dtype, 'float32')
        assert_equals(shepp2d().shape, (1, 512, 512))
        assert_equals(shepp2d(size=(128, 256)).shape, (1, 128, 256))
        assert_equals(shepp2d(size=64).shape, (1, 64, 64))

    def test_shepp3d(self):
        assert_equals(shepp3d(size=(6, 8, 10)).dtype, 'float32')
        assert_equals(shepp3d(size=(6, 8, 10)).shape, (6, 8, 10))
        assert_equals(shepp3d(size=(6, 8, 10)).min(), 0)
        assert_equals(shepp3d(size=6).shape, (6, 6, 6))
