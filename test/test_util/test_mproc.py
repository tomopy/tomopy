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

import tomopy.util.mproc as mproc
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import unittest

__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


def _synthetic_func(a, val):
    for m in range(a.shape[0]):
        a[m, :, :] = val


def _test_shape(a, expected_shape):
    assert a.shape == expected_shape
    return a


class TestDistributeJobs(unittest.TestCase):

    def _test_shape(self, a, expected_shape, axis=0, ncore=None, nchunk=None):
        ret = mproc.distribute_jobs(
            a,
            func=_test_shape,
            args=(expected_shape,),
            axis=axis,
            ncore=ncore,
            nchunk=nchunk)
        assert_array_equal(a, ret)

    def test_shape(self):
        a = np.zeros((2, 4, 8))
        # whole array on single core
        self._test_shape(a, a.shape, axis=0, ncore=1, nchunk=None)
        # chunk=0 for diff axis
        self._test_shape(a, (4, 8), axis=0, ncore=1, nchunk=0)
        self._test_shape(a, (4, 8), axis=0, ncore=None, nchunk=0)
        self._test_shape(a, (2, 8), axis=1, ncore=1, nchunk=0)
        self._test_shape(a, (2, 8), axis=1, ncore=None, nchunk=0)
        self._test_shape(a, (2, 4), axis=2, ncore=1, nchunk=0)
        self._test_shape(a, (2, 4), axis=2, ncore=None, nchunk=0)
        # two core tests
        self._test_shape(a, (1, 4, 8), axis=0, ncore=2, nchunk=None)
        self._test_shape(a, (2, 2, 8), axis=1, ncore=2, nchunk=None)
        self._test_shape(a, (2, 4, 4), axis=2, ncore=2, nchunk=None)
        # change nchunk size
        self._test_shape(a, (1, 4, 8), axis=0, ncore=None, nchunk=1)
        self._test_shape(a, (2, 2, 8), axis=1, ncore=None, nchunk=2)
        self._test_shape(a, (2, 4, 2), axis=2, ncore=None, nchunk=2)

    def test_distribute_jobs(self):
        assert_allclose(
            mproc.distribute_jobs(
                np.zeros((8, 8, 8)),
                func=_synthetic_func,
                args=(1.,),
                axis=0),
            np.ones((8, 8, 8)))
