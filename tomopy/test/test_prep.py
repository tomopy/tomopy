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


def test_normalize():
    data = np.array(
        [[[29, 85, 39, 45], 
          [24, 53, 12, 89]],
         [[25, 74, 63, 98], 
          [63, 27, 43, 68]],
         [[13, 65, 33, 12], 
          [42, 33, 87, 16]]], dtype='float32')
    white = np.array(
        [[[52, 53, 51, 56], 
          [24, 53, 12, 89]],
         [[51, 52, 49, 50], 
          [42, 33, 87, 16]]], dtype='float32')
    dark = np.array(
        [[[1, 3, 0, 4], 
          [1, 0, 0, 2]]], dtype='float32')
    ndata = np.array(
        [[[0.55445546, 1.65656567, 0.77999997, 0.83673471],
          [0.71875000, 1.23255813, 0.24242425, 1.72277224]],
         [[0.47524753, 1.43434346, 1.25999999, 1.91836739],
          [1.93750000, 0.62790698, 0.86868685, 1.30693066]],
         [[0.23762377, 1.25252521, 0.66000003, 0.16326530],
          [1.28125000, 0.76744187, 1.75757575, 0.27722773]]]
          , dtype='float32')
    assert_array_almost_equal(normalize(data, white, dark), ndata)

if __name__ == '__main__':
    import nose
    nose.runmodule(exit=False)
