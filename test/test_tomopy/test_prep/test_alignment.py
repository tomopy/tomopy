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
 
import os.path
import unittest
from numpy.testing import assert_allclose

from ..util import read_file
from tomopy.prep.alignment import distortion_correction_proj\
, distortion_correction_sino, load_distortion_coefs
 
__author__ = "Nghia Vo"
__copyright__ = "Copyright (c) 2019, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
 
 
class DistortionCorrectionTestCase(unittest.TestCase):
    def test_distortion_correction_proj(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))       
        file_path = os.path.join(
            os.path.dirname(test_dir),'test_data','discoef.txt')
        (xc, yc, list_fact) = load_distortion_coefs(file_path)        
        assert_allclose(
            distortion_correction_proj(
                read_file('distortion_3d.npy'), xc, yc, list_fact)[0],
            read_file('distortion_proj.npy'), rtol=1e-2)
 
    def test_distortion_correction_sino(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))        
        file_path = os.path.join(
            os.path.dirname(test_dir),'test_data','discoef.txt')
        (xc, yc, list_fact) = load_distortion_coefs(file_path)
        assert_allclose(
            distortion_correction_sino(
                read_file('distortion_3d.npy'), 5, xc, yc, list_fact),
            read_file('distortion_sino.npy'), rtol=1e-2)
