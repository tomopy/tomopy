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

"""
Module for reconstruction algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tomopy.util.extern as extern
import tomopy.util.dtype as dtype
from tomopy.sim.project import get_center
from tomopy.recon.algorithm import init_tomo
import logging

logger = logging.getLogger(__name__)


__author__ = "Doga Gursoy"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['vector', 'vector2', 'vector3']


def vector(tomo, theta, center=None, num_iter=1):
    tomo = dtype.as_float32(tomo)
    theta = dtype.as_float32(theta)

    # Initialize tomography data.
    tomo = init_tomo(tomo, sinogram_order=False, sharedmem=False)

    recon_shape = (tomo.shape[0], tomo.shape[2], tomo.shape[2])
    recon1 = np.zeros(recon_shape, dtype=np.float32)
    recon2 = np.zeros(recon_shape, dtype=np.float32)

    center_arr = get_center(tomo.shape, center)

    extern.c_vector(tomo, center_arr, recon1, recon2, theta, 
        num_gridx=tomo.shape[2], num_gridy=tomo.shape[2], num_iter=num_iter)
    return recon1, recon2


def vector2(tomo1, tomo2, theta1, theta2, center1=None, center2=None, num_iter=1, axis1=1, axis2=2):
    tomo1 = dtype.as_float32(tomo1)
    tomo2 = dtype.as_float32(tomo2)
    theta1 = dtype.as_float32(theta1)
    theta2 = dtype.as_float32(theta2)

    # Initialize tomography data.
    tomo1 = init_tomo(tomo1, sinogram_order=False, sharedmem=False)
    tomo2 = init_tomo(tomo2, sinogram_order=False, sharedmem=False)

    recon_shape = (tomo1.shape[0], tomo1.shape[2], tomo1.shape[2])
    recon1 = np.zeros(recon_shape, dtype=np.float32)
    recon2 = np.zeros(recon_shape, dtype=np.float32)
    recon3 = np.zeros(recon_shape, dtype=np.float32)

    center_arr1 = get_center(tomo1.shape, center1)
    center_arr2 = get_center(tomo2.shape, center2)

    extern.c_vector2(tomo1, tomo2, center_arr1, center_arr2, recon1, recon2, recon3, theta1, theta2, 
        num_gridx=tomo1.shape[2], num_gridy=tomo1.shape[2], num_iter=num_iter, axis1=axis1, axis2=axis2)
    return recon1, recon2, recon3


def vector3(tomo1, tomo2, tomo3, theta1, theta2, theta3, center1=None, center2=None, center3=None, num_iter=1, axis1=0, axis2=1, axis3=2):
    tomo1 = dtype.as_float32(tomo1)
    tomo2 = dtype.as_float32(tomo2)
    tomo3 = dtype.as_float32(tomo3)
    theta1 = dtype.as_float32(theta1)
    theta2 = dtype.as_float32(theta2)
    theta3 = dtype.as_float32(theta3)

    # Initialize tomography data.
    tomo1 = init_tomo(tomo1, sinogram_order=False, sharedmem=False)
    tomo2 = init_tomo(tomo2, sinogram_order=False, sharedmem=False)
    tomo3 = init_tomo(tomo3, sinogram_order=False, sharedmem=False)

    recon_shape = (tomo1.shape[0], tomo1.shape[2], tomo1.shape[2])
    recon1 = np.zeros(recon_shape, dtype=np.float32)
    recon2 = np.zeros(recon_shape, dtype=np.float32)
    recon3 = np.zeros(recon_shape, dtype=np.float32)

    center_arr1 = get_center(tomo1.shape, center1)
    center_arr2 = get_center(tomo2.shape, center2)
    center_arr3 = get_center(tomo3.shape, center3)

    extern.c_vector3(tomo1, tomo2, tomo3, center_arr1, center_arr2, center_arr3, recon1, recon2, recon3, theta1, theta2, theta3,  
        num_gridx=tomo1.shape[2], num_gridy=tomo1.shape[2], num_iter=num_iter, axis1=axis1, axis2=axis2, axis3=axis3)
    return recon1, recon2, recon3