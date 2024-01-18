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

import unittest

import numpy as np
import scipy
from numpy.testing import assert_allclose, assert_equal
from tomopy.misc.corr import (
    gaussian_filter,
    median_filter,
    median_filter3d,
    remove_outlier3d,
    median_filter_nonfinite,
    remove_neg,
    remove_nan,
    remove_outlier,
    circ_mask,
    inpainter_morph,
)
from tomopy.misc.phantom import peppers

from ..util import read_file, loop_dim

__author__ = "Doga Gursoy, William Judge"
__copyright__ = "Copyright (c) 2015, UChicago Argonne, LLC."
__docformat__ = "restructuredtext en"


class ImageFilterTestCase(unittest.TestCase):
    def test_gaussian_filter(self):
        loop_dim(gaussian_filter, read_file("cube.npy"))

    def test_median_filter(self):
        loop_dim(median_filter, read_file("cube.npy"))

    def test_median_filter_nonfinite(self):
        # Set a standard random value to make reproducible
        np.random.seed(1)

        data_org = np.ones(shape=(100, 100, 100))

        # Add some random non-finite values to an array of all ones
        for non_finite in [np.nan, np.inf, -np.inf]:
            for i in range(50):
                x = np.random.randint(0, 100)
                y = np.random.randint(0, 100)
                z = np.random.randint(0, 100)
                data_org[z, x, y] = non_finite

        data_post_corr = median_filter_nonfinite(
            data_org.copy(),
            size=5,
            callback=None,
        )

        # All the post filtering values should be 1 because all of the finite
        # values are 1.
        assert np.all(data_post_corr == 1.0)

        # Making sure filter raises ValueError when function finds a filter
        # filled with non-finite values.
        for non_finite in [np.nan, np.inf, -np.inf]:
            data_org = np.empty((1, 3, 3))
            data_org[:, -2:, -2:] = non_finite
            with self.assertRaises(ValueError):
                result = median_filter_nonfinite(
                    data_org.copy(),
                    size=3,
                    callback=None,
                )

    def test_median_filter3d(self):
        A = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        assert_equal(
            scipy.ndimage.median_filter(np.float32(A), size=3),
            median_filter3d(np.float32(A), size=3),
        )

    def test_remove_outlier3d(self):
        A = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        A[2, 2, 2] = 1000.0  # introduce an outlier
        A_dezinged = remove_outlier3d(np.float32(A), dif=500, size=3)
        A[2, 2, 2] = 75  # substituted value by dezinger
        assert_equal(np.float32(A), A_dezinged)

    def test_remove_neg(self):
        assert_allclose(remove_neg([-2, -1, 0, 1, 2]), [0, 0, 0, 1, 2])

    def test_remove_nan(self):
        assert_allclose(remove_nan([np.nan, 1.5, 2, np.nan, 1]), [0, 1.5, 2, 0, 1])

    def test_remove_outlier(self):
        proj = read_file("proj.npy")
        proj[8][4][6] = 20
        assert_allclose(remove_outlier(proj, dif=10), read_file("proj.npy"))

    def test_circ_mask(self):
        loop_dim(circ_mask, read_file("obj.npy"))

    def test_inpainter2d(self):
        input_image = peppers(size=512)[0, :, :]
        mask = np.zeros((512, 512))
        mask[270:285, :] = 1  # crop out the horizontal region
        mask = np.array(mask, dtype="bool")

        inpainted2d_mean = inpainter_morph(
            input_image, mask, size=3, iterations=2, inpainting_type="mean", axis=None
        )
        inpainted2d_median = inpainter_morph(
            input_image, mask, size=3, iterations=2, inpainting_type="median", axis=None
        )
        inpainted2d_random = inpainter_morph(
            input_image, mask, size=3, iterations=2, inpainting_type="random", axis=None
        )
        assert_allclose(
            np.mean(inpainted2d_mean, axis=(0, 1)).sum(), 104.20617, rtol=1e-6
        )
        assert_allclose(
            np.mean(inpainted2d_median, axis=(0, 1)).sum(), 104.26762, rtol=1e-6
        )
        # providing a range as the method is probabilistic
        assert 103.0 <= np.mean(inpainted2d_random, axis=(0, 1)).sum() <= 106.0

    def test_inpainter3d_as_2d(self):
        input_vol3d = np.float32(np.zeros((512, 3, 512)))
        mask3d = np.zeros((512, 3, 512))
        mask2d = np.zeros((512, 512))
        mask2d[270:285, :] = 1  # crop out the horizontal region
        for j in range(3):
            input_vol3d[:, j, :] = peppers(size=512)[0, :, :]
            mask3d[:, j, :] = mask2d

        mask3d = np.array(mask3d, dtype="bool")

        inpainted3d_mean = inpainter_morph(
            input_vol3d, mask3d, size=3, iterations=2, inpainting_type="mean", axis=1
        )
        inpainted3d_median = inpainter_morph(
            input_vol3d, mask3d, size=3, iterations=2, inpainting_type="median", axis=1
        )
        inpainted3d_random = inpainter_morph(
            input_vol3d, mask3d, size=3, iterations=2, inpainting_type="random", axis=1
        )
        assert_allclose(np.mean(inpainted3d_mean), 104.20617, rtol=1e-6)
        assert_allclose(np.mean(inpainted3d_median), 104.26761, rtol=1e-6)
        # providing a range as the method is probabilistic
        assert 103.0 <= np.mean(inpainted3d_random) <= 106.0

    def test_inpainter3d(self):
        input_vol3d = np.float32(np.zeros((512, 3, 512)))
        mask3d = np.zeros((512, 3, 512))
        mask2d = np.zeros((512, 512))
        mask2d[270:285, :] = 1  # crop out the horizontal region
        for j in range(3):
            input_vol3d[:, j, :] = peppers(size=512)[0, :, :]
            mask3d[:, j, :] = mask2d

        mask3d = np.array(mask3d, dtype="bool")

        inpainted3d_mean = inpainter_morph(
            input_vol3d, mask3d, size=3, iterations=0, inpainting_type="mean"
        )
        inpainted3d_random = inpainter_morph(
            input_vol3d, mask3d, size=1, iterations=0, inpainting_type="random"
        )
        assert_allclose(np.mean(inpainted3d_mean), 104.20623, rtol=1e-6)
        # pproviding a range as the method is probabilistic
        assert 103.0 <= np.mean(inpainted3d_random) <= 106.0
