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

from __future__ import absolute_import

__version__ = '1.2.1'

__all__ = ['algorithms', 'image_quality', 'exit_action',
           'output_image', 'print_size', 'convert_image',
           'normalize', 'trim_border', 'fill_border',
           'rescale_image', 'quantify_difference', 'output_images',
           'image_comparison']

import os
import pylab
import numpy as np
import scipy.ndimage as ndimage
import numpy.linalg as LA
import timemory

algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem', 'sirt',
              'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad',
              'tv', 'grad', 'tikh']
image_quality = {}


def exit_action(errcode):
    man = timemory.manager()
    timemory.report(ign_cutoff=True)
    fname = 'signal_error_{}.out'.format(errcode)
    f = open(fname, 'w')
    f.write('{}\n'.format(man))
    f.close()


@timemory.util.auto_timer()
def output_image(image, fname):
    """Save an image and check that it exists afterward."""
    pylab.imsave(fname, image, cmap='gray')

    if not os.path.exists(fname):
        print("  ##################### WARNING #####################")
        print("  --> No image file at @ '{}' (expected) ...".format(fname))


def print_size(rec, msg=""):
    print("{} Image size: {} x {} x {}".format(
        msg,
        rec[0].shape[0],
        rec[0].shape[1],
        rec.shape[0]))


@timemory.util.auto_timer()
def convert_image(fname, current_format, new_format):
    """Create a copy of an image in a new_format.

    Parameters
    ----------
    fname : string
        The current image filename sans extension.
    current_format : string
        The current image file extension.
    new_fromat : string
        The new image file extension.

    """
    _fext = new_format
    _success = True

    try:
        from PIL import Image
        _cur_img = "{}.{}".format(fname, current_format)
        img = Image.open(_cur_img)
        out = img.convert("RGB")
        out.save(fname, "jpeg", quality=95)
        # print("  --> Converted '{}' to {} format...".format(fname, new_format.upper()))

    except Exception as e:

        print("  --> ##### {}...".format(e))
        print("  --> ##### Exception occurred converting '{}' to {} format...".format(
            fname, new_format.upper()))

        _fext = current_format
        _success = False

    _fname = "{}.{}".format(fname, _fext)
    return [_fname, _success, _fext]


def normalize(rec):
    """Normalize rec to the range [-1, 1]."""
    rec_n = rec.copy()
    try:
        _min = np.amin(rec_n)  # shift so range min is zero
        rec_n -= _min
        _max = np.amax(rec_n)
        if _max > 0.0:  # prevent division by zero
            rec_n /= 0.5 * _max  # rescale to range [0, 2]
        rec_n -= 1  # shift to range [-1, 1]
    except Exception as e:
        print("  --> ##### {}...".format(e))
        rec_n = rec.copy()

    return rec_n


def trim_border(rec, nimages, drow, dcol):
    """Crop rec along three dimensions.

    Axes 1 and 2 are trimmed from both sides with half stripped
    from each end.

    Parameters
    ----------
    rec : np.ndarray
    nimages : int
        The new length of axis 0.
    drow, dcol : int
        The number of indices along axes 1 and 2 to remove.

    """
    rec_n = rec.copy()
    nrows = rec[0].shape[0] - drow
    ncols = rec[0].shape[1] - dcol

    try:
        rec_tmp = np.ndarray([nimages, nrows, ncols])
        for i in range(nimages):
            _xb = int(0.5*drow)
            _xe = _xb + nrows
            _yb = int(0.5*dcol)
            _ye = _yb + ncols
            rec_tmp[i, :, :] = rec_n[i, _xb:_xe, _yb:_ye]
        rec_n = rec_tmp
    except Exception as e:
        print("  --> ##### {}...".format(e))
        rec_n = rec.copy()

    return rec_n


def fill_border(rec, nimages, drow, dcol):
    """Pad rec along axes 1 and 2 by drow and dcol. Crop axes 0 to nimages.

    The padding along axes 1 and 2 is split evenly between before and after
    the array.
    """
    rec_n = rec.copy()
    nrows = rec[0].shape[0] + drow
    ncols = rec[0].shape[1] + dcol

    try:
        rec_tmp = np.ndarray([nimages, nrows, ncols])
        for i in range(nimages):
            _xb = int(0.5*drow)
            _xe = _xb + rec_n[i].shape[0]
            _yb = int(0.5*dcol)
            _ye = _yb + rec_n[i].shape[1]
            rec_tmp[i, _xb:_xe, _yb:_ye] = rec_n[i, :, :]
        rec_n = rec_tmp
    except Exception as e:
        print("  --> ##### {}...".format(e))
        rec_n = rec.copy()

    return rec_n


@timemory.util.auto_timer()
def rescale_image(rec, nimages, scale, transform=True):
    """Resize a stack of images by positive scale."""
    rec_n = normalize(rec.copy())
    resize_kwargs = {'anti_aliasing': False, 'mode': 'constant'}
    try:
        import skimage.transform
        if transform is True:
            _nrows = rec[0].shape[0] * scale
            _ncols = rec[0].shape[1] * scale
            rec_tmp = np.ndarray([nimages, _nrows, _ncols])
            for i in range(nimages):
                rec_tmp[i] = skimage.transform.resize(rec_n[i],
                                                      (rec_n[i].shape[0] * scale,
                                                       rec_n[i].shape[1] * scale),
                                                       **resize_kwargs)
            rec_n = rec_tmp

    except Exception as e:
        print("  --> ##### {}...".format(e))
        rec_n = rec.copy()

    return rec_n


def quantify_difference(label, img, rec):
    """Return the L1,L2 norms of the diff and and grad diff of the two images.
    """
    _img = normalize(img)
    _rec = normalize(rec)

    _nrow = _rec[0].shape[0]
    _ncol = _rec[0].shape[1]
    _nimg = _rec.shape[0]

    _img = _img.reshape([_nrow*_nimg, _ncol])
    _rec = _rec.reshape([_nrow*_nimg, _ncol])

    # pixel diff
    _sub = _img - _rec
    # x-gradient diff
    _sx = ndimage.sobel(_img, axis=0, mode='reflect') - \
        ndimage.sobel(_rec, axis=0, mode='reflect')
    # y-gradient diff
    _sy = ndimage.sobel(_img, axis=1, mode='reflect') - \
        ndimage.sobel(_rec, axis=1, mode='reflect')

    _l1_pix = LA.norm(_sub, ord=1)
    _l2_pix = LA.norm(_sub, ord=2)

    _l1_grad = LA.norm(_sx, ord=1) + LA.norm(_sy, ord=1)
    _l2_grad = LA.norm(_sx, ord=2) + LA.norm(_sy, ord=2)

    print("")
    print("[{}]: pixel comparison".format(label))
    print("    L1-norm: {}".format(_l1_pix))
    print("    L2-norm: {}".format(_l2_pix))
    print("[{}]: gradient comparison".format(label))
    print("    L1-norm: {}".format(_l1_grad))
    print("    L2-norm: {}".format(_l2_grad))
    print("")

    return [[_l1_pix, _l2_pix], [_l1_grad, _l2_grad]]


@timemory.util.auto_timer()
def output_images(rec, fpath, format="jpeg", scale=1, ncol=1):
    """Save an image stack as a series of concatenated images.

    Each set of ncol images are concatenated horizontally and saved together
    into files named {fpath}_0_{ncol}.{img_format},
    {fpath}_{ncol}_{2*ncol}.{img_format}, {fpath}_{ncol}_{3*ncol}.{img_format},
    ...
    """
    imgs = []
    nitr = 0
    nimages = rec.shape[0]
    rec_i = None
    fname = "{}".format(fpath)

    if nimages < ncol:
        ncol = nimages

    rec_n = rec.copy()
    if scale > 1:
        rescale_image(rec, nimages, scale)

    print("Image size: {} x {} x {}".format(
        rec[0].shape[0],
        rec[0].shape[1],
        rec.shape[0]))

    print("Scaled Image size: {} x {} x {}".format(
        rec_n[0].shape[0],
        rec_n[0].shape[1],
        rec_n.shape[0]))

    for i in range(nimages):
        nitr += 1

        _f = "{}{}".format(fpath, i)
        _fimg = "{}.{}".format(_f, format)

        if rec_i is None:
            rec_i = rec_n[i]
        else:
            rec_i = np.concatenate((rec_i, rec_n[i]), axis=1)

        if nitr % ncol == 0 or i+1 == nimages:
            fname = "{}{}.{}".format(fname, i, format)
            output_image(rec_i, fname)
            imgs.append(fname)
            rec_i = None
            fname = "{}".format(fpath)
        else:
            fname = "{}{}_".format(fname, i)

    return imgs


class image_comparison(object):
    """
    A class for combining image slices into a column comparison
    """

    def __init__(self, ncompare, nslice, nrows, ncols, solution=None, dtype=float):
        self.input_dims = [nslice, nrows, ncols]
        self.store_dims = [nslice, nrows, ncols * (ncompare + 1)]
        self.tags = ["soln"]
        if solution is None:
            self.solution = np.zeros(self.input_dims, dtype=dtype)
        else:
            self.solution = normalize(solution)
        # array of results
        self.array = np.ndarray(self.store_dims, dtype=dtype)
        self.array[:, :, 0:ncols] = self.solution[:, :, :]
        # difference
        self.delta = np.ndarray(self.store_dims, dtype=dtype)
        self.delta[:, :, 0:ncols] = self.solution[:, :, :]

    def assign(self, label, block, array):
        self.tags.append(label)
        array = normalize(array)
        _b = (self.input_dims[2] * block)
        _e = (self.input_dims[2] * (block+1))
        try:
            self.array[:, :, _b:_e] = array[:, :, :]
        except Exception as e:
            print("storage: {}".format(self.store_dims))
            print("label: {}, block: {}".format(label, block))
            print("target = [{}, {}, {}:{}]".format(
                self.input_dims[0],
                self.input_dims[1],
                _b, _e))
            print(e)
            print("array: {}".format(array))
            raise
        delta = array - self.solution
        self.delta[:, :, _b:_e] = delta

    def tagname(self):
        return "-".join(self.tags)
