#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2019, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2019. UChicago Argonne, LLC. This software was produced       #
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

"""Utilities for TomoPy + PyCTest."""

import os.path
import timemory
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy.linalg as LA


def exit_action(errcode):
    man = timemory.manager()
    timemory.report(ign_cutoff=True)
    fname = 'signal_error_{}.out'.format(errcode)
    f = open(fname, 'w')
    f.write('{}\n'.format(man))
    f.close()


algorithm_choices = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem', 'sirt',
                     'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad',
                     'tv', 'grad']

phantom_choices = ["baboon", "cameraman", "barbara", "checkerboard",
                   "lena", "peppers", "shepp2d", "shepp3d"]

image_quality = {}


@timemory.util.auto_timer()
def save_image(image, fname):
    """Save an image and check that it exists afterward."""
    plt.imsave(fname, image, cmap='gray')
    if os.path.exists(fname):
        print("  --> Image file found @ '{}'...".format(fname))
    else:
        print("  ##################### WARNING #####################")
        print("  --> No image file at @ '{}' (expected) ...".format(fname))


# FIXME: convert_image is unused
@timemory.util.auto_timer()
def convert_image(fname, current_format, new_format="jpeg"):
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
        out = img.convert(mode="RGB")
        out.save(fname, format=new_format, quality=95)
        print("  --> Converted '{}' to {} format...".format(
              fname, new_format.upper()))
    except Exception as e:
        print("  --> ##### {}...".format(e))
        print("  --> ##### Exception occurred converting "
              "'{}' to {} format...".format(fname, new_format.upper()))
        _fext = current_format
        _success = False
        _fname = "{}.{}".format(fname, _fext)
        return [_fname, _success, _fext]


def normalize(rec):
    """Normalize rec to the range [-1, 1]."""
    rec_n = np.asarray(rec).copy()
    try:
        _min = np.amin(rec_n)
        rec_n -= _min  # shift sp range min is zero
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
    rec_n = np.asarray(rec).copy()
    # compute new dimensions
    nrows = rec.shape[1] - drow
    ncols = rec.shape[2] - dcol
    # compute starting and ending coords of trimmed array
    _xb = drow // 2
    _xe = _xb + nrows
    _yb = dcol // 2
    _ye = _yb + ncols
    try:
        return rec_n[:nimages, _xb:_xe, _yb:_ye]
    except Exception as e:
        print("  --> ##### {}...".format(e))
        return rec_n


#  FIXME: fill_border is unused
def fill_border(rec, nimages, drow, dcol):
    """Pad rec along axes 1 and 2 by drow and dcol. Crop axes 0 to nimages.

    The padding along axes 1 and 2 is split evenly between before and after
    the array.
    """
    rec_n = np.asarray(rec).copy()
    # compute new dimensions
    nrows = rec.shape[1] + drow
    ncols = rec.shape[2] + dcol
    # compute starting and ending coords of padded array
    _xb = drow // 2
    _xe = _xb + rec_n[i].shape[0]
    _yb = dcol // 2
    _ye = _yb + rec_n[i].shape[1]
    try:
        rec_tmp = np.empty([nimages, nrows, ncols])
        rec_tmp[:, _xb:_xe, _yb:_ye] = rec_n[0:nimages, :, :]
        return rec_tmp
    except Exception as e:
        print("  --> ##### {}...".format(e))
        return rec_n

#  FIXME: is this decorator necessary when there is no `with timemory...`
# statement?
@timemory.util.auto_timer()
def resize_image(stack, scale=1):
    """Resize a stack of images by positive scale."""
    if scale > 1:
        try:
            import skimage.transform
            return skimage.transform.resize(
                image=stack,
                output_shape=(
                    stack.shape[0],
                    stack.shape[1] * scale,
                    stack.shape[2] * scale,
                ),
            )
        except Exception as e:
            print("  --> ##### {}...".format(e))
    return stack.copy()


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
    _sx = ndimage.sobel(_img, axis=0, mode='constant') - \
        ndimage.sobel(_rec, axis=0, mode='constant')
    # y-gradient diff
    _sy = ndimage.sobel(_img, axis=1, mode='constant') - \
        ndimage.sobel(_rec, axis=1, mode='constant')

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
def output_images(rec, fpath, img_format="jpeg", scale=1, ncol=1):
    """Save an image stack as a series of concatenated images.

    Each set of ncol images are concatenated horizontally and saved together
    into files named {fpath}_0_{ncol}.{img_format},
    {fpath}_{ncol}_{2*ncol}.{img_format}, {fpath}_{ncol}_{3*ncol}.{img_format},
    ...
    FIXME: What is the naming scheme supposed to be when there is more than one
    row?

    """
    rec_n = resize_image(rec, scale)
    print("Image size: {} x {} x {}".format(
        rec.shape[1],
        rec.shape[2],
        rec.shape[0],
    ))
    print("Resized image size: {} x {} x {}".format(
        rec_n.shape[1],
        rec_n.shape[2],
        rec_n.shape[0],
    ))

    filenames = list()
    for lo in range(0, len(rec_n), ncol):
        hi = min(lo + ncol, len(rec_n))
        rec_row = np.concatenate(rec_n[lo:hi], axis=1)
        fname = "{}{}_{}.{}".format(fpath, lo, hi, img_format)
        save_image(rec_row, fname)
        filenames.append(fname)
    return filenames


class ImageComparison(object):
    """A class for combining image slices into a column comparison.

    FIXME: Document how this class is supposed to work.
    """

    def __init__(self, ncompare, nslice, nrows, ncols, solution=None,
                 dtype=float):
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
