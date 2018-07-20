#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TomoPy example script to reconstruct a single data set.
"""

from __future__ import print_function

import os
import sys
import json
import argparse
import numpy as np
import collections

import h5py
import tomopy
import dxchange
import timemory
from tomopy.misc.benchmark import *


#------------------------------------------------------------------------------#
def get_dx_dims(fname, dataset):
    """
    Read array size of a specific group of Data Exchange file.

    Parameters
    ----------
    fname : str
        String defining the path of file or file name.
    dataset : str
        Path to the dataset inside hdf5 file where data is located.

    Returns
    -------
    ndarray
        Data set size.
    """

    grp = '/'.join(['exchange', dataset])

    with h5py.File(fname, "r") as f:
        try:
            data = f[grp]
        except KeyError:
            return None

        shape = data.shape

    return shape


#------------------------------------------------------------------------------#
def restricted_float(x):

    x = float(x)
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


#------------------------------------------------------------------------------#
def read_rot_centers(fname):

    try:
        with open(fname) as json_file:
            json_string = json_file.read()
            dictionary = json.loads(json_string)

        return collections.OrderedDict(sorted(dictionary.items()))

    except Exception as error:
        print("ERROR: the json file containing the rotation axis locations is missing")
        print("ERROR: run: python find_center.py to create one first")
        exit()


#------------------------------------------------------------------------------#
@timemory.util.auto_timer()
def reconstruct(h5fname, sino, rot_center, args, blocked_views=None):

    # Read APS 32-BM raw data.
    proj, flat, dark, theta = dxchange.read_aps_32id(h5fname, sino=sino)

    # Manage the missing angles:
    if blocked_views is not None:
        print("Blocked Views: ", blocked_views)
        proj = np.concatenate((proj[0:blocked_views[0],:,:], proj[blocked_views[1]+1:-1,:,:]), axis=0)
        theta = np.concatenate((theta[0:blocked_views[0]], theta[blocked_views[1]+1:-1]))

    # Flat-field correction of raw data.
    data = tomopy.normalize(proj, flat, dark, cutoff=1.4)

    # remove stripes
    data = tomopy.remove_stripe_fw(data,level=7,wname='sym16',sigma=1,pad=True)

    print("Raw data: ", h5fname)
    print("Center: ", rot_center)

    data = tomopy.minus_log(data)

    data = tomopy.remove_nan(data, val=0.0)
    data = tomopy.remove_neg(data, val=0.00)
    data[np.where(data == np.inf)] = 0.00

    algorithm = args.algorithm
    ncores = args.ncores
    nitr = args.num_iter

    # always add algorithm
    _kwargs = {"algorithm": algorithm}

    # assign number of cores
    _kwargs["ncore"] = ncores

    # don't assign "num_iter" if gridrec or fbp
    if not algorithm in ["fbp", "gridrec"]:
        _kwargs["num_iter"] = nitr

    # Reconstruct object.
    with timemory.util.auto_timer("[tomopy.recon(algorithm='{}')]".format(algorithm)):
        rec = tomopy.recon(prj, ang, **_kwargs)

    # Mask each reconstructed slice with a circle.
    rec = tomopy.circ_mask(rec, axis=0, ratio=0.95)

    return rec


#------------------------------------------------------------------------------#
@timemory.util.auto_timer()
def rec_full(h5fname, rot_center, args, blocked_views):

    data_size = get_dx_dims(h5fname, 'data')

    output_dir = os.path.join(args.output_dir, 'rec_full')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select sinogram range to reconstruct.
    sino_start = 0
    sino_end = data_size[1]

    chunks = 16         # number of sinogram chunks to reconstruct
                        # only one chunk at the time is reconstructed
                        # allowing for limited RAM machines to complete a full reconstruction

    nSino_per_chunk = (sino_end - sino_start)/chunks
    print("Reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" %
        ((sino_end - sino_start), sino_start, sino_end, chunks, nSino_per_chunk))

    imgs = []
    strt = 0
    for iChunk in range(0,chunks):
        print('\n  -- chunk # %i' % (iChunk+1))
        sino_chunk_start = sino_start + nSino_per_chunk*iChunk
        sino_chunk_end = sino_start + nSino_per_chunk*(iChunk+1)
        print('\n  --------> [%i, %i]' % (sino_chunk_start, sino_chunk_end))

        if sino_chunk_end > sino_end:
            break

        sino = (int(sino_chunk_start), int(sino_chunk_end))

        # Reconstruct.
        rec = reconstruct(h5fname, sino, rot_center, args, blocked_views)

        # Write data as stack of TIFs.
        fname = os.path.join(output_dir,'recon_{}_'.format(args.algorithm))
        print("Reconstructions: ", fname)

        imgs.extend(output_images(rec, fname, args.format, args.scale, args.ncol))
        #dxchange.write_tiff_stack(rec, fname=fname, start=strt)
        strt += sino[1] - sino[0]


#------------------------------------------------------------------------------#
@timemory.util.auto_timer()
def rec_slice(h5fname, nsino, rot_center, args, blocked_views):

    data_size = get_dx_dims(h5fname, 'data')
    ssino = int(data_size[1] * nsino)

    output_dir = os.path.join(args.output_dir, 'rec_slice')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select sinogram range to reconstruct
    sino = None
    imgs = []

    start = ssino
    end = start + 1
    sino = (start, end)

    # Reconstruct
    rec = reconstruct(h5fname, sino, rot_center, args, blocked_views)

    fname = os.path.join(output_dir,'recon_{}_'.format(args.algorithm))

    #dxchange.write_tiff_stack(rec, fname=fname)
    print("Rec: ", fname)
    print("Slice: ", start)
    imgs.extend(output_images(rec, fname, args.format, args.scale, args.ncol))
    return imgs

#------------------------------------------------------------------------------#
def main(arg):

    import multiprocessing as mp
    default_ncores = mp.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("fname",
        help="file name of a tmographic dataset: /data/sample.h5")
    parser.add_argument("--axis", nargs='?', type=str, default="0",
        help="rotation axis location: 1024.0 (default 1/2 image horizontal size)")
    parser.add_argument("--type", nargs='?', type=str, default="slice",
        help="reconstruction type: full (default slice)")
    parser.add_argument("--nsino", nargs='?', type=restricted_float, default=0.5,
        help="location of the sinogram used by slice reconstruction (0 top, 1 bottom): 0.5 (default 0.5)")
    parser.add_argument("-a", "--algorithm", help="Select the algorithm",
                        default="gridrec", choices=algorithms, type=str)
    parser.add_argument("-n", "--ncores", help="number of cores",
                        default=default_ncores, type=int)
    parser.add_argument("-f", "--format", help="output image format",
                        default="jpeg", type=str)
    parser.add_argument("-S", "--scale", help="scale image by a positive factor",
                        default=1, type=int)
    parser.add_argument("-c", "--ncol", help="Number of images per row",
                        default=1, type=int)
    parser.add_argument("-i", "--num-iter", help="Number of iterations",
                        default=1, type=int)
    parser.add_argument("-o", "--output-dir", help="Output directory",
                        default=None, type=str)

    args = parser.parse_args()

    if args.output_dir is None:
        fpath = os.path.basename(os.path.dirname(args.fname))
        args.output_dir = os.path.join(fpath + "_output", args.algorithm)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    manager = timemory.manager()

    # Set path to the micro-CT data to reconstruct.
    fname = args.fname

    rot_center = float(args.axis)

    # Set default rotation axis location
    if rot_center == 0:
        data_size = get_dx_dims(fname, 'data')
        rot_center =  data_size[2]/2

    nsino = float(args.nsino)

    blocked_views = None

    slice = False
    if args.type == "slice":
        slice = True

    imgs = []
    if os.path.isfile(fname):
        if slice:
            imgs = rec_slice(fname, nsino, rot_center, args, blocked_views)
        else:
            imgs = rec_full(fname, rot_center, args, blocked_views)

    else:
        print("File Name does not exist: ", fname)

    #
    # timing report to stdout
    print('{}'.format(manager))

    fpath = args.output_dir
    timemory.options.output_dir = fpath
    timemory.options.set_report("run_tomopy.out")
    timemory.options.set_serial("run_tomopy.json")
    manager.report()

    #------------------------------------------------------------------#
    # provide timing plots
    try:
        print("\nPlotting TiMemory results...\n")
        timemory.plotting.plot(files=[timemory.options.serial_filename],
                               echo_dart=True,
                               output_dir=timemory.options.output_dir)
    except Exception as e:
        print("Exception [timemory.plotting] - {}".format(e))

    #------------------------------------------------------------------#
    # provide results to dashboard
    try:
        print("\nEchoing dart tags...\n")
        for i in range(0, len(imgs)):
            print("imgs[{}] = {}".format(i, imgs[i]))
            img_type = args.format
            print("Image name: {}".format(img_type))
            img_name = os.path.basename(imgs[i]).replace(
                ".{}".format(args.format), "")
            print("Image name: {}".format(img_name))
            img_path = imgs[i]
            print("name: {}, type: {}, path: {}".format(img_name, img_type, img_path))
            timemory.plotting.echo_dart_tag(img_name, img_path, img_type)
    except Exception as e:
        print("Exception [echo_dart_tag] - {}".format(e))

    #------------------------------------------------------------------#
    # provide ASCII results
    try:
        print("\nWriting notes...\n")
        notes = manager.write_ctest_notes(directory=fpath)
        print('"{}" wrote CTest notes file : {}'.format(__file__, notes))
    except Exception as e:
        print("Exception [write_ctest_notes] - {}".format(e))

if __name__ == "__main__":
    main(sys.argv[1:])

