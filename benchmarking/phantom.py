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

"""TomoPy script to reconstruct a built-in phantom."""

from __future__ import absolute_import

import sys
import shutil
import os
import argparse
import traceback
import multiprocessing as mp

import tomopy
import timemory
import timemory.options as options

from benchmarking.utils import *


def get_basepath(args, algorithm, phantom):
    """Return the folder where data for a given reconstruction goes."""
    return os.path.join(os.getcwd(), args.output_dir, phantom, algorithm)


@timemory.util.auto_timer()
def generate(phantom="shepp3d", nsize=512, nangles=360):
    """Return the simulated data for the given phantom."""
    with timemory.util.auto_timer("[tomopy.misc.phantom.{}]".format(phantom)):
        obj = getattr(tomopy.misc.phantom, phantom)(size=nsize)
    with timemory.util.auto_timer("[tomopy.angles]"):
        ang = tomopy.angles(nangles)
    with timemory.util.auto_timer("[tomopy.project]"):
        prj = tomopy.project(obj, ang)
    return [prj, ang, obj]


@timemory.util.auto_timer()
def run(phantom, algorithm, args, get_recon=False):
    """Run reconstruction benchmarks for phantoms.

    Parameters
    ----------
        phantom : string
            The name of the phantom to use.
        algorithm : string
            The name of the algorithm to test.
        args : argparser args

    Returns
    -------
    Either rec or imgs
    rec : np.ndarray
        The reconstructed image.
    imgs : list
        A list of the original, reconstructed, and difference image
    """
    prj, ang, obj = generate(phantom, args.size, args.angles)
    # always add algorithm and ncores
    _kwargs = {
        "algorithm": algorithm,
        "ncore": args.ncores,
    }
    # don't assign "num_iter" if gridrec or fbp
    if algorithm not in ["fbp", "gridrec"]:
        _kwargs["num_iter"] = args.num_iter
    print("kwargs: {}".format(_kwargs))

    with timemory.util.auto_timer("[tomopy.recon(algorithm='{}')]".format(
                                  algorithm)):
        rec = tomopy.recon(prj, ang, **_kwargs)
    # trim rec because the data was padded to keep the entire object in the
    # field of view
    rec = trim_border(rec, rec.shape[0],
                      rec.shape[1] - obj.shape[1],
                      rec.shape[2] - obj.shape[2])

    label = "{} @ {}".format(algorithm.upper(), phantom.upper())

    quantify_difference(label, obj, rec)

    global image_quality

    if "orig" not in image_quality:
        image_quality["orig"] = obj

    dif = obj - rec
    image_quality[algorithm] = dif

    if get_recon is True:
        return rec

    imgs = []
    bname = get_basepath(args, algorithm, phantom)
    oname = os.path.join(bname, "orig_{}_".format(algorithm))
    fname = os.path.join(bname, "stack_{}_".format(algorithm))
    dname = os.path.join(bname, "diff_{}_".format(algorithm))

    print("oname = {}, fname = {}, dname = {}".format(oname, fname, dname))
    imgs.extend(output_images(obj, oname, args.format, args.scale, args.ncol))
    imgs.extend(output_images(rec, fname, args.format, args.scale, args.ncol))
    imgs.extend(output_images(dif, dname, args.format, args.scale, args.ncol))

    return imgs


def main(args):
    """
    FIXME: Document the expected filestructure, types of files that are
    created, and what information they contain.

    """

    global image_quality

    manager = timemory.manager()

    print(("\nArguments:\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n"
          "{} = {}\n{} = {}\n{} = {}\n{} = {}\n").format(
          "\tPhantom", args.phantom,
          "\tAlgorithm", args.algorithm,
          "\tSize", args.size,
          "\tAngles", args.angles,
          "\tFormat", args.format,
          "\tScale", args.scale,
          "\tnumber of cores", args.ncores,
          "\tnumber of columns", args.ncol,
          "\tnumber iterations", args.num_iter))

    algorithm = "comparison" if len(args.algorithm) > 1 else args.algorithm[0]

    if algorithm is "comparison":
        comparison = None
        for nitr, alg in enumerate(args.algorithm):
            print("Reconstructing {} with {}...".format(args.phantom, alg))
            tmp = run(args.phantom, alg, args, get_recon=True)
            if comparison is None:
                comparison = ImageComparison(
                    len(args.algorithm), tmp.shape[0], tmp[0].shape[0],
                    tmp[0].shape[1], image_quality["orig"]
                    )
            comparison.assign(alg, nitr+1, tmp)
        bname = get_basepath(args, algorithm, args.phantom)
        fname = os.path.join(bname, "stack_{}_".format(comparison.tagname()))
        dname = os.path.join(bname, "diff_{}_".format(comparison.tagname()))
        imgs = []
        imgs.extend(
            output_images(comparison.array, fname,
                          args.format, args.scale, args.ncol))
        imgs.extend(
            output_images(comparison.delta, dname,
                          args.format, args.scale, args.ncol))
    else:
        print("Reconstructing with {}...".format(algorithm))
        imgs = run(args.phantom, algorithm, args)

    # timing report to stdout
    print('{}\n'.format(manager))

    timemory.options.output_dir = "{}/{}/{}".format(
        args.output_dir, args.phantom, algorithm)
    timemory.options.set_report("run_tomopy.out")
    timemory.options.set_serial("run_tomopy.json")
    manager.report()

    # provide timing plots
    try:
        timemory.plotting.plot(files=[timemory.options.serial_filename],
                               echo_dart=True,
                               output_dir=timemory.options.output_dir)
    except Exception as e:
        print("Exception - {}".format(e))

    # provide results to dashboard
    try:
        for i in range(0, len(imgs)):
            img_base = "{}_{}_stack_{}".format(args.phantom, algorithm, i)
            img_name = os.path.basename(imgs[i]).replace(
                ".{}".format(args.format), "").replace(
                "stack_{}_".format(algorithm), img_base)
            img_type = args.format
            img_path = imgs[i]
            timemory.plotting.echo_dart_tag(img_name, img_path, img_type)
    except Exception as e:
        print("Exception - {}".format(e))

    # provide ASCII results
    try:
        notes = manager.write_ctest_notes(
            directory="{}/{}/{}".format(args.output_dir, args.phantom,
                                        algorithm))
        print('"{}" wrote CTest notes file : {}'.format(__file__, notes))
    except Exception as e:
        print("Exception - {}".format(e))


if __name__ == "__main__":
    """Parse inputs then call main function."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--phantom",
        help="name a phantom to use for the benchmark(s)",
        default="lena", choices=phantom_choices, type=str)
    parser.add_argument(
        "-a", "--algorithm",
        help=("name one or more reconstruction algorithm to use for the "
              "benchmark(s)"),
        default=["art"], choices=algorithm_choices + ["all"], type=str,
        nargs='+')
    parser.add_argument(
        "-A", "--angles",
        help="specify the number projection angles for the simulation(s)",
        default=360, type=int)
    parser.add_argument(
        "-s", "--size",
        help="specify the edge length in pixels of the reconstructed image(s)",
        default=512, type=int)
    parser.add_argument(
        "-n", "--ncores", help="specify the number of cpu cores to use",
        default=mp.cpu_count(), type=int)
    parser.add_argument(
        "-f", "--format",
        help="specify the file format of the output images",
        default="jpeg", type=str)
    parser.add_argument(
        "-S", "--scale",
        help="scale size of the output images by this positive factor",
        default=1, type=int)
    # FIXME: Add better docstring for ncol. What rows?
    parser.add_argument(
        "-c", "--ncol", help="Number of images per row",
        default=1, type=int)
    parser.add_argument(
        "-i", "--num-iter",
        help="specify a number of iterations for iterative algorithms",
        default=10, type=int)

    args = timemory.options.add_args_and_parse_known(parser)

    print("\nargs: {}\n".format(args))

    # FIXME: unnecessary? timemory already sets output_dir to "."
    if args.output_dir is None:
        args.output_dir = "."

    # create a folder for the phantom
    pdir = os.path.join(os.getcwd(), args.output_dir, args.phantom)
    if not os.path.exists(pdir):
        os.makedirs(pdir)

    # Check if algorithm is "all"
    if len(args.algorithm) == 1 and args.algorithm[0].lower() == "all":
        args.algorithm = list(algorithm_choices)

    # Make the folder for output images; remove existing
    if len(args.algorithm) > 1:
        alg = "comparison"
    else:
        alg = args.algorithm[0]
    adir = os.path.join(pdir, alg)
    shutil.rmtree(adir, ignore_errors=True)
    os.makedirs(adir)

    try:
        with timemory.util.timer('\nTotal time for "{}"'.format(__file__)):
            main(args)
        sys.exit(0)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print('Exception - {}'.format(e))
        sys.exit(2)
