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
import xdesign as xd
import pprint

import tomopy
import timemory
import timemory.options as options

from benchmarking.utils import *


def get_basepath(
    output_dir="", phantom="", algorithm="", filter_name="", **kwargs
):
    """Return the folder where data for a given reconstruction goes."""
    return os.path.join(
        output_dir, phantom, algorithm, filter_name,
    )


@timemory.util.auto_timer()
def generate_phantom(
    phantom="shepp3d", nsize=512, nangles=360, output_dir="", format="png",
    **kwargs,
):
    """Simulate data acquisition for tomography using TomoPy.

    Reorder the projections optimally and save a numpyz file with the original,
    sinogram, and angles to the disk. Return the original.
    """
    with timemory.util.auto_timer("[tomopy.misc.phantom.{}]".format(phantom)):
        original = getattr(tomopy.misc.phantom, phantom)(size=nsize)
    angles = tomopy.angles(nangles)
    # Reorder projections optimally
    p = multilevel_order(len(angles)).astype(np.int32)
    angles = angles[p, ...]
    with timemory.util.auto_timer("[tomopy.project]"):
        sinogram = tomopy.project(original, angles, pad=True)

    basepath = get_basepath(output_dir, phantom=phantom, algorithm="")
    os.makedirs(basepath, exist_ok=True)
    dynam_range = np.max(original) - np.min(original)
    save_image(
        original[0, ...],
        os.path.join(basepath, "original.{}".format(format.lower())),
        vmin=0, vmax=1.1*dynam_range,
        )
    np.savez(
        os.path.join(basepath, "simulated_data.npz"),
        original=original, angles=angles, sinogram=sinogram
    )
    return original


@timemory.util.auto_timer()
def run(
    phantom, algorithm, ncores, num_iter,
    output_dir="", filter_name="", format="png", **kwargs
):
    """Run reconstruction benchmarks for phantoms using algorithm.

    Save each iteration's reconstructed image to the algorithms folder along
    with the MS-SSIM rating for that image. Return the best reconstructed
    image.
    """
    # Load the simulated from the disk
    basepath = get_basepath(output_dir=output_dir, phantom=phantom)
    simdata = np.load(os.path.join(basepath, "simulated_data.npz"))
    obj = simdata["original"]
    dynamic_range = np.max(obj) - np.min(obj)

    # Set kwargs for tomopy.recon
    _kwargs = {
        "algorithm": algorithm,
        "ncore": ncores,
    }
    # don't assign "num_iter" if gridrec or fbp
    step = 1
    if algorithm not in ["fbp", "gridrec"]:
        _kwargs["num_iter"] = step
    else:
        num_iter = 1

    print("Reconstructing {} with {}...".format(phantom, _kwargs))

    best_quality = np.zeros(len(obj))
    best_rec = np.zeros_like(obj)
    # initial reconstruction guess; use defaults unique to each algorithm
    recon = None
    basepath = get_basepath(output_dir, phantom, algorithm, filter_name)
    os.makedirs(basepath, exist_ok=True)
    for i in range(1, num_iter+1, step):
        filename = os.path.join(basepath, "{:03d}".format(i))
        # look for the ouput; only reconstruct if it doesn't exist
        if False:  # os.path.isfile(filename + '.npz'):
            existing_data = np.load(filename + '.npz')
            recon = existing_data['recon']
        else:
            with timemory.util.auto_timer(
                "[tomopy.recon(algorithm='{}')]".format(algorithm)
            ):
                recon = tomopy.recon(
                    init_recon=recon,
                    tomo=simdata['sinogram'],
                    theta=simdata['angles'],
                    **_kwargs,
                )
        # padding was added to keep square image in the field of view
        rec = trim_border(
            recon, recon.shape[0],
            recon.shape[1] - obj.shape[1],
            recon.shape[2] - obj.shape[2],
        )
        msssim = np.empty(len(obj))
        for z in range(len(obj)):
            # compute the reconstructed image quality metrics
            scales, msssim[z], quality_maps = xd.msssim(
                obj[z],
                rec[z],
                L=dynamic_range,
            )
            if msssim[z] > best_quality[z]:
                best_quality[z] = msssim[z]
                best_rec[z] = rec[z]
        print("[{phantom} {algo} @ {i}] : ms-ssim = {msssim:05.3f}".format(
            algo=algorithm.upper(),
            phantom=phantom.upper(),
            msssim=np.mean(msssim),
            i=i,
        ))
        # save all information
        np.savez(
            filename + ".npz",
            recon=rec,
            msssim=msssim,
        )
        save_image(
            rec[0],
            "{}.{}".format(filename, format.lower()),
            vmin=0, vmax=1.1*dynamic_range,
        )
    return best_rec


def main(args):
    """Run benchmarks and publish reports.

    Create the following file structure to organize the results.
    ```
    /{output_dir}
        /{phantom}
            /{algorithm[0]}
                001.{format}  # RGB reconstruction after 1 iteration
                001.npz  # float reconstruction and quality rating
                ...
            /{algorithm[1]}
                001.{format}
                001.npz
                ...
            ...
            diff_soln-{algoritms}_0.{format}  # comparison of difference images
            ...
            original.{format}  # the original phantom
            # other benchmarking artifacts here
            simulated_data.npz  # contains sinogram and angles
            stack_soln-{algorithms}_0.{format}   # comparison of best images
            ...
    ```

    FIXME: Document the expected filestructure, types of files that are
    created, and what information they contain.
    """
    manager = timemory.manager()

    original = generate_phantom(**vars(args))

    comparison = ImageComparison(
        len(args.algorithm),
        original.shape[0], original.shape[1], original.shape[2],
        solution=original
        )

    # For each algorithm run benchmarks
    for nitr, alg in enumerate(args.algorithm):
        params = vars(args).copy()
        params["algorithm"] = alg
        comparison.assign(alg, nitr+1, run(get_recon=True, **params))

    # Publish reports

    # Save results and diff images from each reconstruction algorithm
    bname = get_basepath(output_dir=args.output_dir, phantom=args.phantom)
    fname = os.path.join(bname, "stack_{}".format(comparison.tagname()))
    dname = os.path.join(bname, "diff_{}".format(comparison.tagname()))
    imgs = []
    imgs.extend(
        output_images(comparison.results, fname,
                      args.format, args.scale,
                      # scale images with range of original
                      vmin=0, vmax=np.max(original),
                      )
    )
    imgs.extend(
        output_images(comparison.delta, dname,
                      args.format, args.scale,
                      # scale images =/- range of original
                      vmin=-np.max(original), vmax=np.max(original)
                      )
    )

    # timing report to stdout
    print('{}\n'.format(manager))

    timemory.options.output_dir = get_basepath(
        output_dir=args.output_dir,
        phantom=args.phantom,
    )
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
            # remove file extension
            img_name = os.path.splitext(imgs[i])[0]
            # remove file path
            img_name = os.path.basename(img_name)
            # prepend phantom name
            img_name = "_".join([args.phantom, img_name])
            timemory.plotting.echo_dart_tag(
                img_name, filepath=imgs[i], img_type=args.format
            )
    except Exception as e:
        print("Exception - {}".format(e))

    # provide ASCII results
    try:
        notes = manager.write_ctest_notes(
            directory=timemory.options.output_dir)
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

    # Replace "all" with list of all algorithms
    if len(args.algorithm) == 1 and args.algorithm[0].lower() == "all":
        args.algorithm = list(algorithm_choices)
    # Replace relative paths with full paths
    args.output_dir = os.path.abspath(args.output_dir)

    # "Pretty print" the input arguments
    print("Arguments:")
    pprint.pprint(vars(args), indent=4)

    # create a folder for the phantom
    # FIXME: Should we overwrite exisiting data or raise an error?
    pdir = get_basepath(output_dir=args.output_dir, phantom=args.phantom)
    if os.path.exists(pdir):
        # raise FileExistsError(
        #     "{} exists. Choose another output directory "
        #     "to prevent overwriting exisiting data.".format(pdir)
        # )
        shutil.rmtree(pdir, ignore_errors=True)

    try:
        with timemory.util.timer('\nTotal time for "{}"'.format(__file__)):
            main(args)
        sys.exit(0)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print('Exception - {}'.format(e))
        sys.exit(2)
