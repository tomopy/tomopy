#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TomoPy script to reconstruct a built-in phantom."""

import sys
import os
import argparse
import traceback

import tomopy
import dxchange
import tornado
import matplotlib
import timemory
import timemory.options as options
import signal
import numpy as np
import time as t
import pylab
from tomopy.misc.benchmark import *


def get_basepath(args, algorithm, phantom):
    basepath = os.path.join(os.getcwd(), args.output_dir, phantom, algorithm)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    return basepath

@timemory.util.auto_timer()
def generate(phantom="shepp3d", nsize=512, nangles=360):

    with timemory.util.auto_timer("[tomopy.misc.phantom.{}]".format(phantom)):
        obj = getattr(tomopy.misc.phantom, phantom)(size=nsize)
    obj = tomopy.misc.morph.pad(obj, axis=1, mode='constant')
    obj = tomopy.misc.morph.pad(obj, axis=2, mode='constant')
    with timemory.util.auto_timer("[tomopy.angles]"):
        ang = tomopy.angles(nangles)
    with timemory.util.auto_timer("[tomopy.project]"):
        prj = tomopy.project(obj, ang)

    return [prj, ang, obj]


@timemory.util.auto_timer()
def run(phantom, algorithm, args, get_recon=False):

    global image_quality

    imgs = []
    bname = get_basepath(args, algorithm, phantom)
    pname = os.path.join(bname, "proj_{}_".format(algorithm))
    oname = os.path.join(bname, "orig_{}_".format(algorithm))
    fname = os.path.join(bname, "stack_{}_".format(algorithm))
    dname = os.path.join(bname, "diff_{}_".format(algorithm))

    prj, ang, obj = generate(phantom, args.size, args.angles)
    proj = np.zeros(shape=[prj.shape[1], prj.shape[0], prj.shape[2]], dtype=np.float)
    for i in range(0, prj.shape[1]):
        proj[i,:,:] = prj[:,i,:]
    print(proj.shape)
    output_images(proj, pname, args.format, args.scale, args.ncol)

    # always add algorithm
    _kwargs = {"algorithm": algorithm}

    # assign number of cores
    _kwargs["ncore"] = args.ncores

    # don't assign "num_iter" if gridrec or fbp
    if algorithm not in ["fbp", "gridrec"]:
        _kwargs["num_iter"] = args.num_iter

    print("kwargs: {}".format(_kwargs))
    with timemory.util.auto_timer("[tomopy.recon(algorithm='{}')]".format(
                                  algorithm)):
        rec = tomopy.recon(prj, ang, **_kwargs)

    obj_min = np.amin(obj)
    rec_min = np.amin(rec)
    obj_max = np.amax(obj)
    rec_max = np.amax(rec)
    print("obj bounds = [{:8.3f}, {:8.3f}], rec bounds = [{:8.3f}, {:8.3f}]".format(obj_min, obj_max,
                                                              rec_min, rec_max))

    obj = normalize(obj)
    rec = normalize(rec)
    obj_max = np.amax(obj)
    rec_max = np.amax(rec)
    print("Max obj = {}, rec = {}".format(obj_max, rec_max))

    rec = trim_border(rec, rec.shape[0],
                      rec[0].shape[0] - obj[0].shape[0],
                      rec[0].shape[1] - obj[0].shape[1])

    label = "{} @ {}".format(algorithm.upper(), phantom.upper())

    quantify_difference(label + " (self)", rec, np.zeros(rec.shape, dtype=rec.dtype))
    quantify_difference(label, obj, rec)

    if "orig" not in image_quality:
        image_quality["orig"] = obj

    dif = obj - rec
    image_quality[algorithm] = dif

    if get_recon is True:
        return rec


    print("oname = {}, fname = {}, dname = {}".format(oname, fname, dname))
    imgs.extend(output_images(obj, oname, args.format, args.scale, args.ncol))
    imgs.extend(output_images(rec, fname, args.format, args.scale, args.ncol))
    imgs.extend(output_images(dif, dname, args.format, args.scale, args.ncol))

    return imgs


def main(args):

    global image_quality

    manager = timemory.manager()

    algorithm = args.algorithm
    if len(args.compare) > 0:
        algorithm = "comparison"

    print(("\nArguments:\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n"
          "{} = {}\n{} = {}\n{} = {}\n{} = {}\n{} = {}\n").format(
          "\tPhantom", args.phantom,
          "\tAlgorithm", algorithm,
          "\tSize", args.size,
          "\tAngles", args.angles,
          "\tFormat", args.format,
          "\tScale", args.scale,
          "\tcomparison", args.compare,
          "\tnumber of cores", args.ncores,
          "\tnumber of columns", args.ncol,
          "\tnumber iterations", args.num_iter))

    if len(args.compare) > 0:
        args.ncol = 1
        args.scale = 1
        nitr = 1
        comparison = None
        for alg in args.compare:
            print("Reconstructing {} with {}...".format(args.phantom, alg))
            tmp = run(args.phantom, alg, args, get_recon=True)
            tmp = rescale_image(tmp, args.size, args.scale, transform=False)
            if comparison is None:
                comparison = image_comparison(
                    len(args.compare), tmp.shape[0], tmp[0].shape[0],
                    tmp[0].shape[1], image_quality["orig"]
                    )
            comparison.assign(alg, nitr, tmp)
            nitr += 1
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
        print("Reconstructing with {}...".format(args.algorithm))
        imgs = run(args.phantom, args.algorithm, args)

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

    parser = argparse.ArgumentParser()

    # phantom choices
    phantom_choices = ["baboon", "cameraman", "barbara", "checkerboard",
                       "lena", "peppers", "shepp2d", "shepp3d"]

    import multiprocessing as mp
    ncores = mp.cpu_count()

    parser.add_argument("-p", "--phantom", help="Phantom to use",
                        default="shepp2d", choices=phantom_choices, type=str)
    parser.add_argument("-a", "--algorithm", help="Select the algorithm",
                        default="sirt", choices=algorithms, type=str)
    parser.add_argument("-A", "--angles", help="number of angles",
                        default=360, type=int)
    parser.add_argument("-s", "--size", help="size of image",
                        default=512, type=int)
    parser.add_argument("-n", "--ncores", help="number of cores",
                        default=ncores, type=int)
    parser.add_argument("-f", "--format", help="output image format",
                        default="jpeg", type=str)
    parser.add_argument("-S", "--scale",
                        help="scale image by a positive factor",
                        default=1, type=int)
    parser.add_argument("-c", "--ncol", help="Number of images per row",
                        default=1, type=int)
    parser.add_argument("--compare", help="Generate comparison",
                        nargs='*', default=["none"], type=str)
    parser.add_argument("-i", "--num-iter", help="Number of iterations",
                        default=10, type=int)
    parser.add_argument("-P", "--preserve-output-dir", help="Do not clean up output directory",
                        action='store_true')
    
    args = timemory.options.add_args_and_parse_known(parser)

    print("\nargs: {}\n".format(args))

    if args.output_dir is None:
        args.output_dir = "."

    if len(args.compare) == 1 and args.compare[0].lower() == "all":
        args.compare = list(algorithms)
    elif len(args.compare) == 1:
        args.compare = []

    # unique output directory w.r.t. phantom
    adir = os.path.join(os.getcwd(), args.output_dir, args.phantom)
    # directory extension based on algorithms
    dext = "comparison" if len(args.compare) > 0 else "{}".format(args.phantom)
    # unique output directory w.r.t. phantom and extension
    adir = os.path.join(adir, dext)
    # reset as the output directory
    #args.output_dir = adir
    
    if not args.preserve_output_dir:
        try:
            import shutil
            if os.path.exists(adir) and adir != os.getcwd():
                shutil.rmtree(adir)
                os.makedirs(adir)
        except:
            pass
    else:
        os.makedirs(adir)
        
    ret = 0
    try:
        with timemory.util.timer('\nTotal time for "{}"'.format(__file__)):
            main(args)

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5)
        print('Exception - {}'.format(e))
        ret = 1

    sys.exit(ret)
