#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for TomoPy
"""

import os
import sys
import shutil
import platform
import argparse
import traceback
import multiprocessing

import pyctest.pyctest as pyctest
import pyctest.pycmake as pycmake
import pyctest.helpers as helpers



#------------------------------------------------------------------------------#
#
def cleanup(path=None, exclude=[]):
    files = [ "coverage.xml", "pyctest_tomopy_rec.py",
               "pyctest_tomopy_phantom.py", "pyctest_tomopy_utils.py",
               "tomopy/sharedlibs/libtomopy.so",
               "tomopy/sharedlibs/libtomopy.dll",
               "tomopy/sharedlibs/libtomopy.dylib",
               "tomopy.egg-info", "dist", "build"]

    helpers.Cleanup(path, extra=files, exclude=exclude)


#------------------------------------------------------------------------------#
#
def configure():

    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="TomoPy",
                                    source_dir=os.getcwd(),
                                    binary_dir=os.getcwd(),
                                    python_exe=sys.executable,
                                    submit=False)

    # default algorithm choices
    available_algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem', 'sirt',
                            'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad',
                            'tv', 'grad']
    # default phantom choices
    available_phantoms = ["baboon", "cameraman", "barbara", "checkerboard",
                          "lena", "peppers", "shepp2d", "shepp3d"]

    # choices for algorithms
    algorithm_choices = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem', 'sirt',
                         'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad',
                         'tv', 'grad', 'none', 'all']
    # phantom choices
    phantom_choices = ["baboon", "cameraman", "barbara", "checkerboard",
                       "lena", "peppers", "shepp2d", "shepp3d", "none", "all"]

    # number of cores
    default_ncores = multiprocessing.cpu_count()
    # number of iterations
    default_nitr = 10
    # default algorithm choices
    default_algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem', 'sirt',
                          'ospml_hybrid', 'ospml_quad', 'pml_hybrid', 'pml_quad',
                          'tv', 'grad']
    # default phantom choices
    default_phantoms = ["baboon", "cameraman", "barbara", "checkerboard",
                        "lena", "peppers", "shepp2d", "shepp3d"]

    parser.add_argument("-n", "--ncores",
                        help="number of cores",
                        type=int,
                        default=default_ncores)
    parser.add_argument("-i", "--num-iter",
                        help="number of iterations",
                        type=int,
                        default=default_nitr)
    parser.add_argument("--phantoms",
                        help="Phantoms to simulate",
                        type=str,
                        nargs='*',
                        choices=phantom_choices,
                        default=default_phantoms)
    parser.add_argument("--algorithms",
                        help="Algorithms to use",
                        type=str,
                        nargs='*',
                        choices=algorithm_choices,
                        default=default_algorithms)
    parser.add_argument("--globus-path",
                        help="Path to tomobank datasets",
                        type=str,
                        default=None)
    parser.add_argument("--cleanup",
                        help="Cleanup pyctest files",
                        type=str,
                        default=None)
    parser.add_argument("--coverage",
                        help="Enable coverage for compiled code",
                        action='store_true',
                        default=False)
    parser.add_argument("--disable-phantom-tests",
                        help="Disable running phantom tests",
                        action='store_true',
                        default=False)

    # calls PyCTest.helpers.ArgumentParser.parse_args()
    args = parser.parse_args()

    if args.cleanup is not None:
        cleanup(args.cleanup)
        sys.exit(0)

    #-----------------------------------#
    def remove_entry(entry, container):
        for itr in container:
            if itr == entry:
                del itr
    #-----------------------------------#

    #-----------------------------------#
    def remove_duplicates(container):
        container = list(set(container))
    #-----------------------------------#

    if "all" in args.algorithms:
        remove_entry("all", args.algorithms)
        args.algorithms.extend(available_algorithms)

    if "all" in args.phantoms:
        remove_entry("all", args.phantoms)
        args.phantoms.extend(available_phantoms)

    if "none" in args.algorithms:
        args.algorithms = []

    if "none" in args.phantoms:
        args.phantoms = []

    remove_duplicates(args.algorithms)
    remove_duplicates(args.phantoms)

    if args.coverage:
        # read by Makefile.linux and Makefile.darwin
        pyctest.set(
            "ENV{CFLAGS}", "-g -O0 -fprofile-arcs -ftest-coverage -fprofile-dir={}".format(pyctest.BINARY_DIRECTORY))
        pyctest.set("ENV{LD_FLAGS}", "-fprofile-arcs -lgcov")

    return args


#------------------------------------------------------------------------------#
def run_pyctest():

    #--------------------------------------------------------------------------#
    # run argparse, checkout source, copy over files
    #
    args = configure()

    #--------------------------------------------------------------------------#
    # Change the build name to somthing other than default
    #
    pyctest.BUILD_NAME = "[{}] [{} {} {}] [Python ({}) {}]".format(
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4],
        platform.python_implementation(),
        platform.python_version())

    # remove any consecutive spaces
    while "  " in pyctest.BUILD_NAME:
        pyctest.BUILD_NAME = pyctest.BUILD_NAME.replace("  ", " ")

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyctest.BUILD_COMMAND = "{} setup.py install".format(
        pyctest.PYTHON_EXECUTABLE)

    #--------------------------------------------------------------------------#
    # generate the code coverage
    #
    python_path = os.path.dirname(pyctest.PYTHON_EXECUTABLE)
    if platform.system() != "Windows":
        cover_cmd = os.path.join(pyctest.SOURCE_DIRECTORY,
                        os.path.join("benchmarking", "generate_coverage.sh"))
        cover_arg = pyctest.SOURCE_DIRECTORY
        pyctest.COVERAGE_COMMAND = "{};{}".format(cover_cmd, cover_arg)
    else:
        # don't attempt GCov on Windows
        cover_cmd = helpers.FindExePath("coverage", path=python_path)
        if cover_cmd is None:
            helpers.FindExePath("coverage")
        pyctest.COVERAGE_COMMAND = "{};xml".format(cover_cmd)


    #--------------------------------------------------------------------------#
    # copy over files from os.getcwd() to pyctest.BINARY_DIR
    # (implicitly copies over PyCTest{Pre,Post}Init.cmake if they exist)
    #
    copy_files = [os.path.join("benchmarking", "pyctest_tomopy_utils.py"),
                  os.path.join("benchmarking", "pyctest_tomopy_phantom.py"),
                  os.path.join("benchmarking", "pyctest_tomopy_rec.py")]
    pyctest.copy_files(copy_files)

    #--------------------------------------------------------------------------#
    # find the CTEST_TOKEN_FILE
    #
    home = helpers.GetHomePath()
    if home is not None:
        token_path = os.path.join(
            home, os.path.join(".tokens", "nersc-tomopy"))
        if os.path.exists(token_path):
            pyctest.set("CTEST_TOKEN_FILE", token_path)

    #--------------------------------------------------------------------------#
    # create a CTest that checks we imported the correct module
    #
    test = pyctest.test()
    test.SetName("correct_module")
    test.SetCommand([pyctest.PYTHON_EXECUTABLE, "-c",
                     "\"import os, sys, tomopy; " +
                     "print('using tomopy module: {}'.format(tomopy.__file__)); " +
                     "ret=0 if os.getcwd() in tomopy.__file__ else 1; " +
                     "sys.exit(ret)\""])
    # set directory to run test
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")

    #--------------------------------------------------------------------------#
    # create a CTest that wraps "nosetest"
    #
    test = pyctest.test()
    test.SetName("nosetests")
    nosetest_exe = helpers.FindExePath("nosetests", path=python_path)
    if nosetest_exe is None:
        nosetest_exe = helpers.FindExePath("nosetests")
    coverage_exe = helpers.FindExePath("coverage", path=python_path)
    if coverage_exe is None:
        coverage_exe = helpers.FindExePath("coverage")
    # python $(which coverage) run $(which nosetests)
    test.SetCommand([pyctest.PYTHON_EXECUTABLE, coverage_exe, "run", nosetest_exe])
    # set directory to run test
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")

    #--------------------------------------------------------------------------#
    #
    if not args.coverage and args.globus_path is not None:
        phantom = "tomo_00001"
        h5file = os.path.join(
            args.globus_path, os.path.join(phantom, phantom + ".h5"))
        if not os.path.exists(h5file):
            print("Warning! HDF5 file '{}' does not exists! Skipping test...".format(h5file))
            h5file = None

        # loop over args.algorithms and create tests for each
        for algorithm in args.algorithms:
            test = pyctest.test()
            name = "{}_{}".format(phantom, algorithm)
            test.SetName(name)
            test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
            test.SetProperty("TIMEOUT", "3600")  # 1 hour
            test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")
            if h5file is None:
                test.SetCommand([pyctest.PYTHON_EXECUTABLE,
                                "-c",
                                "print(\"Path to Globus file '{}/{}.h5' not specified\")".format(
                                    phantom, phantom)])
            else:
                test.SetCommand([pyctest.PYTHON_EXECUTABLE,
                                ".//benchmarking/pyctest_tomopy_rec.py",
                                h5file,
                                "-a", algorithm,
                                "--type", "slice",
                                "-f", "jpeg",
                                "-S", "1",
                                "-c", "4",
                                "-o", "benchmarking/{}".format(name),
                                "-n", "{}".format(args.ncores),
                                "-i", "{}".format(args.num_iter)])

    #--------------------------------------------------------------------------#
    # loop over args.phantoms
    #
    if not args.coverage and not args.disable_phantom_tests:
        for phantom in args.phantoms:
            nsize = 512 if phantom != "shepp3d" else 128
            # create a test comparing all the args.algorithms
            test = pyctest.test()
            name = "{}_{}".format(phantom, "comparison")
            test.SetName(name)
            test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
            test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")
            test.SetProperty("TIMEOUT", "10800")  # 3 hours
            ncores = args.ncores
            niters = args.num_iter
            if phantom == "shepp3d":
                test.SetProperty("RUN_SERIAL", "ON")
                ncores = multiprocessing.cpu_count()
            test.SetCommand([pyctest.PYTHON_EXECUTABLE,
                            "./benchmarking/pyctest_tomopy_phantom.py",
                            "-p", phantom,
                            "-s", "{}".format(nsize),
                            "-A", "360",
                            "-f", "jpeg",
                            "-S", "1",
                            "-n", "{}".format(ncores),
                            "-i", "{}".format(niters),
                            "--output-dir", "benchmarking/{}".format(name),
                            "--compare"] + args.algorithms)

    #--------------------------------------------------------------------------#
    # generate the CTestConfig.cmake and CTestCustom.cmake
    #
    pyctest.generate_config(pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # generate the CTestTestfile.cmake file
    #
    pyctest.generate_test_file(pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # not used but can run scripts
    # pyctest.add_presubmit_command(pyctest.BINARY_DIRECTORY,
    #    [os.path.join(pyctest.BINARY_DIRECTORY, "measurement.py"), "Coverage",
    #     os.path.join(pyctest.BINARY_DIRECTORY, "cover.xml"), "text/xml"],
    #    clobber=True)
    #
    # pyctest.add_note(pyctest.BINARY_DIRECTORY,
    #   os.path.join(pyctest.BINARY_DIRECTORY, "cover.xml"), clobber=True)

    #--------------------------------------------------------------------------#
    # run CTest -- e.g.
    #   ctest -V
    #         -DSTAGES="Start;Build;Test;Coverage"
    #         -S Stages.cmake
    #         <binary_dir>
    #
    ctest_args = pyctest.ARGUMENTS
    if not "-V" in ctest_args:
        ctest_args.append("-V")
    pyctest.run(ctest_args, pyctest.BINARY_DIRECTORY)

#------------------------------------------------------------------------------#
if __name__ == "__main__":

    try:

        run_pyctest()

    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)

    sys.exit(0)
