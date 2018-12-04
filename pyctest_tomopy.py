#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for TomoPy
"""

import os
import sys
import glob
import shutil
import platform
import argparse
import traceback
import subprocess as sp
import multiprocessing as mp

import pyctest.pyctest as pyctest
import pyctest.pycmake as pycmake
import pyctest.helpers as helpers


def cleanup(path=None, exclude=[]):
    files = [ "coverage.xml", "pyctest_tomopy_rec.py",
              "pyctest_tomopy_phantom.py", "pyctest_tomopy_utils.py"]

    sp.call((sys.executable, os.path.join(os.getcwd(), "setup.py"), "clean"))
    helpers.RemovePath(os.path.join(os.getcwd(), "tomopy.egg-info"))
    helpers.RemovePath(os.path.join(os.getcwd(), "MANIFEST"))
    helpers.Cleanup(path, extra=files, exclude=exclude)


def configure():
    # Get pyctest argument parser that include PyCTest arguments
    parser = helpers.ArgumentParser(project_name="TomoPy",
                                    source_dir=os.getcwd(),
                                    binary_dir=os.getcwd(),
                                    python_exe=sys.executable,
                                    submit=False,
                                    ctest_args=["-V"])
    # default algorithm choices
    available_algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem',
                            'sirt', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                            'pml_quad', 'tv', 'grad']
    # default phantom choices
    available_phantoms = ["baboon", "cameraman", "barbara", "checkerboard",
                          "lena", "peppers", "shepp2d", "shepp3d"]
    # choices for algorithms
    algorithm_choices = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem',
                         'sirt', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                         'pml_quad', 'tv', 'grad', 'none', 'all']
    # phantom choices
    phantom_choices = ["baboon", "cameraman", "barbara", "checkerboard",
                       "lena", "peppers", "shepp2d", "shepp3d", "none", "all"]
    # number of iterations
    default_nitr = 10
    # number of cores
    default_ncores = mp.cpu_count()
    # default algorithm choices
    default_algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem',
                          'sirt', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                          'pml_quad', 'tv', 'grad']
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
    parser.add_argument("--phantom-size",
                        type=int,
                        help="Size parameter for the phantom reconstructions",
                        default=None)
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
    parser.add_argument("--skip-cleanup",
                        help="Skip cleanup of any old pyctest files",
                        action='store_true',
                        default=False)
    parser.add_argument("--cleanup",
                        help="Cleanup of any old pyctest files and exit",
                        action='store_true',
                        default=False)
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

    if args.cleanup:
        cleanup(pyctest.BINARY_DIRECTORY)
        sys.exit(0)

    if not args.skip_cleanup:
        cleanup(pyctest.BINARY_DIRECTORY)

    def remove_entry(entry, container):
        if entry in container:
            container.remove(entry)

    def remove_duplicates(container):
        container = list(set(container))

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
            "ENV{CFLAGS}", "-g -O0 -fprofile-arcs -ftest-coverage")
        pyctest.set("ENV{LDFLAGS}", "-fprofile-arcs -lgcov")

    git_exe = helpers.FindExePath("git")
    if git_exe is not None:
        pyctest.UPDATE_COMMAND = "{}".format(git_exe)
        pyctest.set("CTEST_UPDATE_TYPE", "git")

    return args


def run_pyctest():
    # run argparse, checkout source, copy over files
    args = configure()
    # Change the build name to somthing other than default
    pyctest.BUILD_NAME = "[{}] [{} {} {}] [Python ({}) {}]".format(
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        helpers.GetSystemVersionInfo(),
        platform.uname()[4],
        platform.python_implementation(),
        platform.python_version())
    # when coverage is enabled, we compile in debug so modify the build name
    # so that the history of test timing is not affected
    if args.coverage:
        pyctest.BUILD_NAME = "{} [coverage]".format(pyctest.BUILD_NAME)
    # remove any consecutive spaces
    while "  " in pyctest.BUILD_NAME:
        pyctest.BUILD_NAME = pyctest.BUILD_NAME.replace("  ", " ")
    # how to build the code
    pyctest.BUILD_COMMAND = "{} setup.py install".format(
        pyctest.PYTHON_EXECUTABLE)
    # generate the code coverage
    python_path = os.path.dirname(pyctest.PYTHON_EXECUTABLE)
    cover_exe = helpers.FindExePath("coverage", path=python_path)
    if args.coverage:
        gcov_cmd = helpers.FindExePath("gcov")
        if gcov_cmd is not None:
            pyctest.COVERAGE_COMMAND = "{}".format(gcov_cmd)
            pyctest.set("CTEST_COVERAGE_EXTRA_FLAGS", "-m")
            pyctest.set("CTEST_EXTRA_COVERAGE_GLOB", "{}/*.gcno".format(
                pyctest.SOURCE_DIRECTORY))
    else:
        # assign to just generate python coverage
        pyctest.COVERAGE_COMMAND = "{};xml".format(cover_exe)
    # copy over files from os.getcwd() to pyctest.BINARY_DIR
    # (implicitly copies over PyCTest{Pre,Post}Init.cmake if they exist)
    copy_files = [os.path.join("benchmarking", "pyctest_tomopy_utils.py"),
                  os.path.join("benchmarking", "pyctest_tomopy_phantom.py"),
                  os.path.join("benchmarking", "pyctest_tomopy_rec.py")]
    pyctest.copy_files(copy_files)
    # find the CTEST_TOKEN_FILE
    home = helpers.GetHomePath()
    if home is not None:
        token_path = os.path.join(home, ".tokens", "nersc-tomopy")
        if os.path.exists(token_path):
            pyctest.set("CTEST_TOKEN_FILE", token_path)
    # create a CTest that checks we imported the correct module
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
    # create a CTest that wraps "nosetest"
    test = pyctest.test()
    test.SetName("nosetests")
    test.SetProperty("DEPENDS", "correct_module")
    test.SetProperty("RUN_SERIAL", "ON")
    nosetest_exe = helpers.FindExePath("nosetests", path=python_path)
    if nosetest_exe is None:
        nosetest_exe = helpers.FindExePath("nosetests")
    coverage_exe = helpers.FindExePath("coverage", path=python_path)
    if coverage_exe is None:
        coverage_exe = helpers.FindExePath("coverage")
    # python $(which coverage) run $(which nosetests)
    test.SetCommand([pyctest.PYTHON_EXECUTABLE, coverage_exe, "run",
                    nosetest_exe])
    # set directory to run test
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")
    # Generating C code coverage is enabled
    if args.coverage:
        # if generating C code coverage, generating the Python coverage
        # needs to be put inside a test (that runs after nosetest)
        # because pyctest.COVERAGE_COMMAND is used to generate GCov files
        coverage_cmd = ""
        if platform.system() != "Windows":
            cover_cmd = os.path.join(pyctest.SOURCE_DIRECTORY,
                                     "benchmarking", "generate_coverage.sh")
            coverage_cmd = [cover_cmd, pyctest.SOURCE_DIRECTORY]
        else:
            # don't attempt GCov on Windows
            cover_cmd = helpers.FindExePath("coverage", path=python_path)
            coverage_cmd = [cover_cmd, "xml"]
        test = pyctest.test()
        test.SetName("python_coverage")
        test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
        test.SetProperty("DEPENDS", "nosetests")
        test.SetCommand(coverage_cmd)
    # If path to globus is provided, skip when generating C coverage (too long)
    if not args.coverage and args.globus_path is not None:
        phantom = "tomo_00001"
        h5file = os.path.join(args.globus_path, phantom, phantom + ".h5")
        if not os.path.exists(h5file):
            print("Warning! HDF5 file '{}' does not exists! "
                  "Skipping test...".format(h5file))
            h5file = None
        # loop over args.algorithms and create tests for each
        for algorithm in args.algorithms:
            test = pyctest.test()
            name = "{}_{}".format(phantom, algorithm)
            # original number of iterations before num-iter added to test name
            if args.num_iter != 10:
                name = "{}_itr{}".format(name, args.num_iter)
            test.SetName(name)
            test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
            test.SetProperty("TIMEOUT", "7200")  # 2 hour
            test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")
            test.SetProperty("DEPENDS", "nosetests")
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
    # loop over args.phantoms, skip when generating C coverage (too long)
    if not args.coverage and not args.disable_phantom_tests:
        for phantom in args.phantoms:
            # create a test comparing all the args.algorithms
            test = pyctest.test()
            name = "{}_{}".format(phantom, "comparison")

            nsize = 512 if phantom != "shepp3d" else 128
            # if size customized, create unique test-name
            if args.phantom_size is not None and args.phantom_size != 512:
                nsize = (args.phantom_size if phantom != "shepp3d" else
                         int(args.phantom_size / 4))
                name = "{}_pix{}".format(name, nsize)
            # original number of iterations before num-iter added to test name
            if args.num_iter != 10:
                name = "{}_itr{}".format(name, args.num_iter)

            test.SetName(name)
            test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
            test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")
            test.SetProperty("TIMEOUT", "10800")  # 3 hours
            test.SetProperty("DEPENDS", "nosetests")
            ncores = args.ncores
            niters = args.num_iter
            if phantom == "shepp3d":
                test.SetProperty("RUN_SERIAL", "ON")
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
    # generate the CTestConfig.cmake and CTestCustom.cmake
    pyctest.generate_config(pyctest.BINARY_DIRECTORY)
    # generate the CTestTestfile.cmake file
    pyctest.generate_test_file(pyctest.BINARY_DIRECTORY)
    # run CTest
    pyctest.run(pyctest.ARGUMENTS, pyctest.BINARY_DIRECTORY)


if __name__ == "__main__":
    try:
        run_pyctest()
    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)
    sys.exit(0)
