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

def find_exe(name, path=None):
    try:
        from shutil import which
        return which(name, path=path) is not None
    except:
        pass

    try:
        from distutils.spawn import find_executable
        return find_executable(name, path=path)
    except:
        pass

    return None


#------------------------------------------------------------------------------#

class Conda(object):

    def __init__(self):
        self.prefix = os.environ.get("CONDA_PREFIX")
        self.environ = os.environ.get("CONDA_ENVIRONMENT")
        self.python = os.environ.get("CONDA_PYTHON_EXE")
        self.default_env = os.environ.get("CONDA_DEFAULT_ENV")
        self.shlvl = os.environ.get("CONDA_SHLVL")

    def __str__(self):
        return "{}={};{}={};{}={};{}={};{}={}".format(
            "CONDA_PREFIX", self.prefix,
            "CONDA_ENVIRONMENT", self.environ,
            "CONDA_PYTHON_EXE", self.python,
            "CONDA_DEFAULT_ENV", self.default_env,
            "CONDA_SHLVL", self.shlvl)


#------------------------------------------------------------------------------#

def cleanup(path=None):
    for f in ["Build.cmake", "CTestConfig.cmake", "CTestCustom.cmake",
              "CTestTestfile.cmake", "Coverage.cmake", "DartConfiguration.tcl",
              "Glob.cmake", "Init.cmake", "Makefile", "MemCheck.cmake",
              "PyCTestPreInit.cmake", "Stages.cmake", "Submit.cmake",
              "Test.cmake", "Testing", "Utilities.cmake",
              "cmake_install.cmake", "coverage.xml",
              "CMakeFiles", "CMakeCache.txt", "__pycache__",
              "pyctest_tomopy_rec.py", "pyctest_tomopy_phantom.py",
              "pyctest_tomopy_utils.py",
              "tomopy/sharedlibs/libtomopy.so",
              "tomopy/sharedlibs/libtomopy.dll",
              "tomopy/sharedlibs/libtomopy.dylib",
              "tomopy.egg-info", "dist", "build"]:

        if path is None:
            path = pyctest.SOURCE_DIRECTORY

        if path is not None and os.path.exists(path):
            fname = os.path.join(path, f)
            if os.path.exists(fname):
                if os.path.isdir(fname):
                    shutil.rmtree(fname)
                else:
                    os.remove(fname)


#------------------------------------------------------------------------------#


def configure():

    #conda = Conda()

    helpers.ParseArgs(project_name="TomoPy",
                      source_dir=os.getcwd(),
                      binary_dir=os.getcwd(),
                      python_exe=sys.executable)

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

    # argument parser
    parser = argparse.ArgumentParser()

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

    return args


#------------------------------------------------------------------------------#
def run_pyctest():

    #--------------------------------------------------------------------------#
    # run argparse, checkout source, copy over files
    #
    args = configure()

    #--------------------------------------------------------------------------#
    # dist info
    #
    version_info = None
    if platform.system() == 'Darwin':
        version_info = "macOS {}".format(platform.mac_ver()[0])
    elif platform.system() == 'Linux':
        version_info = "{} {}".format(platform.linux_distribution()[0],
                                      platform.linux_distribution()[1])
    else:
        version_info = "Windows {}".format(platform.version())

    #--------------------------------------------------------------------------#
    # Change the build name to somthing other than default
    #
    pyctest.BUILD_NAME = "[{}] [{}] [{} {} {}] [Python ({}) {}]".format(
        pyctest.PROJECT_NAME,
        pyctest.GetGitBranch(pyctest.SOURCE_DIRECTORY),
        platform.uname()[0],
        version_info,
        platform.uname()[4],
        platform.python_implementation(),
        platform.python_version())

    #--------------------------------------------------------------------------#
    # how to checkout the code
    #
    #pyctest.CHECKOUT_COMMAND = "${} -E copy_directory {} {}/".format(
    #    "{CTEST_CMAKE_COMMAND}",
    #    pyctest.SOURCE_DIRECTORY,
    #    pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # how to build the code
    #
    pyctest.BUILD_COMMAND = "{} setup.py install".format(
        pyctest.PYTHON_EXECUTABLE)

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
    # run CMake to generate DartConfiguration.tcl if no CMakeLists.txt
    #
    cmake_list_file = os.path.join(pyctest.BINARY_DIRECTORY, "CMakeLists.txt")
    if not os.path.exists(cmake_list_file):
        cm = pycmake.cmake(pyctest.BINARY_DIRECTORY, pyctest.PROJECT_NAME)
        #---------------------------------------------------------------------#
        # remove temporary CMakeLists.txt, CMakeCache.txt, and CMakeFiles
        cache_file = os.path.join(pyctest.BINARY_DIRECTORY, "CMakeCache.txt")
        cache_folder = os.path.join(pyctest.BINARY_DIRECTORY, "CMakeFiles")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        if os.path.exists(cache_folder):
            shutil.rmtree(cache_folder)
        if os.path.exists(cmake_list_file):
            os.remove(cmake_list_file)

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
    test.SetCommand(["nosetests", "test", "--cover-xml",
                     "--cover-xml-file=coverage.xml"])
    # set directory to run test
    test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
    test.SetProperty("ENVIRONMENT", "OMP_NUM_THREADS=1")

    #--------------------------------------------------------------------------#
    #
    h5file = None
    phantom = "tomo_00001"
    if args.globus_path is not None:
        h5file = os.path.join(
            args.globus_path, os.path.join(phantom, phantom + ".h5"))
        if not os.path.exists(h5file):
            print(
                "Warning! HDF5 file '{}' does not exists! Skipping test...".format(h5file))

    # loop over args.algorithms and create tests for each
    for algorithm in args.algorithms:
        # args.algorithms
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
    for phantom in args.phantoms:
        nsize = 512 if phantom != "shepp3d" else 128
        if phantom == "shepp3d":
            # for shepp3d only
            # loop over args.algorithms and create tests for each
            for algorithm in args.algorithms:
                #-------------------------------------------#
                # SKIP FOR NOW -- TOO MUCH OUTPUT/INFORMATION
                #-------------------------------------------#
                continue
                #-------------------------------------------#
                # args.algorithms
                test = pyctest.test()
                name = "{}_{}".format(phantom, algorithm)
                test.SetName(name)
                test.SetProperty("WORKING_DIRECTORY", pyctest.BINARY_DIRECTORY)
                test.SetCommand([pyctest.PYTHON_EXECUTABLE,
                                 "./benchmarking/pyctest_tomopy_phantom.py",
                                 "-a", algorithm,
                                 "-p", phantom,
                                 "-s", "{}".format(nsize),
                                 "-A", "360",
                                 "-f", "jpeg",
                                 "-S", "2",
                                 "-c", "8",
                                 "-n", "{}".format(args.ncores),
                                 "-i", "{}".format(args.num_iter),
                                 "--output-dir", "benchmarking/{}".format(name)])

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
    # configuration files, copy over
    # {Build,Coverage,Glob,Init,Memcheck,Stages,Submit,Test}.cmake
    # files located in the pyctest installation directory
    # - These are helpers for the workflow
    #
    pyctest.generate_config(pyctest.BINARY_DIRECTORY)

    #--------------------------------------------------------------------------#
    # generate the CTestTestfile.cmake file
    # CRITICAL:
    #   call after creating/running dummy CMake as the cmake call will
    #   generate an empty CTestTestfile.cmake file that this package overwrites
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
    # run CTest -- e.g. ctest -VV -S Test.cmake <binary_dir>
    #
    ctest_args = pyctest.ARGUMENTS
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
