#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyCTest driver for TomoPy
"""

import os
import sys
import glob
import shutil
import argparse
import platform
import warnings
import traceback
import subprocess as sp
import multiprocessing as mp

import pyctest.pyctest as pyctest
import pyctest.pycmake as pycmake
import pyctest.helpers as helpers

from benchmarking import (
    default_nitr,
    default_phantom_size,
    find_python_nosetest,
    find_python_coverage,
    find_ctest_token,
    build_option_append,
    build_name_append,
    create_correct_module_test,
    create_nosetest_test,
    create_coverage_test,
    create_phantom_test,
    create_globus_test
)


def cleanup(path=None, exclude=[]):
    """
    route for cleaning up testing files
    """
    sp.call((sys.executable, os.path.join(os.getcwd(), "setup.py"), "clean"))
    helpers.RemovePath(os.path.join(os.getcwd(), "tomopy.egg-info"))
    helpers.RemovePath(os.path.join(os.getcwd(), "dist"))
    helpers.RemovePath(os.path.join(os.getcwd(), "MANIFEST"))
    helpers.Cleanup(path, exclude=exclude)


def configure():
    # set site if set in environ
    if os.environ.get("CTEST_SITE") is not None:
        pyctest.CTEST_SITE = os.environ.get("CTEST_SITE")

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
                            'pml_quad', 'tv', 'grad', 'tikh']
    # default phantom choices
    available_phantoms = ["baboon", "cameraman", "barbara", "checkerboard",
                          "lena", "peppers", "shepp2d", "shepp3d"]
    # choices for algorithms
    algorithm_choices = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem',
                         'sirt', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                         'pml_quad', 'tv', 'grad', 'tikh',  'none', 'all']
    # phantom choices
    phantom_choices = ["baboon", "cameraman", "barbara", "checkerboard",
                       "lena", "peppers", "shepp2d", "shepp3d", "none", "all"]
    # number of cores
    default_ncores = mp.cpu_count()
    # default algorithm choices
    default_algorithms = ['gridrec', 'art', 'fbp', 'bart', 'mlem', 'osem',
                          'sirt', 'ospml_hybrid', 'ospml_quad', 'pml_hybrid',
                          'pml_quad', 'tv', 'grad', 'tikh']
    # default phantom choices
    default_phantoms = ["baboon", "cameraman", "barbara", "checkerboard",
                        "lena", "peppers", "shepp2d", "shepp3d"]

    # default globus phantoms
    default_globus_phantoms = ["tomo_00001"]

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
    parser.add_argument("--exclude-phantoms",
                        help="Phantoms to simulate",
                        type=str,
                        nargs='*',
                        choices=default_phantoms,
                        default=[])
    parser.add_argument("--phantom-size",
                        type=int,
                        help="Size parameter for the phantom reconstructions",
                        default=default_phantom_size)
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
    parser.add_argument("--globus-phantoms",
                        help="Globus phantom files (without extension)",
                        type=str,
                        default=default_globus_phantoms,
                        nargs='*')
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
    parser.add_argument("--customize-build-name",
                        help="Customize the build name",
                        type=str,
                        default=None)
    parser.add_argument("--cuda-arch", help="CUDA architecture flag",
                        type=int, default=53)

    def add_bool_opt(args, opt, enable_opt, disable_opt):
        if enable_opt and disable_opt:
            msg = """\nWarning! python options for CMake argument '{}' was enabled \
    AND disabled.\nGiving priority to disable...\n""".format(opt)
            warnings.warn(msg)
            enable_opt = False

        if enable_opt:
            args.cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
        if disable_opt:
            args.cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))


    def add_option(parser, lc_name, disp_name):
        # enable option
        parser.add_argument("--enable-{}".format(lc_name), action='store_true',
                            help="Explicitly enable {} build".format(disp_name))
        # disable option
        parser.add_argument("--disable-{}".format(lc_name), action='store_true',
                            help="Explicitly disable {} build".format(disp_name))

    add_option(parser, "cuda", "CUDA")
    add_option(parser, "nvtx", "NVTX (NVIDIA Nsight)")
    add_option(parser, "arch", "Hardware optimized")
    add_option(parser, "avx512", "AVX-512 optimized")
    add_option(parser, "gperf", "gperftools")
    add_option(parser, "timemory", "TiMemory")
    add_option(parser, "sanitizer", "Enable sanitizer (default=leak)")
    add_option(parser, "tasking", "Tasking library (PTL)")

    parser.add_argument("--sanitizer-type", default="leak",
                        help="Set the sanitizer type",
                        type=str, choices=["leak", "thread", "address", "memory"])

    args = parser.parse_args()

    # Grab CMake args from command line
    args.cmake_args = pycmake.ARGUMENTS

    add_bool_opt(args, "TOMOPY_USE_CUDA", args.enable_cuda, args.disable_cuda)
    add_bool_opt(args, "TOMOPY_USE_NVTX", args.enable_nvtx, args.disable_nvtx)
    if args.enable_avx512 and not args.enable_arch:
        args.enable_arch = True
        args.disable_arch = False
    add_bool_opt(args, "TOMOPY_USE_ARCH", args.enable_arch, args.disable_arch)
    add_bool_opt(args, "TOMOPY_USE_AVX512", args.enable_avx512, args.disable_avx512)
    add_bool_opt(args, "TOMOPY_USE_GPERF", args.enable_gperf, args.disable_gperf)
    add_bool_opt(args, "TOMOPY_USE_TIMEMORY", args.enable_timemory, args.disable_timemory)
    add_bool_opt(args, "TOMOPY_USE_SANITIZER",
                 args.enable_sanitizer, args.disable_sanitizer)
    add_bool_opt(args, "TOMOPY_USE_PTL", args.enable_tasking, args.disable_tasking)

    if args.enable_sanitizer:
        args.cmake_args.append("-DSANITIZER_TYPE:STRING={}".format(args.sanitizer_type))

    if args.enable_cuda:
        args.cmake_args.append("-DCUDA_ARCH={}".format(args.cuda_arch))

    if len(args.cmake_args) > 0:
        print("\n\n\tCMake arguments set via command line: {}\n".format(
            args.cmake_args))

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

    for p in args.exclude_phantoms:
        if p in args.phantoms:
            args.phantoms.remove(p)

    remove_duplicates(args.algorithms)
    remove_duplicates(args.phantoms)

    git_exe = helpers.FindExePath("git")
    if git_exe is not None:
        pyctest.UPDATE_COMMAND = "{}".format(git_exe)
        pyctest.set("CTEST_UPDATE_TYPE", "git")

    if args.enable_sanitizer:
        pyctest.set("CTEST_MEMORYCHECK_TYPE", "{}Sanitizer".format(
            args.sanitizer_type.lower().capitalize()))

    return args



def run_pyctest():
    '''
    Configure PyCTest and execute
    '''
    # run argparse, checkout source, copy over files
    args = configure()

    # shorthand directories
    source_dir = pyctest.SOURCE_DIRECTORY

    # executables
    pyexe = pyctest.PYTHON_EXECUTABLE
    pycoverage = find_python_coverage()
    gcovcommand = helpers.FindExePath("gcov")
    if gcovcommand is None:
        args.coverage = False

    # properties
    bench_props = {
        "WORKING_DIRECTORY" : pyctest.SOURCE_DIRECTORY,
        "DEPENDS" : "nosetests",
        "TIMEOUT" : "10800"
    }

    #   CTEST_TOKEN
    find_ctest_token()

    #   BUILD_NAME
    pyctest.BUILD_NAME = "[{}]".format(pyctest.GetGitBranch(source_dir))
    build_name_append(platform.uname()[0], separate=False, suffix="")
    build_name_append(helpers.GetSystemVersionInfo(), prefix="(", suffix=")")
    build_name_append(platform.uname()[4], separate=False, prefix="")
    build_name_append(platform.python_version(), prefix="[Python ")
    build_name_append(args.sanitizer_type.lower(), check=args.enable_sanitizer)
    build_name_append("PTL", check=args.enable_tasking)
    build_name_append(args.customize_build_name)
    build_name_append("coverage", check=args.coverage)


    #   BUILD_COMMAND
    pyctest.BUILD_COMMAND = "{} setup.py --hide-listing -q install".format(pyexe)
    pyctest.BUILD_COMMAND += " --build-type=Debug" if args.coverage else ""
    pyctest.BUILD_COMMAND += " -- {}".format(" ".join(args.cmake_args))

    build_option_append(args.enable_sanitizer, "TOMOPY_USE_SANITIZER", "ON")
    build_option_append(args.enable_sanitizer, "SANITIZER_TYPE", args.sanitizer_type)
    build_option_append(args.coverage, "TOMOPY_USE_COVERAGE", "ON")

    print("TomoPy BUILD_COMMAND: '{}'...".format(pyctest.BUILD_COMMAND))

    #   COVERAGE_COMMAND
    pyctest.COVERAGE_COMMAND = "{};xml".format(pycoverage)
    if args.coverage:
        pyctest.COVERAGE_COMMAND = "{}".format(gcovcommand)
        pyctest.set("CTEST_COVERAGE_EXTRA_FLAGS", "-m")
        pyctest.set("CTEST_EXTRA_COVERAGE_GLOB", "{}/*.gcno".format(source_dir))

    # unit tests
    create_correct_module_test()
    create_nosetest_test(args)
    create_coverage_test(args)

    # globus tests
    for phantom in args.globus_phantoms:
        for algorithm in args.algorithms:
            create_globus_test(args, bench_props, algorithm, phantom)

    # phantom tests
    for phantom in args.phantoms:
        create_phantom_test(args, bench_props, phantom)

    print('Running PyCTest:\n\n\t{}\n\n'.format(pyctest.BUILD_NAME))

    pyctest.run()


if __name__ == "__main__":
    try:
        run_pyctest()
    except Exception as e:
        print('Error running pyctest - {}'.format(e))
        exc_type, exc_value, exc_trback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_trback, limit=10)
        sys.exit(1)
    sys.exit(0)
