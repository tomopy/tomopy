#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil, subprocess as sp
from skbuild import setup
from skbuild.setuptools_wrap import create_skbuild_argparser
import argparse
import warnings
import platform

cmake_args = []
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", help="Print help", action='store_true')

def add_bool_opt(opt, enable_opt, disable_opt):
    global cmake_args
    if enable_opt and disable_opt:
        msg = """\nWarning! python options for CMake argument '{}' was enabled \
AND disabled.\nGiving priority to disable...\n""".format(opt)
        warnings.warn(msg)
        enable_opt = False

    if enable_opt:
        cmake_args.append("-D{}:BOOL={}".format(opt, "ON"))
    if disable_opt:
        cmake_args.append("-D{}:BOOL={}".format(opt, "OFF"))

def add_option(lc_name, disp_name):
    global parser
    # enable option
    parser.add_argument("--enable-{}".format(lc_name), action='store_true',
                        help="Explicitly enable {} build".format(disp_name))
    # disable option
    parser.add_argument("--disable-{}".format(lc_name), action='store_true',
                        help="Explicitly disable {} build".format(disp_name))

add_option("cuda", "CUDA")
add_option("nvtx", "NVTX (NVIDIA Nsight)")
add_option("arch", "Hardware optimized")
add_option("avx512", "AVX-512 optimized")
add_option("gperf", "gperftools")
add_option("timemory", "TiMemory")
add_option("sanitizer", "Enable sanitizer (default=leak)")
add_option("tasking", "Tasking library (PTL)")

parser.add_argument("--sanitizer-type", default="leak",
                    help="Set the sanitizer type",
                    type=str, choices=["leak", "thread", "address", "memory"])
parser.add_argument("--cuda-arch", help="CUDA architecture flag",
                    type=int, default=35)

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1]+left

add_bool_opt("TOMOPY_USE_CUDA", args.enable_cuda, args.disable_cuda)
add_bool_opt("TOMOPY_USE_NVTX", args.enable_nvtx, args.disable_nvtx)
if args.enable_avx512 and not args.enable_arch:
    args.enable_arch = True
    args.disable_arch = False
add_bool_opt("TOMOPY_USE_ARCH", args.enable_arch, args.disable_arch)
add_bool_opt("TOMOPY_USE_AVX512", args.enable_avx512, args.disable_avx512)
add_bool_opt("TOMOPY_USE_GPERF", args.enable_gperf, args.disable_gperf)
add_bool_opt("TOMOPY_USE_TIMEMORY", args.enable_timemory, args.disable_timemory)
add_bool_opt("TOMOPY_USE_SANITIZER", args.enable_sanitizer, args.disable_sanitizer)
add_bool_opt("TOMOPY_USE_PTL", args.enable_tasking, args.disable_tasking)

if args.enable_cuda:
    cmake_args.append("-DCUDA_ARCH={}".format(args.cuda_arch))

if args.enable_sanitizer:
    cmake_args.append("-DSANITIZER_TYPE:STRING={}".format(args.sanitizer_type))

if len(cmake_args) > 0:
    print("\n\n\tCMake arguments set via command line: {}\n".format(cmake_args))

if platform.system() == "Darwin":
    # scikit-build will set this to 10.6 and C++ compiler check will fail
    version = platform.mac_ver()[0].split('.')
    version = ".".join([version[0], version[1]])
    cmake_args += ["-DCMAKE_OSX_DEPLOYMENT_TARGET={}".format(version)]

# suppress:
#  "setuptools_scm/git.py:68: UserWarning: "/.../tomopy" is shallow and may cause errors"
# since 'error' in output causes CDash to interpret warning as error
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    setup(
        name='tomopy',
        packages=['tomopy'],
        package_dir={"": "source"},
        setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
        use_scm_version=True,
        include_package_data=True,
        zip_safe=False,
        author='Doga Gursoy',
        author_email='dgursoy@aps.anl.gov',
        description='Tomographic Reconstruction in Python.',
        long_description_content_type='text/x-rst',
        keywords=['tomography', 'reconstruction', 'imaging'],
        url='http://tomopy.readthedocs.org',
        download_url='http://github.com/tomopy/tomopy.git',
        license='BSD-3',
        cmake_args=cmake_args,
        cmake_languages=('C'),
        platforms='Any',
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: BSD License',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Education',
            'Intended Audience :: Developers',
            'Natural Language :: English',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: C']
    )
