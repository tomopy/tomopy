#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil, subprocess as sp
from setuptools import find_packages
from skbuild import setup
from skbuild.setuptools_wrap import create_skbuild_argparser
import argparse
import warnings

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
                        help="Explicitly disnable {} build".format(disp_name))

add_option("cuda", "CUDA")
add_option("openacc", "OpenACC")
add_option("openmp", "OpenMP")
add_option("nvtx", "NVTX (NVIDIA Nsight)")
add_option("arch", "Hardware optimized")
add_option("avx512", "AVX-512 optimized")
add_option("gperf", "gperftools")
add_option("timemory", "TiMemory")

args, left = parser.parse_known_args()
# if help was requested, print these options and then add '--help' back
# into arguments so that the skbuild/setuptools argparse catches it
if args.help:
    parser.print_help()
    left.append("--help")
sys.argv = sys.argv[:1]+left

add_bool_opt("TOMOPY_USE_CUDA", args.enable_cuda, args.disable_cuda)
add_bool_opt("TOMOPY_USE_OPENACC", args.enable_openacc, args.disable_openacc)
add_bool_opt("TOMOPY_USE_OPENMP", args.enable_openmp, args.disable_openmp)
add_bool_opt("TOMOPY_USE_NVTX", args.enable_nvtx, args.disable_nvtx)
if args.enable_avx512 and not args.enable_arch:
    args.enable_arch = True
    args.disable_arch = False
add_bool_opt("TOMOPY_USE_ARCH", args.enable_arch, args.disable_arch)
add_bool_opt("TOMOPY_USE_AVX512", args.enable_avx512, args.disable_avx512)
add_bool_opt("TOMOPY_USE_GPERF", args.enable_gperf, args.disable_gperf)
add_bool_opt("TOMOPY_USE_TIMEMORY", args.enable_timemory, args.disable_timemory)

if len(cmake_args) > 0:
    print("\n\n\tCMake arguments set via command line: {}\n".format(cmake_args))

setup(
    name='tomopy',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='Tomographic Reconstruction in Python.',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://tomopy.readthedocs.org',
    download_url='http://github.com/tomopy/tomopy.git',
    license='BSD-3',
    cmake_args=cmake_args,
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
