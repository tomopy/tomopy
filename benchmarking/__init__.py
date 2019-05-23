#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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

"""
PyCTest driver functions
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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


__version__ = '1.3.0'
__all__ = [
    # variable section
    'default_nitr',
    'default_phantom_size',
    # find section
    'find_python_nosetest',
    'find_python_coverage',
    'find_ctest_token',
    # build section
    'build_option_append',
    'build_name_append',
    # create test section
    'create_correct_module_test',
    'create_nosetest_test',
    'create_coverage_test',
    'create_phantom_test',
    'create_globus_test'
]


########################################################################################
#
#                                   VARIABLE SECTION
#
########################################################################################


default_nitr = 10
""" if iterations is changed, test name will change """


default_phantom_size = 512
""" if phantom size is changed, test name will change """


########################################################################################
#
#                                   FIND SECTION
#
########################################################################################


def find_python_nosetest():
    """
    Find the python nosetests script
    """
    pypath = os.path.dirname(pyctest.PYTHON_EXECUTABLE)
    pyexe = helpers.FindExePath("nosetests", path=pypath)
    if pyexe is None:
        pyexe = helpers.FindExePath("nosetests")
    return pyexe


def find_python_coverage():
    """
    Find the python coverage script
    """
    pypath = os.path.dirname(pyctest.PYTHON_EXECUTABLE)
    pyexe = helpers.FindExePath("coverage", path=pypath)
    if pyexe is None:
        pyexe = helpers.FindExePath("coverage")
    return pyexe


def find_ctest_token():
    """
    Add authentification token if it exists
    """
    token_string = os.environ.get("CTEST_TOKEN")
    token_path = os.environ.get("CTEST_TOKEN_FILE")
    home = helpers.GetHomePath()
    if token_string is not None:
        return
    elif token_path is not None:
        pyctest.set("CTEST_TOKEN_FILE", token_path)
    elif home is not None:
        for fname in ["nersc-tomopy", "nersc-cdash"]:
            token_path = os.path.join(home, ".tokens", fname)
            if os.path.exists(token_path):
                pyctest.set("CTEST_TOKEN_FILE", token_path)
                return
    message = "Warning! No CTEST_TOKEN or CTEST_TOKEN_FILE set. Submission will fail"
    warnings.warn(message)


########################################################################################
#
#                                   COMMAND SECTION
#
########################################################################################


def build_option_append(check, var, val):
    pyctest.BUILD_COMMAND += " -D{}={}".format(var, val) if check else ""


def build_name_append(item, separate=True, prefix="[", suffix="]", check=True, reset=False):
    if reset:
        pyctest.BUILD_NAME = ""
    # if not list, check if is None
    if item is None or not item:
        return
    if check:
        pyctest.BUILD_NAME += " "
        item = item.strip()
        pyctest.BUILD_NAME += "{}".format(prefix)
        pyctest.BUILD_NAME += "{}".format(item)
        pyctest.BUILD_NAME += "{}".format(suffix) if separate else ""
        if not separate:
            pyctest.BUILD_NAME += "{}".format(suffix)
        pyctest.BUILD_NAME += " "
        pyctest.BUILD_NAME = pyctest.BUILD_NAME.strip()
        while "  " in pyctest.BUILD_NAME:
            pyctest.BUILD_NAME = pyctest.BUILD_NAME.replace("  ", " ")


########################################################################################
#
#                                   TEST SECTION
#
########################################################################################


def create_correct_module_test():
    """
    Create a test that checks we are using the locally build module
    """
    pyexe = pyctest.PYTHON_EXECUTABLE
    binary_dir = pyctest.BINARY_DIRECTORY

    # test properties
    props = {
        "WORKING_DIRECTORY": binary_dir,
        "RUN_SERIAL": "ON",
        "LABEL": "unit"
        }

    # test command
    cmd = [pyexe, "-c",
        "\"import os, sys, tomopy; " +
        "print('using tomopy module: {}'.format(tomopy.__file__)); " +
        "ret=0 if os.getcwd() in tomopy.__file__ else 1; " +
        "sys.exit(ret)\""]

    pyctest.test("correct_module", cmd, props)


def create_nosetest_test(args):
    """
    Create a test that runs nosetests
    """
    pyexe = pyctest.PYTHON_EXECUTABLE
    binary_dir = pyctest.BINARY_DIRECTORY
    pynosetest = find_python_nosetest()
    pycoverage = find_python_coverage()

    # test properties
    props = {
        "DEPENDS": "correct_module",
        "RUN_SERIAL": "ON",
        "LABEL": "unit",
        "WORKING_DIRECTORY": binary_dir,
        "ENVIRONMENT": "TOMOPY_USE_C_ALGORITHMS=1"
        }

    # test command: python $(which coverage) run $(which nosetest)
    cmd = [pyexe, pycoverage, "run", pynosetest]

    # create test
    pyctest.test("nosetests", cmd, props)


def create_coverage_test(args):
    """
    Create test that generates the coverage
    """
    source_dir = pyctest.SOURCE_DIRECTORY
    binary_dir = pyctest.BINARY_DIRECTORY
    pycoverage = find_python_coverage()

    # test properties
    props = {
        "WORKING_DIRECTORY": binary_dir,
        "DEPENDS": "nosetests",
        "RUN_SERIAL": "ON",
        "LABEL": "unit"
    }

    # test command
    cmd = [os.path.join(source_dir, ".coverage.sh"), source_dir]
    if platform.system() == "Windows":
        cmd = [pycoverage, "xml"]

    pyctest.test("coverage", cmd, props)


def create_phantom_test(args, bench_props, phantom):
    """
    Create test(s) for the specified algorithms
    """
    pyexe = pyctest.PYTHON_EXECUTABLE
    this_dir = os.path.dirname(__file__)

    # skip when generating C coverage
    if args.coverage or args.disable_phantom_tests:
        return

    nalgs = len(args.algorithms)
    psize = args.phantom_size

    # construct test name
    name = ((phantom + "_") +
            ("".join(args.algorithms) if nalgs == 1 else "comparison"))

    # determine phantom size
    nsize = psize if phantom != "shepp3d" else psize // 2

    # customize name
    if psize != default_phantom_size:
        name = "{}_pix{}".format(name, nsize)
    if args.num_iter != default_nitr:
        name = "{}_itr{}".format(name, args.num_iter)

    # test arguments
    test_args = [
        "-A", "360",
        "-f", "jpeg",
        "-S", "1",
        "-p", phantom,
        "-s", "{}".format(nsize),
        "-n", "{}".format(args.ncores),
        "-i", "{}".format(args.num_iter),
        "--output-dir", os.path.join(this_dir, name)
    ]
    test_args.append("-a" if nalgs == 1 else "--compare")
    test_args.extend(args.algorithms)

    # test properties
    test_props = bench_props
    if phantom.lower() == "shepp3d":
        test_props["RUN_SERIAL"] = "ON"

    # test command
    cmd = [pyexe, "-Om", "benchmarking.phantom"] + test_args

    # create test
    pyctest.test(name, cmd, properties=test_props)


def create_globus_test(args, bench_props, algorithm, phantom):
    """
    Create a test from TomoBank (data provided by globus)
    """
    # skip when generating C coverage when globus enabled
    if args.coverage or args.globus_path is None:
        return

    pyexe = pyctest.PYTHON_EXECUTABLE
    this_dir = os.path.dirname(__file__)

    name = "{}_{}".format(phantom, algorithm)
    # original number of iterations before num-iter added to test name
    if args.num_iter != 10:
        name = "{}_itr{}".format(name, args.num_iter)

    global_args = [
        "--type", "slice",
        "-f", "jpeg",
        "-S", "1",
        "-c", "4",
        "-n", "{}".format(args.ncores),
        "-i", "{}".format(args.num_iter)
    ]

    # try this path
    h5file = os.path.join(args.globus_path, phantom + ".h5")

    # alternative path
    if not os.path.exists(h5file):
        h5file = os.path.join(args.globus_path, phantom, phantom + ".h5")

    # could not locate
    if not os.path.exists(h5file):
        print("HDF5 file '{}' does not exist.".format(h5file))
        cmd = [pyexe, "-c",
               "print(\"No valid path to '{}'\")".format(h5file)]
        h5file = None
    else:
        cmd = [pyexe, "-Om", "benchmarking.rec", h5file]
        cmd += (global_args +
                ["-a", algorithm,
                "-o", os.path.join(this_dir, name)])

    pyctest.test(name, cmd, properties=bench_props)
