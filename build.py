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

"""
Configures system-specifc settings and runs `make` to build libtomopy on the
current system.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from os.path import join as pjoin
import logging
import sys
import subprocess
import glob
import shutil
import time
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if 'CONDA_PREFIX' in os.environ:
    PREFIX = os.environ['CONDA_PREFIX']
else:
    PREFIX = sys.prefix

def get_makefile():
    """Get the Makefile for the system"""
    if sys.platform.lower().startswith('win'):
        return 'Makefile.windows'
    elif sys.platform == 'darwin':
        return 'Makefile.darwin'
    else:
        return 'Makefile.linux'

def get_config(install_prefix):
    if sys.platform.lower().startswith('win'):
        conf = config_windows(install_prefix)
    elif sys.platform == 'darwin':
        conf = config_macosx(install_prefix)
    else:
        conf = config_linux(install_prefix)
    return conf


def build_libtomopy(install_prefix='.'):
    """Build libtomopy shared library for the current system.

    This does the following steps:
      1. write `Mk.config` for current os
      2. run `make -j4` for the the current os
    """
    install_prefix = os.path.abspath(install_prefix)

    conf = get_config(install_prefix)
    logger.info("Config output:\n" + conf.format())
    with open('Mk.config', 'w') as fout:
        fout.write(conf.format())
    cmd = ['make', '-j4', '-f', get_makefile()]
    subprocess.check_call(tuple(cmd))

    src  = os.path.abspath(os.path.join("..", 'tomopy', 'sharedlibs', conf.sharedlib))
    dest = os.path.join(install_prefix, 'tomopy', 'sharedlibs', conf.sharedlib)
    shutil.copy(src, dest)
    os.chmod(dest, 493) # chmod 755

def clean_libtomopy(install_prefix='.'):
    """Clean libtomopy shared library for the current system."""
    install_prefix = os.path.abspath(install_prefix)
    conf = get_config(install_prefix)
    dylib = os.path.abspath(os.path.join("..", 'tomopy', 'sharedlibs', conf.sharedlib))
    clean_files = [dylib]
    for pattern in ('*.o', '*.gcda', '*.gcno', '*.gcov'):
        clean_files.extend(glob.glob(pattern))

    for fname in clean_files:
        try:
            os.unlink(fname)
        except OSError:
            logger.info("could not clean %s" % fname)

class Config:
    """A string formatter for the Makefile"""
    def __init__(self, install_prefix):
        self.compilerdir = 'gcc'
        self.sharedlib = ''
        self.arch_target = ''
        self.conda_compat = ''
        self.install_prefix = install_prefix
        self.includes = [pjoin(os.path.dirname(os.getcwd()), 'include')]
        self.linklibs = ['%s' % pjoin(PREFIX, 'lib')]
        # anaconda compat?
        if 'conda' in sys.version:
            compat = pjoin(PREFIX, 'compiler_compat')
            if os.path.exists(compat) and os.path.isdir(compat):
                self.conda_compat = '-B %s' % compat
        if 'GCC' in os.environ:
            self.compilerdir = os.environ["GCC"]
        # CC environment variable is standard, not GCC
        if 'CC' in os.environ:
            self.compilerdir = os.environ["CC"]
            print("### Compiler set via CC environment variable: '{}'".format(
                self.compilerdir))
        # includes
        top_include = pjoin(PREFIX, 'include')
        includes = [top_include]
        for fname in os.listdir(top_include):
            tdir = pjoin(top_include, fname)
            if os.path.isdir(tdir) and 'python' in fname:
                includes.append(tdir)
        self.includes += includes

    def format(self):
        """Return the formatted string, replacing Windows \\ with Unix /."""
        include = ' '.join(['-I%s' % s for s in self.includes])
        linklib = ' '.join(['-L%s' % s for s in self.linklibs])
        buff = ['## generated by build.py  %s' % time.ctime(),
                'COMPILER_DIR   = %s' % self.compilerdir,
                'SHAREDLIB      = %s' % self.sharedlib,
                'ARCH_TARGET    = %s' % self.arch_target,
                'LINK_LIB       = %s' % linklib,
                'INCLUDE        = %s' % include,
                'CONDA_COMPAT   = %s' % self.conda_compat,
                'INSTALL_PREFIX = %s' % self.install_prefix,
                '####', '']
        return '\n'.join(buff).replace('\\', '/')


def config_linux(install_prefix):
    """config for Linux"""
    logger.info("Config for Linux")
    config = Config(install_prefix)
    config.sharedlib = 'libtomopy.so'
    return config


def config_macosx(install_prefix):
    """config for MacOSX"""
    logger.info("Config for MacOS")
    config = Config(install_prefix)
    config.sharedlib = 'libtomopy.dylib'
    config.arch_target = '-arch x86_64'
    return config


def config_windows(install_prefix):
    """config for Windows"""
    logger.info("Config for Windows")
    config = Config(install_prefix)
    compilerdir = None
    if 'conda' in sys.version:
        # Look for GCC in the conda directory
        mingw_path = pjoin(PREFIX, 'MinGW', 'bin')
        mingw_gcc = pjoin(mingw_path, 'gcc.exe')
        if os.path.exists(mingw_gcc):
            logger.info("COMPILER_DIR is {}".format(mingw_gcc))
            compilerdir = mingw_path
        else:
            logger.warn("Compiler not found at {}".format(mingw_gcc))
    if compilerdir is None:
        for pdir in os.environ['PATH'].split(';'):
            gcc = pjoin(pdir, 'gcc.exe')
            if os.path.exists(gcc):
                compilerdir = pdir
                logger.info("COMPILER_DIR is {}".format(pdir))
                break
    if compilerdir is not None:
        config.compilerdir = compilerdir
    config.sharedlib = 'libtomopy.dll'
    config.includes += [pjoin(PREFIX, 'Library', 'include')]
    config.linklibs = [PREFIX,
                       pjoin(PREFIX, 'Library', 'bin'),
                       os.path.dirname(os.path.dirname(PREFIX)),
                       "C:/Windows/System32",
                       ]
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clean",
                        help="Clean build of TomoPy",
                        action='store_true')
    parser.add_argument("-p", "--prefix",
                        help="Install prefix",
                        type=str, default=None)
    args = parser.parse_args()

    curpath = os.getcwd()
    os.chdir('config')
    if args.prefix is None:
        args.prefix = '.'

    if args.clean:
        clean_libtomopy(install_prefix=args.prefix)
    else:
        build_libtomopy(install_prefix=args.prefix)
    os.chdir(curpath)
