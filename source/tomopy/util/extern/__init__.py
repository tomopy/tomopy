#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
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

import ctypes
import os
import sys
import warnings


def c_shared_lib(lib_name, error=True):
    """Get the path and import the C-shared library."""
    load_dll = ctypes.cdll.LoadLibrary
    ext = '.so'
    if sys.platform == 'darwin':
        ext = '.dylib'
    if os.name == 'nt':
        ext = '.dll'
        load_dll = ctypes.windll.LoadLibrary
    base_path = os.path.abspath(os.path.dirname(__file__))
    sharedlib = os.path.join(base_path, '%s%s' % (lib_name, ext))
    if os.path.exists(sharedlib):
        return load_dll(sharedlib)
    if error:
        raise ModuleNotFoundError(
            f'The following shared library is missing:\n{sharedlib}')
    warnings.warn(
        'Some compiled functions are unavailable because an optional shared'
        f' library is missing:\n{sharedlib}', ImportWarning)


def _missing_library(function):
    raise ModuleNotFoundError(
        f"The {function} algorithm is unavailable because its shared library"
        " is missing. Check CMake logs to determine if TomoPy was"
        " built with all dependencies required by this algorithm.")


from tomopy.util.extern.recon import *
from tomopy.util.extern.accel import *
from tomopy.util.extern.gridrec import *
from tomopy.util.extern.prep import *
from tomopy.util.extern.misc import *
