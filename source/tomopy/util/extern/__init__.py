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
import ctypes.util
import os
import warnings


def c_shared_lib(lib_name, error=True):
    """Get the path and import the C-shared library.

    The ctypes.util.find_library function preprends "lib" to the name.
    """
    if os.name == 'nt':
        load_dll = ctypes.windll.LoadLibrary
    else:
        load_dll = ctypes.cdll.LoadLibrary

    # Returns None or a library name
    sharedlib = ctypes.util.find_library(lib_name)

    if sharedlib is not None:
        try:
            # No error if sharedlib is None; error if library name wrong
            return load_dll(sharedlib)
        except OSError:
            pass

    explanation = (
        'TomoPy links to compiled components which are installed separately'
        ' and loaded using ctypes.util.find_library().'
    )
    if error:
        raise ModuleNotFoundError(
            explanation +
            f' A required library, {lib_name}, was not found.')
    warnings.warn(
        explanation +
        'Some functionality is unavailable because an optional shared'
        f' library, {lib_name}, is missing.', ImportWarning)
    return None


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
