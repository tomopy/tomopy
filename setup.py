#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io
import platform
import warnings

from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension, find_packages

#myplatform = platform.uname()[0]

# Check Python packages.
try:
    import numpy
except ImportError:
    raise ImportError("tomopy requires numpy 1.8.0 " +
                  "(hint: pip install numpy)")
try:
    import scipy
except ImportError:
    raise ImportError("tomopy requires scipy 0.13.2 " +
                  "(hint: pip install scipy)")
try:
    import h5py
except ImportError:
    raise ImportError("tomopy requires h5py 2.2.1 " +
                  "(hint: pip install h5py)")
try:
    from scipy.misc import toimage
except ImportError:
    raise ImportError("tomopy requires pillow 2.3.0 " +
                  "(hint: pip install pillow)")
try:
    import pywt
except ImportError:
    raise ImportError("tomopy requires pywavelets 0.2.2 " +
                  "(hint: pip install pywavelets)")

# Get shared library location from environment variables.
try:
    LD_LIBRARY_PATH = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
except KeyError:
    LD_LIBRARY_PATH = []
    warnings.warn("you may need to manually set LD_LIBRARY_PATH to " +
                  "link the shared libraries correctly")

try:
    C_INCLUDE_PATH = os.environ['C_INCLUDE_PATH'].split(os.pathsep)
except KeyError:
    C_INCLUDE_PATH = []
    warnings.warn("you may need to manually set C_INCLUDE_PATH to " +
                  "link the shared libraries correctly")

C_INCLUDE_PATH += {os.path.abspath('tomopy/recon/gridrec')}


# Create FFTW shared-library.
ext_fftw = Extension(name='tomopy.lib.libfftw',
                    sources=['tomopy/tools/fftw.cpp'],
                    include_dirs=C_INCLUDE_PATH,
                    library_dirs=LD_LIBRARY_PATH,
                    extra_link_args=['-lfftw3f'])
                    
# Create preprocessing shared-library.
ext_prep = Extension(name='tomopy.lib.libprep',
                    sources=['tomopy/preprocess/correct_drift.c',
                             'tomopy/preprocess/apply_padding.c',
                             'tomopy/preprocess/downsample.c'],
                    include_dirs=C_INCLUDE_PATH)

# Create reconstruction shared-library.
ext_recon = Extension(name='tomopy.lib.librecon',
                    sources=['tomopy/recon/art.c',
                             'tomopy/recon/mlem.c',
                             'tomopy/recon/upsample.c',
                             'tomopy/recon/gridrec/filters.cpp',
                             'tomopy/recon/gridrec/grid.cpp',
                             'tomopy/recon/gridrec/MessageQueue.cpp',
                             'tomopy/recon/gridrec/pswf.cpp',
                             'tomopy/recon/gridrec/tomoRecon.cpp',
                             'tomopy/recon/gridrec/tomoReconPy.cpp'],
                    include_dirs=C_INCLUDE_PATH,
                    library_dirs=LD_LIBRARY_PATH,
                    extra_link_args=['-lfftw3f',
                                     '-lboost_thread',
                                     '-lboost_system',
                                     '-lboost_date_time'])

# Main setup configuration.
setup(
      name='tomopy',
      version='0.0.1',

      packages = find_packages(),
      include_package_data = True,

      ext_modules=[ext_fftw, ext_recon, ext_prep],

      author='Doga Gursoy',
      author_email='dgursoy@aps.anl.gov',

      description='Toolbox for synchrotron tomographic imaging',
      keywords=['tomography', 'reconstruction', 'imaging'],
      url='http://aps.anl.gov/tomopy',
      download_url='http://github.com/tomopy/tomopy',

      license='BSD',
      platforms='Any',

      classifiers=['Development Status :: 4 - Beta',
		   'License :: OSI Approved :: BSD License',
		   'Intended Audience :: Science/Research',
		   'Intended Audience :: Education',
		   'Intended Audience :: Developers',
		   'Natural Language :: English',
		   'Operating System :: OS Independent',
		   'Programming Language :: Python',
		   'Programming Language :: Python :: 2.6',
		   'Programming Language :: Python :: 2.7',
		   'Programming Language :: C',
		   'Programming Language :: C++']
      )
