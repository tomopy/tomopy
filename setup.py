#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io

from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, Extension, find_packages

# Check Python packages.
try:
    import numpy
except ImportError:
    raise ImportError("tomopy requires numpy 1.8.0 (hint: pip install numpy)")
try:
    import scipy
except ImportError:
    raise ImportError("tomopy requires scipy 0.13.2 (hint: pip install numpy)")
try:
    import h5py
except ImportError:
    raise ImportError("tomopy requires h5py 2.2.1 (hint: pip install h5py)")
try:
    from scipy.misc import toimage
except ImportError:
    raise ImportError("tomopy requires pillow 2.3.0 (hint: pip install pillow)")
try:
    import pywt
except ImportError:
    raise ImportError("tomopy requires pywavelets 0.2.2 (hint: pip install pywavelets)")

# Get shared library location from environment variables.
try:
    LD_LIBRARY_PATH = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
except KeyError:
    raise KeyError("You need to set LD_LIBRARY_PATH")
    
try:
    C_INCLUDE_PATH = os.environ['C_INCLUDE_PATH'].split(os.pathsep)
    C_INCLUDE_PATH += {os.path.abspath('tomopy/c/gridrec/include')}
    C_INCLUDE_PATH += {os.path.abspath('tomopy/c/fftw/include')}
except KeyError:
    raise KeyError("You need to set C_INCLUDE_PATH")

# Create FFTW shared-libraries.
ext_fftw = Extension(name='tomopy.lib.libfftw',
		     sources=['tomopy/c/fftw/src/fftw.cpp'],
		     include_dirs=C_INCLUDE_PATH,
		     library_dirs=LD_LIBRARY_PATH,
		     extra_link_args=['-lfftw3f'])

# Create Gridrec shared-libraries.
ext_gridrec = Extension(name='tomopy.lib.libgridrec',
			sources=['tomopy/c/gridrec/src/filters.cpp',
				 'tomopy/c/gridrec/src/grid.cpp',
				 'tomopy/c/gridrec/src/MessageQueue.cpp',
				 'tomopy/c/gridrec/src/pswf.cpp',
				 'tomopy/c/gridrec/src/tomoRecon.cpp',
				 'tomopy/c/gridrec/src/tomoReconPy.cpp'],
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
      
      # Specify C/C++ file paths. They will be compiled and put into tomopy.lib:
      ext_modules=[ext_fftw, ext_gridrec],
      
      author='Doga Gursoy',
      author_email='dgursoy@aps.anl.gov',

      description='Imaging toolbox',
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

