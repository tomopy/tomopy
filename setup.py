#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import io

VERSION = '0.0.3'

# Import setuptools.
from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages, Extension

# Read stuff in README and CHANGES.
def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)
long_description = read('README.md', 'CHANGES.txt')

# Allow setup.py to be run from any path.
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

# Create FFTW shared-libraries.
name = 'tomopy.lib.libfftw'
sources = ['tomopy/c/fftw/src/fftw.cpp']
libraries = []
include_dirs = []
define_macros = []
library_dirs = []
extra_compile_args = []
extra_link_args = ['-lfftw3f']
ext_module_fftw = Extension(name=name,
                            sources=sources,
                            libraries=libraries,
                            include_dirs=include_dirs,
                            define_macros=define_macros,
                            library_dirs=library_dirs,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args)

# Create Gridrec shared-libraries.
name = 'tomopy.lib.libgridrec'
sources = ['tomopy/c/gridrec/src/filters.cpp',
           'tomopy/c/gridrec/src/grid.cpp',
           'tomopy/c/gridrec/src/MessageQueue.cpp',
           'tomopy/c/gridrec/src/pswf.cpp',
           'tomopy/c/gridrec/src/tomoRecon.cpp',
           'tomopy/c/gridrec/src/tomoReconPy.cpp']
libraries = []
include_dirs = ['tomopy/c/gridrec/include']
define_macros = []
library_dirs = []
extra_compile_args = []
extra_link_args = ['-lfftw3f',
                   '-lboost_thread-mt',
                   '-lboost_date_time']
ext_module_gridrec = Extension(name=name,
                            sources=sources,
                            libraries=libraries,
                            include_dirs=include_dirs,
                            define_macros=define_macros,
                            library_dirs=library_dirs,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args)

# Main setup configuration.
setup(
      name = 'tomopy',
      version = VERSION,
      packages = find_packages(),
      install_requires = ['h5py==2.2.1',
                          'pyWavelets==0.2.2',
                          'scipy==0.13.2',
                          'numpy==1.8.0'],

      package_data = {
          # If any package contains *.txt or *.rst files, include them:
          '': ['*.txt', '*.rst'],
      },
      
      # Specify C/C++ file paths. They will be compiled and put into tomopy.lib:
      ext_modules = [ext_module_fftw, ext_module_gridrec],
      
      # metadata for upload to PyPI
      author = 'Doga Gursoy',
      author_email = 'dgursoy@aps.anl.gov',
      description = 'Imaging toolbox',
      long_description = long_description,
      license = 'BSD',
      keywords = 'tomography reconstruction imaging',
      url = 'http://aps.anl.gov/tomopy',
      download_url = 'http://github.com/dgursoy/tomopy-test',
      platforms = 'x86, x86_64',
      classifiers = [
                     'Development Status :: 4 - Beta',
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
                     'Programming Language :: C++',
                     ]
      )
