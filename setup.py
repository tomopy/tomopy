#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings

from setuptools import setup, Extension, find_packages

# Set Python package requirements for installation.
install_requires = [
            'numpy>=1.8.0',
            'scipy>=0.13.2',
            'h5py>=2.2.1',
            'pywavelets>=0.2.2'
            ]

# enforce these same requirements at packaging time
import pkg_resources
for requirement in install_requires:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        msg = 'Python package requirement not satisfied: ' + requirement
        msg += '\nsuggest using this command:'
        msg += '\n\tpip install -U ' + requirement.split('=')[0].rstrip('>')
        raise pkg_resources.DistributionNotFound, msg

# Get shared library locations (list of directories).
LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', None)
if LD_LIBRARY_PATH is None:
    warnings.warn("you may need to manually set LD_LIBRARY_PATH to " +
                  "link the shared libraries correctly")
    LD_LIBRARY_PATH = ''
LD_LIBRARY_PATH = LD_LIBRARY_PATH.split(':')

# Get header file locations (list of directories).
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', None)
if C_INCLUDE_PATH is None:
    warnings.warn("you may need to manually set C_INCLUDE_PATH to " +
                  "link the shared libraries correctly")
    C_INCLUDE_PATH = ''
C_INCLUDE_PATH = C_INCLUDE_PATH.split(':')

# add ourselves to the list
C_INCLUDE_PATH += [os.path.abspath('tomopy/algorithms/recon/gridrec')]


# Create FFTW shared-library.
ext_fftw = Extension(name='tomopy.lib.libfftw',
                    sources=['tomopy/tools/fftw.cpp'],
                    include_dirs=C_INCLUDE_PATH,
                    library_dirs=LD_LIBRARY_PATH,
                    extra_link_args=['-lfftw3f'])
                    
# Create preprocessing shared-library.
ext_prep = Extension(name='tomopy.lib.libprep',
                    sources=['tomopy/algorithms/preprocess/correct_drift.c',
                             'tomopy/algorithms/preprocess/apply_padding.c',
                             'tomopy/algorithms/preprocess/downsample.c'],
                    include_dirs=C_INCLUDE_PATH)

# Create reconstruction shared-library.
ext_recon = Extension(name='tomopy.lib.librecon',
                    sources=['tomopy/algorithms/recon/art.c',
                             'tomopy/algorithms/recon/mlem_emission.c',
                             'tomopy/algorithms/recon/mlem_transmission.c',
                             'tomopy/algorithms/recon/upsample.c',
                             'tomopy/algorithms/recon/gridrec/filters.cpp',
                             'tomopy/algorithms/recon/gridrec/grid.cpp',
                             'tomopy/algorithms/recon/gridrec/MessageQueue.cpp',
                             'tomopy/algorithms/recon/gridrec/pswf.cpp',
                             'tomopy/algorithms/recon/gridrec/tomoRecon.cpp',
                             'tomopy/algorithms/recon/gridrec/tomoReconPy.cpp'],
                    include_dirs=C_INCLUDE_PATH,
                    library_dirs=LD_LIBRARY_PATH,
                    extra_link_args=['-lfftw3f',
                                     '-lboost_thread',
                                     '-lboost_system',
                                     '-lboost_date_time'])

# Main setup configuration.
setup(
      name='tomopy',
      version=open('VERSION').read().strip(),

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
      install_requires = install_requires,

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
