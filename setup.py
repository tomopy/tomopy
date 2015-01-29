#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings

from setuptools import setup, Extension, find_packages

# Set Python package requirements for installation.
install_requires = ['numpy>=1.8.0', 'scipy>=0.13.2', 'h5py>=2.2.1',
                    'pywavelets>=0.2.2', 'scikit-image>=0.9']

# enforce these same requirements at packaging time
import pkg_resources

for requirement in install_requires:
    try:
        pkg_resources.require(requirement)
    except pkg_resources.DistributionNotFound:
        msg = 'Python package requirement not satisfied: ' + requirement
        msg += '\n\nInstall all required packages with\n\tpip install -r requirements.txt'
        raise pkg_resources.DistributionNotFound, msg

# Get shared library locations (list of directories).
LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', None)
if LD_LIBRARY_PATH is None:
    warnings.warn("you may need to manually set LD_LIBRARY_PATH to " +
                  "link the shared libraries correctly")
    LD_LIBRARY_PATH = ''
#for windows split with ; , unix split with :
if os.name == 'nt':
    LD_LIBRARY_PATH = LD_LIBRARY_PATH.split(';')
else:
    LD_LIBRARY_PATH = LD_LIBRARY_PATH.split(':')

# Get header file locations (list of directories).
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', None)
if C_INCLUDE_PATH is None:
    warnings.warn("you may need to manually set C_INCLUDE_PATH to " +
                  "link the shared libraries correctly")
    C_INCLUDE_PATH = []
else:
    #for windows split with ; , unix split with :
    if os.name == 'nt':
        C_INCLUDE_PATH = C_INCLUDE_PATH.split(';')
    else:
        C_INCLUDE_PATH = C_INCLUDE_PATH.split(':')

# add ourselves to the list
C_INCLUDE_PATH += [os.path.abspath('tomopy/algorithms/recon/gridrec')]


#for windows we need to know the boost version
if os.name == 'nt':
    BOOST_VER = os.environ.get('BOOST_VER', None)
    if BOOST_VER is None:
        warnings.warn("you may need to manually set BOOST_VER to " +
                      "the compiled version. defaulting to 'mgw47-mt-1_57'")
        BOOST_VER = 'mgw47-mt-1_57'

if os.name == 'nt':
    fftw_extra_link_args = ['-lfftw3f-3']
    recon_extra_link_args = ['-lfftw3f-3', '-lboost_thread-'+BOOST_VER, '-lboost_system-'+BOOST_VER, '-lboost_date_time-'+BOOST_VER]
else:
    fftw_extra_link_args = ['-lfftw3f']
    recon_extra_link_args = ['-lfftw3f', '-lboost_thread', '-lboost_system', '-lboost_date_time']
	
# Create FFTW shared-library.
ext_fftw = Extension(name='tomopy.lib.libfftw',
                     sources=['tomopy/tools/fftw.cpp'],
                     include_dirs=C_INCLUDE_PATH,
                     library_dirs=LD_LIBRARY_PATH,
                     extra_link_args=fftw_extra_link_args)

# Create preprocessing shared-library.
ext_prep = Extension(name='tomopy.lib.libprep',
                     sources=['tomopy/algorithms/preprocess/correct_drift.c',
                              'tomopy/algorithms/preprocess/apply_padding.c',
                              'tomopy/algorithms/preprocess/downsample.c'],
                     include_dirs=C_INCLUDE_PATH)

# Create reconstruction shared-library.
ext_recon = Extension(name='tomopy.lib.librecon',
                      sources=['tomopy/algorithms/recon/art.c',
                               'tomopy/algorithms/recon/sirt.c',
                               'tomopy/algorithms/recon/mlem.c',
                               'tomopy/algorithms/recon/pml.c',
                               'tomopy/algorithms/recon/upsample.c',
                               'tomopy/algorithms/recon/gridrec/filters.cpp',
                               'tomopy/algorithms/recon/gridrec/grid.cpp',
                               'tomopy/algorithms/recon/gridrec/MessageQueue.cpp',
                               'tomopy/algorithms/recon/gridrec/pswf.cpp',
                               'tomopy/algorithms/recon/gridrec/tomoRecon.cpp',
                               'tomopy/algorithms/recon/gridrec/tomoReconPy.cpp'],
                      include_dirs=C_INCLUDE_PATH,
                      library_dirs=LD_LIBRARY_PATH,
                      extra_link_args=recon_extra_link_args)

ext_test = Extension(name='tomopy.lib.libtest',
                     sources=['tomopy/algorithms/recon/mlem.c'])

# Main setup configuration.
setup(
    name='tomopy',
    version=open('VERSION').read().strip(),
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[ext_fftw, ext_recon, ext_prep, ext_test],
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='Toolbox for synchrotron tomographic imaging',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://aps.anl.gov/tomopy',
    download_url='http://github.com/tomopy/tomopy',
    license='BSD',
    platforms='Any',
    install_requires=install_requires,
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
                 'Programming Language :: C++'])
