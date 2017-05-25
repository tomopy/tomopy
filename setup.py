#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages, os

# Get shared library locations.
LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', None)
if LD_LIBRARY_PATH is None:
    LD_LIBRARY_PATH = []
else:
    LD_LIBRARY_PATH = LD_LIBRARY_PATH.strip(':').split(':')

# Get header file locations.
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', None)
if C_INCLUDE_PATH is None:
    C_INCLUDE_PATH = []
else:
    C_INCLUDE_PATH = C_INCLUDE_PATH.split(':')

extra_comp_args = ['-std=c99']
extra_link_args = ['-lm']
if os.name == 'nt':
    import sys
    if sys.version_info.major == 3:
        extra_comp_args += ['-DPY3K']
    extra_comp_args += ['-DWIN32']
    extra_link_args += ['-lfftw3f-3']
else:
    extra_link_args += ['-lfftw3f']

tomoc = Extension(
    name='tomopy.libtomopy',
    extra_compile_args=extra_comp_args,
    extra_link_args=extra_link_args,
    library_dirs=LD_LIBRARY_PATH,
    include_dirs=C_INCLUDE_PATH,
    sources=[
        'src/utils.c',
        'src/project.c',
        'src/gridrec.c',
        'src/art.c',
        'src/bart.c',
        'src/fbp.c',
        'src/mlem.c',
        'src/osem.c',
        'src/ospml_hybrid.c',
        'src/ospml_quad.c',
        'src/pml_hybrid.c',
        'src/pml_quad.c',
        'src/prep.c',
        'src/sirt.c',
        'src/morph.c',
        'src/stripe.c',
        'src/remove_ring.c'])

ext_mods = [tomoc]

# Remove external C code for RTD builds
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    ext_mods = []

setup(
    name='tomopy',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    ext_modules=ext_mods,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='Tomographic Reconstruction in Python.',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://tomopy.readthedocs.org',
    download_url='http://github.com/tomopy/tomopy.git',
    license='BSD-3',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: C']
)
