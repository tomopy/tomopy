#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, Extension, find_packages

# Get shared library locations.
LD_LIBRARY_PATH = os.environ.get('LD_LIBRARY_PATH', None)
if LD_LIBRARY_PATH is None:
    LD_LIBRARY_PATH = []
else:
    if os.name == 'nt':
        LD_LIBRARY_PATH = LD_LIBRARY_PATH.strip(';').split(';')
    else:
        LD_LIBRARY_PATH = LD_LIBRARY_PATH.strip(':').split(':')
# Get header file locations.
C_INCLUDE_PATH = os.environ.get('C_INCLUDE_PATH', None)
if C_INCLUDE_PATH is None:
    C_INCLUDE_PATH = []
else:
    if os.name == 'nt':
        C_INCLUDE_PATH = C_INCLUDE_PATH.split(';')
    else:
        C_INCLUDE_PATH = C_INCLUDE_PATH.split(':')

use_mkl = os.environ.get('DISABLE_MKL') is None

extra_link_args = ['-lm']
C_INCLUDE_PATH.append(os.path.join(sys.prefix, 'include'))
LD_LIBRARY_PATH.append(os.path.join(sys.prefix, 'Library'))
if os.name == 'nt':
    from distutils import cygwinccompiler
    def get_msvcr140_hack():
        return ['vcruntime140']
    if sys.version_info[0] == 3:
        cygwinccompiler.get_msvcr = get_msvcr140_hack

    extra_comp_args = []
    if sys.version_info.major == 3:
        extra_comp_args += ['-DPY3K']
    extra_comp_args += ['-DWIN32']
    extra_link_args += ['-lmkl_rt'] if use_mkl else ['-lfftw3f-3']
     
    C_INCLUDE_PATH.append(os.path.join(sys.prefix, 'Library', 'include'))
    LD_LIBRARY_PATH.append(os.path.join(sys.prefix, 'Library', 'bin'))

    # intel mkl .lib are not copied to %sys.prefix%\Library\lib. 
    # we need to add link paths to these in the package directory
    base = sys.prefix
    for envar in ('MINICONDA', 'PYTHONHOME'):
        if envar in os.environ:
            base = os.environ[envar]
    mkl_libdir = None
   
    for pkg in os.listdir(os.path.join(base, "pkgs")):
        fullname = os.path.join(base, 'pkgs', pkg)
        if os.path.isdir(fullname) and pkg.startswith("mkl-20"):
            _ldir = os.path.join(fullname, "Library", "lib")
            if os.path.isdir(_ldir) and 'mkl_rt.lib' in os.listdir(_ldir):
                 mkl_libdir = _ldir
			
    if use_mkl and mkl_libdir is not None:
        LD_LIBRARY_PATH.append(mkl_libdir)
		
else:
    extra_comp_args = ['-std=c99']
    extra_link_args += ['-lmkl_rt'] if use_mkl else ['-lfftw3f']

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
        'src/vector.c',
        'src/morph.c',
        'src/stripe.c',
        'src/remove_ring.c'],
    define_macros=[('USE_MKL', None)] if use_mkl else [])

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
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: C']
)
