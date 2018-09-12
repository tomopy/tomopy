#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

if os.name == 'nt':
    # this is needed to trick the mingw compiler to link against msvcr100 instead of a non-existent msvcr140.
    # when mingw will support UCRT, this code block will no longer be needed 
    import sys
    msc_pos = sys.version.find('MSC v.')
    if msc_pos != -1:
        msc_ver = sys.version[msc_pos+6:msc_pos+10]
        if int(msc_ver) >= 1900:
            sys.version = "".join([sys.version[:msc_pos+6], "1600",sys.version[msc_pos+10:]])


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

print(C_INCLUDE_PATH)
use_mkl = os.environ.get('DISABLE_MKL') is None



extra_link_args = ['-lm']
if os.name == 'nt':
    import sys
    extra_comp_args = []
    if sys.version_info.major == 3:
        extra_comp_args += ['-DPY3K']
    extra_comp_args += ['-DWIN32']
    extra_link_args += ['-lmkl_rt'] if use_mkl else ['-lfftw3f-3']

    # intel mkl .lib are not copied to %PYTHONHOME%\Library\lib. we need to add it to our link paths
    base = os.path.join(os.environ.get("PYTHONHOME", os.environ.get("MINICONDA")), "pkgs")
    last = [i for i in os.listdir(base) if i.startswith("mkl-20") and os.path.isdir(os.path.join(base, i))]
    LD_LIBRARY_PATH.append(os.path.join(base, last[-1], "Library", "lib"))

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
	'src/tv.c',
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
