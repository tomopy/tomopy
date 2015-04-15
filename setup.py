#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages

prep = Extension(
    name='tomopy.lib.libtomopy_prep',
    extra_compile_args=['-std=c99'],
    sources=[
        'tomopy/src/corr.c'])


recon = Extension(
    name='tomopy.lib.libtomopy_recon',
    extra_compile_args=['-std=c99'],
    sources=[
        'tomopy/src/utils.c',
        'tomopy/src/simulate.c',
        'tomopy/src/gridrec.c',
        'tomopy/src/fft.c',
        'tomopy/src/art.c',
        'tomopy/src/bart.c',
        'tomopy/src/fbp.c',
        'tomopy/src/mlem.c',
        'tomopy/src/osem.c',
        'tomopy/src/ospml_hybrid.c',
        'tomopy/src/ospml_quad.c',
        'tomopy/src/pml_hybrid.c',
        'tomopy/src/pml_quad.c',
        'tomopy/src/sirt.c'])


misc = Extension(
    name='tomopy.lib.libtomopy_misc',
    extra_compile_args=['-std=c99'],
    sources=[
        'tomopy/src/morph.c'])


setup(
    name='tomopy',
    packages=find_packages(),
    package_data={'tomopy': ['data/*.tif']},
    version=open('VERSION').read().strip(),
    ext_modules=[prep, recon, misc],
    include_package_data=True,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='X-ray imaging toolbox',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://tomopy.readthedocs.org',
    download_url='http://github.com/tomopy/tomopy.git',
    license='BSD',
    platforms='Any',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: C']
)
