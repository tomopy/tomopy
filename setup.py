#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, Extension, find_packages

tomoc = Extension(
    name='lib.libtomopy',
    extra_compile_args=['-std=c99'],
    sources=[
        'src/corr.c',
        'src/utils.c',
        'src/simulate.c',
        'src/gridrec.c',
        'src/fft.c',
        'src/art.c',
        'src/bart.c',
        'src/fbp.c',
        'src/mlem.c',
        'src/osem.c',
        'src/ospml_hybrid.c',
        'src/ospml_quad.c',
        'src/pml_hybrid.c',
        'src/pml_quad.c',
        'src/sirt.c',
        'src/morph.c'])

setup(
    name='tomopy',
    packages=find_packages(),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    ext_modules=[tomoc],
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='Tomographic Reconstruction in Python.',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://tomopy.readthedocs.org',
    download_url='http://github.com/dgursoy/tomopy.git',
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
