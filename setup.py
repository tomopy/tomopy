#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import shutil
import subprocess as sp
from setuptools import setup, find_packages, Extension
from distutils.command.clean import clean
from distutils.command.build_ext import build_ext


class TomoPyBuild(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        sp.check_call((sys.executable, 'build.py',
            '--prefix={}'.format(extdir)))


class TomoPyClean(clean):

    def run(self):
        sp.check_call((sys.executable, 'build.py', '--clean'))
        for d in [ "build", "tomopy.egg-info", "dist" ]:
            d = os.path.join(os.getcwd(), d)
            shutil.rmtree(d, ignore_errors=True)


setup(
    name='tomopy',
    packages=find_packages(exclude=['test*']),
    version=open('VERSION').read().strip(),
    include_package_data=True,
    zip_safe=False,
    author='Doga Gursoy',
    author_email='dgursoy@aps.anl.gov',
    description='Tomographic Reconstruction in Python.',
    keywords=['tomography', 'reconstruction', 'imaging'],
    url='http://tomopy.readthedocs.org',
    download_url='http://github.com/tomopy/tomopy.git',
    license='BSD-3',
    platforms='Any',
    # add extension module
    ext_modules=[Extension('libtomopy', sources=[])],
    # add custom build_ext command
    cmdclass=dict(build_ext=TomoPyBuild,clean=TomoPyClean),
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
