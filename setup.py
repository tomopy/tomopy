#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext

# ---------------------------------------------------------------------------- #
#
class TomoPyExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


# ---------------------------------------------------------------------------- #
#
class TomoPyBuild(build_ext, Command):

    #--------------------------------------------------------------------------#
    # run
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    #--------------------------------------------------------------------------#
    # build extension
    def build_extension(self, ext):
        import build as builder
        curpath = os.getcwd()
        os.chdir('config')
        builder.build_libtomopy()
        os.chdir(curpath)


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
    ext_modules=[TomoPyExtension('libtomopy')],
    # add custom build_ext command
    cmdclass=dict(build_ext=TomoPyBuild),
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
