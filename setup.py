#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import shutil
import subprocess as sp
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext


def TomoPyLibraryFile():
    """
    Determines the library output file
    """

    libname = None
    if sys.platform.lower().startswith('win'):
        libname = 'libtomopy.dll'
    elif sys.platform == 'darwin':
        libname = 'libtomopy.dylib'
    else:
        libname = 'libtomopy.so'

    path = os.path.join(os.getcwd(), "tomopy", "sharedlibs", libname)
    if os.path.exists(path):
        return path
    else:
        paths = glob.glob(os.path.join(
            os.getcwd(), "tomopy", "sharedlibs", "libtomopy.*"))
        if len(paths) > 0:
            return paths[0]

    # return best guess
    return os.path.join(os.getcwd(), "tomopy", "sharedlibs", libname)


class TomoPyExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class TomoPyBuild(build_ext, Command):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        sp.check_call((sys.executable, 'build.py'))
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        src = TomoPyLibraryFile()
        src_base = os.path.basename(src)
        dst = os.path.join(extdir, "tomopy", "sharedlibs", src_base)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print("Exception occurred copying '{}' to '{}'...".format(src, dst))
            raise e


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
