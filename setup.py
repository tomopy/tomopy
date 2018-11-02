#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from setuptools import setup, find_packages, Extension, Command
from setuptools.command.build_ext import build_ext


# ---------------------------------------------------------------------------- #
#
def join_path_list(path_list):
    path = path_list[0]
    path_list.remove(path)
    for _path in path_list:
        path = os.path.join(path, _path)
    return path


# ---------------------------------------------------------------------------- #
#
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

    path = join_path_list([os.getcwd(), "tomopy", "sharedlibs", libname])
    if os.path.exists(path):
        return path
    else:
        path = join_path_list([os.getcwd(), "tomopy", "sharedlibs", "libtomopy"])
        for ext in ["so", "dylib", "dll"]:
            tmp = "{}.{}".format(path, ext)
            if os.path.exists(tmp):
                return tmp

    # return best guess
    return join_path_list([os.getcwd(), "tomopy", "sharedlibs", libname])


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
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        src = TomoPyLibraryFile()
        src_base = os.path.basename(src)
        dst = join_path_list([extdir, "tomopy", "sharedlibs", src_base])
        print("")
        print('Library: {}'.format(src))
        print('Destination: {}'.format(dst))
        print("")
        try:
            import shutil
            shutil.copy2(src, dst)
        except Exception as e:
            print("Exception occurred copying '{}' to '{}'... {}".format(src, dst, e))


# ---------------------------------------------------------------------------- #
#
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
