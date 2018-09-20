# configure system-specifc settings for tomopy Makefiles
#
# writes systems-dependent Mk.config file

import sys
import os
from os.path import join as pjoin
from os.path import exists as pexists
from os.path import isdir
import shutil
import time

class Config:
    def __init__(self):
        self.compilerdir = ''
        self.sharedlib = ''
        self.arch_target = ''
        self.link_lib = ''
        self.includes = []
        self.conda_compat = ''

        self.link_lib = '%s' % pjoin(sys.prefix, 'lib')

        # anaconda compat?
        if 'Anaconda' in sys.version:
            compat = pjoin(sys.prefix, 'compiler_compat')
            if pexists(compat) and isdir(compat):
                self.conda_compat = '-B %s' % compat

        # includes
        top_include = pjoin(sys.prefix, 'include')
        includes = [top_include]
        for fname in os.listdir(top_include):
            tdir = pjoin(top_include, fname)
            if isdir(tdir) and  'python' in fname:
                includes.append(tdir)

        self.includes = includes

    def format(self):
        include = ' '.join(['-I%s' % s for s in self.includes])
        buff = ['#### autogenerated %s' % time.ctime(),
                'COMPILER_DIR  = %s' % self.compilerdir,
                'SHAREDLIB     = %s' % self.sharedlib,
                'ARCH_TARGET   = %s' % self.arch_target,
                'LINK_LIB      = %s' % self.link_lib,
                'INCLUDE       = %s' % include,
                'CONDA_COMPAT  = %s' % self.conda_compat,
                '####', '']

        return '\n'.join(buff)


def config_linux():
    """ config for Linux"""
    config = Config()
    config.sharedlib = 'libtomopy.so'
    return config.format()

def config_macos():
    """ config for MacOS"""
    config = Config()
    config.sharedlib = 'libtomopy.dylib'
    config.arch_target = '-arch x86_64'
    return config.format()

def config_windows():
    """ config for Windows"""
    config = Config()
    compilerdir = None

    if 'Anaconda' in sys.version:
        mingw_path = pjoin(sys.prefix, 'MinGW', 'bin')
        mingw_gcc = pjoin(mingw_path, 'gcc.exe')
        if os.path.exists(mingw_gcc):
            compilerdir = mingw_path

    if compilerdir is None:
        for pdir in os.environ['PATH'].split(';'):
            gcc = pjoin(pdir, 'gcc.exe')
            if pexists(gcc):
                compilerdir = pdir
                break
    if compilerdir is not None:
        config.compilerdir = compilerdir            
    config.includes.append(pjoin(sys.prefix, 'Library', 'include'))
    config.sharedlib = 'libtomopy.dll'
    config.link_lib = '-L%s' % sys.prefix
    return config.format()

config = None
if sys.platform.lower().startswith('win'):
    shutil.copy('Makefile.windows', 'Makefile')
    config = config_windows
elif sys.platform == 'darwin':
    shutil.copy('Makefile.darwin', 'Makefile')
    config = config_macos
elif sys.platform == 'linux':
    shutil.copy('Makefile.linux', 'Makefile')
    config = config_linux

if config is not None:
    fout = open('Mk.config', 'w')
    fout.write( config() )
    fout.close()
    print("configure.py successful, ready for 'make install'")
