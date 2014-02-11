# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import platform
import tempfile
import sys
import tarfile
import hashlib
import shlex
from distutils import version
import subprocess
try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen
try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode

INSTALL_HDF5 = False
INSTALL_FFTW = False
INSTALL_BOOST = False

VERBOSE = False

# Default compilers
fc = 'gfortran'
cc = 'gcc'
cxx = 'g++'

prefix = None


def usage():
    print("")
    print(" Usage: python install.py [options] installation_path")
    print("")
    print(" Available options include:")
    print("")
    print("  --fftw                  compile the FFTW library")
    print("  --boost                 compile the Boost library")
    print("  --hdf5                  compile the HDF5 library")
    print("  --fc-compiler=<value>   force script to use a specific fortran compiler")
    print("  --cc-compiler=<value>   force script to use a specific C compiler")
    print("  --cxx-compiler=<value>  force script to use a specific C++ compiler")
    print("  --verbose               output standard output to terminal")
    print("  --help                  display this message")
    print("")
    sys.exit(1)

if '--help' in sys.argv[1:]:
    usage()

for arg in sys.argv[1:]:
    if arg == '--fftw':
        INSTALL_FFTW = True
    if arg == '--boost':
        INSTALL_BOOST = True
    if arg == '--hdf5':
        INSTALL_HDF5 = True
    if arg == '--verbose':
        VERBOSE = True
    if arg.startswith('--fc-compiler'):
        if '=' in arg:
            fc = arg.split('=')[1].replace('"', '').replace("'", "")
        else:
            usage()
    if arg.startswith('--cc-compiler'):
        if '=' in arg:
            cc = arg.split('=')[1].replace('"', '').replace("'", "")
        else:
            usage()
    if arg.startswith('--cxx-compiler'):
        if '=' in arg:
            cxx = arg.split('=')[1].replace('"', '').replace("'", "")
        else:
            usage()
    if not arg.startswith('-'):
        if prefix is None:
            prefix = arg
        else:
            print("ERROR: only one non-optional argument can be provided (the installation path)")
            usage()


if (INSTALL_HDF5 or INSTALL_FFTW or INSTALL_BOOST) is False:
    print("Nothing to install! Use 'python install.py --help' for options.")
    sys.exit(1)

if prefix is None:
    prefix = '/usr/local'
    print("Default installation directory is taken as /usr/local")
    
def run(command, logfile):
    import subprocess
    if VERBOSE:
        status = subprocess.call(command + ' 2>&1 | tee ' + logfile, shell=True, executable="/bin/bash")
    else:
        status = subprocess.call(command + ' >& ' + logfile, shell=True, executable="/bin/bash")
    if status != 0:
        print("Installation failed.")
        sys.exit(1)
    else:
        return


print(" ")
print("Determining system setup")
start_dir = os.path.abspath('.')
prefix = os.path.abspath(prefix)
print(" -> changing to work directory... ")
work_dir = tempfile.mkdtemp()
os.chdir(work_dir)
print("    %s" % work_dir)
print(" -> determining platform... ", end=' ')
# Determine system
system = platform.uname()[0]
# Determine system version
if system == 'Darwin':
    # Determine MacOS X version
    system_version = "MacOS " + platform.mac_ver()[0]
else:
    system_version = ''
print(system, '/', system_version)

# The following section deals with issues that occur when using the intel
# fortran compiler with gcc.
# Check whether C compiler is gcc
p = subprocess.Popen(shlex.split(cc + ' --version'), stdout=subprocess.PIPE)
output = (p.communicate()[0]).decode('ascii').strip().splitlines()[0]
is_gcc = 'GCC' in output

# Check whether Fortran compiler is ifort
p = subprocess.Popen(shlex.split(fc + ' --version'), stdout=subprocess.PIPE)
output = (p.communicate()[0]).decode('ascii').strip().splitlines()[0]
is_ifort = '(IFORT)' in output

# Check whether Fortran compiler is gfortran
p = subprocess.Popen(shlex.split(fc + ' --version'), stdout=subprocess.PIPE)
output = (p.communicate()[0]).decode('ascii').strip().splitlines()[0]
is_gfortran = 'GNU Fortran' in output
if is_gfortran:
    p = subprocess.Popen(shlex.split(fc + ' -dumpversion'), stdout=subprocess.PIPE)
    gfortran_version = version.LooseVersion((p.communicate()[0]).decode('ascii').splitlines()[0].split()[-1])

# Check whether Fortran compiler is g95
p = subprocess.Popen(shlex.split(fc + ' --version'), stdout=subprocess.PIPE)
output = (p.communicate()[0]).decode('ascii').strip().splitlines()[0]
is_g95 = 'g95' in output

# Check whether Fortran compiler is pgfortran
p = subprocess.Popen(shlex.split(fc + ' --version'), stdout=subprocess.PIPE)
output = (p.communicate()[0]).decode('ascii').strip().splitlines()[0]
is_pgfortran = 'pgfortran' in output or \
               'pgf95' in output

# On MacOS X, when using gcc 4.5 or 4.6, the fortran compiler needs to link
# with libgcc_eh that is in the gcc library directory. This is not needed if
# using gfortran 4.5 or 4.6, but it's easier to just add it for all
# compilers.
if system == 'Darwin' and is_gcc:
    # Determine gcc version
    p = subprocess.Popen(shlex.split(cc + ' -dumpversion'), stdout=subprocess.PIPE)
    gcc_version = version.LooseVersion((p.communicate()[0]).decode('ascii').splitlines()[0])
    if gcc_version >= version.LooseVersion('4.5.0'):
        p = subprocess.Popen(shlex.split(cc + ' -print-search-dirs'),
                             stdout=subprocess.PIPE)
        output = (p.communicate()[0]).decode('ascii').splitlines()[0]
        if output.startswith('install:'):
            libs = output.split(':', 1)[1].strip()
            libs = ' -L' + libs.replace(':', '-L')
            libs += ' -lgcc_eh'
            fc += libs
            print(" -> SPECIAL CASE: adjusting fortran compiler:", fc)
        else:
            print("ERROR: unexpected output for %s -print-search-dirs: %s" % (cc, output))
            sys.exit(1)

    # Check whether the C compiler give different architecture builds by default
    open('test_arch.c', 'w').write(TEST_ARCH_C)
    subprocess.Popen(shlex.split(cc + ' test_arch.c -o test_arch_c')).wait()
    p = subprocess.Popen(shlex.split('file test_arch_c'), stdout=subprocess.PIPE)
    output = (p.communicate()[0]).decode('ascii').splitlines()[0].strip()
    if output == 'test_arch_c: Mach-O 64-bit executable x86_64':
        arch_c = 64
    elif output == 'test_arch_c: Mach-O executable i386':
        arch_c = 32
    else:
        arch_c = None
    
    # Check whether the Fotran compiler give different architecture builds by default
    open('test_arch.f90', 'w').write(TEST_ARCH_F90)
    subprocess.Popen(shlex.split(fc + ' test_arch.f90 -o test_arch_f90')).wait()
    p = subprocess.Popen(shlex.split('file test_arch_f90'), stdout=subprocess.PIPE)
    output = (p.communicate()[0]).decode('ascii').splitlines()[0].strip()
    if output == 'test_arch_f90: Mach-O 64-bit executable x86_64':
        arch_f90 = 64
    elif output == 'test_arch_f90: Mach-O executable i386':
        arch_f90 = 32
    else:
        arch_f90 = None
    if arch_c is None or arch_f90 is None:
        pass  # just be safe and don't assume anything
    elif arch_c != arch_f90:
        if arch_c == 32:
            cc += '- m64'
            cxx += ' -m64'
            print(" -> SPECIAL CASE: adjusting C compiler:", cc)
        else:
            fc += ' -m64'
            print(" -> SPECIAL CASE: adjusting fortran compiler:", fc)


if INSTALL_FFTW:
    FFTW_URL = "http://www.fftw.org/fftw-3.3.3.tar.gz"
    FFTW_SHA1 = '11487180928d05746d431ebe7a176b52fe205cf9'
    print(" ")
    print("Installing FFTW")
    fftw_file = os.path.basename(FFTW_URL)
    if os.path.exists(fftw_file):
        sha1 = hashlib.sha1(open(fftw_file, 'rb').read()).hexdigest()
        if sha1 == FFTW_SHA1:
            print(" -> file exists, skipping download")
        else:
            print(" -> file exists but incorrect SHA1, re-downloading")
            open(fftw_file, 'wb').write(urlopen(FFTW_URL).read())
    else:
        print(" -> downloading")
        open(fftw_file, 'wb').write(urlopen(FFTW_URL).read())
    print(" -> expanding tarfile")
    t = tarfile.open(fftw_file, 'r:gz')
    t.extractall()
    print(" -> configuring")
    os.chdir(fftw_file.replace('.tar.gz', ''))
    run('./configure F77="{fc}" FC="{fc}" CC="{cc}" CXX="{cxx}" --enable-float --enable-shared --prefix={prefix}'.format(fc=fc, cc=cc, cxx=cxx, prefix=prefix), 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')
    os.chdir(work_dir)


if INSTALL_BOOST:
    BOOST_URL = "http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz"
    BOOST_SHA1 = '61ed0e57d3c7c8985805bb0682de3f4c65f4b6e5'
    print(" ")
    print("Installing Boost C++")
    boost_file = os.path.basename(BOOST_URL)
    if os.path.exists(boost_file):
        sha1 = hashlib.sha1(open(boost_file, 'rb').read()).hexdigest()
        if sha1 == BOOST_SHA1:
            print(" -> file exists, skipping download")
        else:
            print(" -> file exists but incorrect SHA1, re-downloading")
            open(boost_file, 'wb').write(urlopen(BOOST_URL).read())
    else:
        print(" -> downloading")
        open(boost_file, 'wb').write(urlopen(BOOST_URL).read())
    print(" -> expanding tarfile")
    t = tarfile.open(boost_file, 'r:gz')
    t.extractall()
    print(" -> configuring")
    os.chdir(boost_file.replace('.tar.gz', ''))
    run('./bootstrap.sh --with-libraries=system,thread,date_time --prefix={prefix}'.format(fc=fc, cc=cc, cxx=cxx, prefix=prefix), 'log_configure')
    print(" -> making")
    run('./b2', 'log_make')
    print(" -> installing")
    run('./b2 install', 'log_make_install')
    os.chdir(work_dir)


if INSTALL_HDF5:
    ZLIB_URL = "http://downloads.sourceforge.net/project/libpng/zlib/1.2.7/zlib-1.2.7.tar.gz"
    ZLIB_SHA1 = '4aa358a95d1e5774603e6fa149c926a80df43559'
    print(" ")
    print("Installing ZLIB (for HDF5)")
    zlib_file = os.path.basename(ZLIB_URL)
    if os.path.exists(zlib_file):
        sha1 = hashlib.sha1(open(zlib_file, 'rb').read()).hexdigest()
        if sha1 == ZLIB_SHA1:
            print(" -> file exists, skipping download")
        else:
            print(" -> file exists but incorrect SHA1, re-downloading")
            open(zlib_file, 'wb').write(urlopen(ZLIB_URL).read())
    else:
        print(" -> downloading")
        open(zlib_file, 'wb').write(urlopen(ZLIB_URL).read())
    print(" -> expanding tarfile")
    t = tarfile.open(zlib_file, 'r:gz')
    t.extractall()
    os.chdir(zlib_file.replace('.tar.gz', ''))
    print(" -> configuring")
    run('./configure --prefix={prefix}'.format(prefix=prefix), 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')
    os.chdir(work_dir)

    HDF5_URL = 'http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz'
    HDF5_SHA1 = '965d954d596cfa694f3260856a6406ea69e46e68'
    print(" ")
    print("Installing HDF5")
    hdf5_file = os.path.basename(HDF5_URL)
    if os.path.exists(hdf5_file):
        sha1 = hashlib.sha1(open(hdf5_file, 'rb').read()).hexdigest()
        if sha1 == HDF5_SHA1:
            print(" -> file exists, skipping download")
        else:
            print(" -> file exists but incorrect SHA1, re-downloading")
            open(hdf5_file, 'wb').write(urlopen(HDF5_URL).read())
    else:
        print(" -> downloading")
        open(hdf5_file, 'wb').write(urlopen(HDF5_URL).read())
    print(" -> expanding tarfile")
    t = tarfile.open(hdf5_file, 'r:gz')
    t.extractall()
    os.chdir(hdf5_file.replace('.tar.gz', ''))

    # SPECIAL CASE - g95 requires patching
    if is_g95:
        print(" -> SPECIAL CASE: patching for g95")
        conf = open('config/gnu-fflags', 'r').read()
        conf = conf.replace('-Wconversion -Wunderflow ', '')
        open('config/gnu-fflags', 'w').write(conf)

    # SPECIAL CASE - gfortran 4.5 and prior requires patching
    if is_gfortran and gfortran_version < version.LooseVersion('4.6.0'):
        print(" -> SPECIAL CASE: patching for gfortran 4.5 and prior")
        conf = open('fortran/src/H5test_kind_SIZEOF.f90', 'r').read()
        conf = conf.replace('DO i = 1,100', 'DO i = 1,18')
        open('fortran/src/H5test_kind_SIZEOF.f90', 'w').write(conf)

    print(" -> configuring")
    run('./configure FC="{fc}" --with-zlib={prefix}/include,{prefix}/lib --prefix={prefix}'.format(fc=fc, prefix=prefix), 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')
    os.chdir(work_dir)

# Go back to starting directory
os.chdir(start_dir)
print(" ")
print("Installation succesful!!!")
print(" ")
print("Before you start installing TomoPy, don't forget to:")
print(" ")
print("    1) Set LD_LIBRARY_PATH permanently in your shell to: %s" % prefix + '/lib')
print("       (hint: setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:%s" % prefix + '/lib')
print(" ")
print("    2) Set C_INCLUDE_PATH permanently in your shell to: %s" % prefix + '/include')
print("       (hint: setenv C_INCLUDE_PATH ${C_INCLUDE_PATH}:%s" % prefix + '/include')
print(" ")
