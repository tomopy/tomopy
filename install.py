# -*- coding: utf-8 -*-


'''install necessary external libraries for TomoPy'''


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
    from urllib import urlopen          #@UnusedImport
except ImportError:
    from urllib.request import urlopen  #@UnusedImport
try:
    from urllib import urlencode        #@UnusedImport
except ImportError:
    from urllib.parse import urlencode  #@UnusedImport


VERBOSE = False


def get_cmd_opts():
    import argparse
    doc = __doc__.strip()
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=doc)

    prefix = '/usr/local'
    msg = 'Installation directory (default: ' + prefix + ')'
    parser.add_argument('installation_path', 
                        action='store', 
                        nargs='?', 
                        help=msg,
                        default=prefix)

    parser.add_argument('--fftw', 
                        action='store_true',
                        dest='install_fftw',
                        help='install fftw library',
                        default=False)
    parser.add_argument('--boost', 
                        action='store_true',
                        dest='install_boost',
                        help='install Boost library',
                        default=False)
    parser.add_argument('--hdf5', 
                        action='store_true',
                        dest='install_hdf5',
                        help='install HDF5 library',
                        default=False)

    fc = 'gfortran'
    msg = 'specify the Fortran compiler (default: '+fc+')'
    parser.add_argument('--fc-compiler',
                        action='store', 
                        dest='fc_compiler', 
                        help=msg, 
                        default=fc)
    cc = 'gcc'
    msg = 'specify the C compiler (default: '+cc+')'
    parser.add_argument('--cc-compiler',
                        action='store', 
                        dest='cc_compiler', 
                        help=msg, 
                        default=cc)
    cxx = 'g++'
    msg = 'specify the C++ compiler (default: '+cxx+')'
    parser.add_argument('--cxx-compiler',
                        action='store', 
                        dest='cxx_compiler', 
                        help=msg, 
                        default=cxx)

    parser.add_argument('--verbose', 
                        action='store_true',
                        dest='verbose',
                        help='send standard output to terminal',
                        default=False)

    return parser.parse_args()
    

def run(command, logfile):
    def do_bash(command):
        return subprocess.call(command, 
                               shell=True, 
                               executable="/bin/bash")
    if VERBOSE:
        status = do_bash(command + ' 2>&1 | tee ' + logfile)
    else:
        status = do_bash(command + ' >& ' + logfile)
    if status != 0:
        print("Installation failed.")
        sys.exit(1)
    else:
        return


def p_communicate(shell_command):
    '''return the output of the shell shell_command'''
    p = subprocess.Popen(shlex.split(shell_command), stdout=subprocess.PIPE)
    return (p.communicate()[0]).decode('ascii')


def p_get_first_line(shell_command):
    '''return the first line of output from the shell command'''
    return p_communicate(shell_command).strip().splitlines()[0]


def download_expand_tarball(name, URL, SHA1):
    print(" ")
    print("Installing " + name)
    ext_file = os.path.basename(URL)
    download_it = True
    if os.path.exists(ext_file):
        sha1 = hashlib.sha1(open(ext_file, 'rb').read()).hexdigest()
        download_it = sha1 != SHA1
        if download_it:
            print(" -> file exists but incorrect SHA1, re-downloading")
        else:
            print(" -> file exists, skipping download")
    if download_it:
        print(" -> downloading")
        open(ext_file, 'wb').write(urlopen(URL).read())
    print(" -> expanding tarfile")
    t = tarfile.open(ext_file, 'r:gz')
    t.extractall()
    os.chdir(ext_file.replace('.tar.gz', ''))


def install_fftw(prefix, fc, cc, cxx):
    FFTW_URL = "http://www.fftw.org/fftw-3.3.3.tar.gz"
    FFTW_SHA1 = '11487180928d05746d431ebe7a176b52fe205cf9'
    download_expand_tarball('FFTW', FFTW_URL, FFTW_SHA1)
    print(" -> configuring")
    command = './configure '
    command += ' F77="' + fc + '"'
    command += ' FC="' + fc + '"'
    command += ' CC="' + cc + '"'
    command += ' CXX="' + cxx + '"'
    command += ' --enable-float'
    command += ' --enable-shared'
    command += ' --prefix=' + prefix
    run(command, 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')


def install_boost(prefix):
    BOOST_URL = "http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz"
    BOOST_SHA1 = '61ed0e57d3c7c8985805bb0682de3f4c65f4b6e5'
    download_expand_tarball('Boost C++', BOOST_URL, BOOST_SHA1)
    print(" -> configuring")
    command = './bootstrap.sh'
    command += ' --with-libraries=system,thread,date_time'
    command += ' --prefix=' + prefix
    run(command, 'log_configure')
    print(" -> making")
    run('./b2', 'log_make')
    print(" -> installing")
    run('./b2 install', 'log_make_install')


def install_zlib(prefix):
    ZLIB_URL = "http://downloads.sourceforge.net/project/libpng/zlib/1.2.7/zlib-1.2.7.tar.gz"
    ZLIB_SHA1 = '4aa358a95d1e5774603e6fa149c926a80df43559'
    download_expand_tarball('ZLIB (for HDF5)', ZLIB_URL, ZLIB_SHA1)
    print(" -> configuring")
    run('./configure --prefix=' + prefix, 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')


def install_hdf5(prefix, fc, is_g95, is_gfortran, gfortran_version):
    HDF5_URL = 'http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8.12/src/hdf5-1.8.12.tar.gz'
    HDF5_SHA1 = '965d954d596cfa694f3260856a6406ea69e46e68'
    download_expand_tarball('HDF5', HDF5_URL, HDF5_SHA1)

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
    command = './configure'
    command += ' FC="' + fc + '"'
    command += ' --with-zlib={prefix}/include,{prefix}/lib'.format(prefix=prefix)
    command += ' --prefix={prefix}'.format(prefix=prefix)
    run(command, 'log_configure')
    print(" -> making")
    run('make', 'log_make')
    print(" -> installing")
    run('make install', 'log_make_install')


def main():
    global VERBOSE
    cmd_opts = get_cmd_opts()
    INSTALL_HDF5  = cmd_opts.install_hdf5
    INSTALL_FFTW  = cmd_opts.install_fftw
    INSTALL_BOOST = cmd_opts.install_boost
    VERBOSE       = cmd_opts.verbose
    fc            = cmd_opts.fc_compiler
    cc            = cmd_opts.cc_compiler
    cxx           = cmd_opts.cxx_compiler
    prefix        = cmd_opts.installation_path
    
    if (INSTALL_HDF5 or INSTALL_FFTW or INSTALL_BOOST) is False:
        print("Nothing to install! Use 'python install.py --help' for options.")
        sys.exit(1)
    
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
    
    # The following section deals with issues that occur 
    # when using the intel fortran compiler with gcc.
    
    # Check whether C compiler is gcc
    # FIXME: does not find 'gcc' : gcc (Ubuntu/Linaro 4.7.3-1ubuntu1) 4.7.3
    output = p_get_first_line(cc + ' --version')
#     is_gcc = 'GCC' in output
    is_gcc = 'GCC' in output.lower()
    
    # get the Fortran compiler response to a version request
    try:
        output = p_get_first_line(fc + ' --version')
    except OSError:
        msg = 'Fortran compiler not found: ' + fc
        raise OSError, msg
    # Check whether Fortran compiler is ifort
    # FIXME: is_ifort is unused
    is_ifort = '(IFORT)' in output

    # Check whether Fortran compiler is g95
    is_g95 = 'g95' in output
    
    # Check whether Fortran compiler is pgfortran
    # FIXME: is_pgfortran is unused
    is_pgfortran = 'pgfortran' in output or \
                   'pgf95' in output
    
    # Check whether Fortran compiler is gfortran
    is_gfortran = 'GNU Fortran' in output
    if is_gfortran:
        gfortran_version = version.LooseVersion(p_get_first_line(fc + ' -dumpversion').split()[-1])
    
    # On MacOS X, when using gcc 4.5 or 4.6, the fortran compiler needs to link
    # with libgcc_eh that is in the gcc library directory. This is not needed if
    # using gfortran 4.5 or 4.6, but it's easier to just add it for all
    # compilers.
    if system == 'Darwin' and is_gcc:
        # Determine gcc version
        gcc_version = version.LooseVersion(p_get_first_line(cc + ' -dumpversion'))
        if gcc_version >= version.LooseVersion('4.5.0'):
            output = p_get_first_line(cc + ' -print-search-dirs')
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
        output = p_get_first_line('file test_arch_c').strip()
        if output == 'test_arch_c: Mach-O 64-bit executable x86_64':
            arch_c = 64
        elif output == 'test_arch_c: Mach-O executable i386':
            arch_c = 32
        else:
            arch_c = None
        
        # Check whether the Fortran compiler give different architecture builds by default
        open('test_arch.f90', 'w').write(TEST_ARCH_F90)
        subprocess.Popen(shlex.split(fc + ' test_arch.f90 -o test_arch_f90')).wait()
        output = p_get_first_line('file test_arch_f90').strip()
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
        install_fftw(prefix, fc, cc, cxx)
        os.chdir(work_dir)
    
    if INSTALL_BOOST:
        install_boost(prefix)
        os.chdir(work_dir)
    
    if INSTALL_HDF5:
        install_zlib(prefix)
        os.chdir(work_dir)
        install_hdf5(prefix, fc, is_g95, is_gfortran, gfortran_version)
        os.chdir(work_dir)
    
    # Go back to starting directory
    os.chdir(start_dir)
    print(" ")
    print("Installation successful!!!")
    print(" ")
    print("Before you start installing TomoPy, don't forget to:")
    print(" ")
    print("    1) Set LD_LIBRARY_PATH permanently in your shell to: %s" % prefix + '/lib')
    print("       hint: setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:%s" % prefix + '/lib')
    print(" ")
    print("    2) Set C_INCLUDE_PATH permanently in your shell to: %s" % prefix + '/include')
    print("       hint: setenv C_INCLUDE_PATH ${C_INCLUDE_PATH}:%s" % prefix + '/include')
    print(" ")


if __name__ == '__main__':
    #sys.argv = sys.argv[:1]
    # sys.argv.append('-h')
    #sys.argv += '/tmp/sandbox --verbose --boost --fftw'.split()
    #sys.argv += '/tmp/sandbox --verbose --hdf5'.split()
    main()
