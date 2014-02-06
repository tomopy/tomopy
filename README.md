## About TomoPy

TomoPy is a Python toolbox to perform data processing and image reconstruction 
tasks at [APS](http://www.aps.anl.gov/ "APS"). It uses
[HDF5 file format](https://subversion.xray.aps.anl.gov/DataExchange/doc/trunk/ "Data Exchange")
as the standard means of data exchange.

## External Dependencies:
- [FFTW 3.3.3](http://www.fftw.org "FFTW3") (only float library is required)
- [Boost C++ 1.55.0](http://www.boost.org "Boost C++") (only thread, system and date_time libraries are required)

Note: You can use ``python install.py <desired-directory> --boost --fftw`` for a quick install of these packages. If ``<desired-directory>`` is unspecified, by default it installs them into ``/usr/local``. 

## Python Dependencies:
- [NumPy 1.8.0](http://www.numpy.org "numpy")
- [SciPy 0.13.2](http://www.scipy.org "scipy")
- [H5Py 2.2.1](http://www.h5py.org "h5py")
- [PyWt 0.2.2](http://www.pybytes.com/pywavelets/ "pywt")
- [Pillow 2.3.0](https://pypi.python.org/pypi/Pillow// "pillow")

## Installing TomoPy

Make sure you have [Python 2.6](http://www.python.org/download/releases/2.6/ "tsss...") or [2.7](http://www.python.org/download/releases/2.7/ "tsss...") and the above dependencies installed in your system. 

Then, set the following environment variables in the shell like (C-shell assumed):
- ``setenv LD_LIBRARY_PATH <my-boost-lib-dir>:<my-fftw-lib-dir>``  
- ``setenv C_INCLUDE_PATH <my-boost-include-dir>:<my-fftw-include-dir>``

Once the environment variables are set correctly, then:

- To insall from an egg distribution download the [latest released egg](https://github.com/tomopy/tomopy/releases) for your system, open shell prompt and type `easy_install my-egg-name` in the directory where the egg resides. 
- To build and install from source, download the [latest source tarball](https://github.com/tomopy/tomopy/releases), open shell prompt and type `python setup.py install` in the directory where `setup.py` resides.

To test if installation succesfull, in the shell try:

- ``python -c "import tomopy"``

If it doesn't complain you are good to go!



