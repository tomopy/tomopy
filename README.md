## About TomoPy

TomoPy is a Python toolbox to perform data processing and image reconstruction 
tasks at [APS](http://www.aps.anl.gov/ "APS"). It uses
[HDF5 file format](https://subversion.xray.aps.anl.gov/DataExchange/doc/trunk/ "Data Exchange")
as the standard means of data exchange.

## External Dependencies:
- [HDF5 1.8.12](http://www.hdfgroup.org/HDF5/ "HDF5")
- [FFTW 3.3.3](http://www.fftw.org "FFTW3")
- [Boost 1.55.0](http://www.boost.org "Boost C++")

## Python Dependencies:
- [NumPy 1.8.0](http://www.numpy.org "numpy")
- [SciPy 0.13.2](http://www.scipy.org "scipy")
- [H5Py 2.2.1](http://www.h5py.org "h5py")
- [PyWt 0.2.2](http://www.pybytes.com/pywavelets/ "pywt")

## Installing TomoPy

Make sure you have Python 2.7 installed and install above dependencies before installing TomoPy. Then:

- To insall from egg distribution download the [latest released egg](https://github.com/tomopy/tomopy/releases) for your system, open shell prompt and type `easy_install tomopy-egg-name`.
- To build and install from source, download the [latest released source](https://github.com/tomopy/tomopy/releases), type `python setup.py install` in the directory where `setup.py` resides.

