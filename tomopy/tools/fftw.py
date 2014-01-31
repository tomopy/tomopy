import ctypes
import numpy as np

import os
libfftw = ctypes.CDLL(os.path.dirname(__name__) + "/fftw/libfftw.so")

def fftw(*args, **kwargs):
    """ 
    Compute the one-dimensional discrete Fourier Transform (DWT).
        
    This function computes the one-dimentional DFT with the
    efficient FFTW algorithm. It is a thin wrapper on FFTW
    C package (http://www.fftw.org).
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
        
    Returns
    -------
    out : complex ndarray
        Output array.
        
    See Also
    --------
    ifftw : The inverse of `fft`.
    ifftw2 : The inverse of `fft2`.
    fftw2 : The two-dimensional FFT.
    
    Notes
    -----
    It is calculated efficiently when the input array is
    symmetric, that is highest when the array sizes are 
    powers of 2.
    """
    try:
        libfftw.fftw()
    except:
        print "fftw wrapper not implemented yet"

def ifftw(*args, **kwargs):
    """
    Compute the one-dimensional inverse discrete Fourier Transform (DWT).
    
    This function computes the inverse of the one-dimentional DFT 
    computed by `fftw`. It is a thin wrapper on FFTW
    C package (http://www.fftw.org).
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    
    Returns
    -------
    out : complex ndarray
        Output array.
    
    See Also
    --------
    ifftw2 : The inverse of `fft2`.
    fftw : The one-dimensional FFT.
    fftw2 : The two-dimensional FFT.
    """
    try:
        libfftw.ifftw()
    except:
        print "ifftw wrapper not implemented yet"

def fftw2(a):
    """
    Compute the two-dimensional discrete Fourier Transform (DWT).
    
    This function computes the two-dimentional DFT with the
    efficient FFTW algorithm. It is a thin wrapper on FFTW
    C package (http://www.fftw.org).
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    
    Returns
    -------
    out : complex ndarray
        Output array.
    
    See Also
    --------
    ifftw : The inverse of `fft`.
    ifftw2 : The inverse of `fft2`.
    fftw : The one-dimensional FFT.
    
    Notes
    -----
    It is calculated efficiently when the input array is
    symmetric, that is highest when the array sizes are
    powers of 2.
    """
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    
    _a = np.array(a, dtype='complex64')
    dimx = np.array(a.shape[1])
    dimy = np.array(a.shape[0])
    direction = np.array(-1)
    libfftw.fftw_2d(_a.ctypes.data_as(c_float_p),
                    dimx.ctypes.data_as(c_int_p),
                    dimy.ctypes.data_as(c_int_p),
                    direction.ctypes.data_as(c_int_p))
    return _a

def ifftw2(a):
    """
    Compute the two-dimensional inverse discrete Fourier Transform (DWT).
    
    This function computes the inverse of the two-dimentional DFT
    computed by `fftw`. It is a thin wrapper on FFTW
    C package (http://www.fftw.org).
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    
    Returns
    -------
    out : complex ndarray
        Output array.
    
    See Also
    --------
    ifftw : The inverse of `fft`.
    fftw : The one-dimensional FFT.
    fftw2 : The two-dimensional FFT.
    """
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)
    
    _a = np.array(a, dtype='complex64')
    dimx = np.array(a.shape[1])
    dimy = np.array(a.shape[0])
    direction = np.array(1)
    libfftw.fftw_2d(_a.ctypes.data_as(c_float_p),
                      dimx.ctypes.data_as(c_int_p),
                      dimy.ctypes.data_as(c_int_p),
                      direction.ctypes.data_as(c_int_p))
    _a = _a / (dimx * dimy)
    return _a