"""
Edited for use with cupy.
"""
try:
    import scipy.fft
    import cupyx.scipy.fft as cufft
    from scipy.fft import next_fast_len
    fftmodule = scipy.fft
    scipy.fft.set_global_backend(cufft)
except ImportError:
    import numpy.fft
    fftmodule = numpy.fft
    from scipy.fftpack import next_fast_len

__all__ = ['fftmodule', 'next_fast_len']