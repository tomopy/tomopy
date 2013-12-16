# -*- coding: utf-8 -*-
# Filename: tomo.py
""" Module for the basic functions to wrap up C++ reconstruction code.
This module is the python version of the IDL wrapper code written
by Mark Rivers with some slight modifications.
"""
import ctypes
import numpy as np
import os
import time
import platform
from dataio.file_types import Hdf5
from dataio.file_types import Tiff


class tomoRecon:
    def __init__(self,
                 dataset,
                 sinoScale=1e4,
                 reconScale=1,
                 paddedSinogramWidth=None,
                 airPixels=10,
                 ringWidth=9,
                 fluorescence=0,
                 reconMethod=0,
                 reconMethodTomoRecon=0,
                 numThreads=24,
                 slicesPerChunk=32,
                 debugFileName='',
                 debug=0,
                 geom=0,
                 pswfParam=6,
                 sampl=1,
                 MaxPixSiz=1,
                 ROI=1,
                 X0=0,
                 Y0=0,
                 ltbl=512,
                 fname='shepp',
                 BP_Method=0,
                 BP_filterName='shepp',
                 BP_filterSize=100,
                 RiemannInterpolation=0,
                 RadonInterpolation=0):
        """ Initialize tomography parameters.

        Parameters
        ----------
        numPixels : scalar
            Number of pixels in sinogram row before padding.

        numProjections : scalar
            Number of angles.

        numSlices : scalar
            Number of slices.

        sinoScale : scalar
            Scale factor to multiply sinogram when airPixels=0.

        reconScale : scalar
            Scale factor to multiple reconstruction.

        paddedSinogramWidth : scalar
            Number of pixels in sinogram after padding. If ``None``, it is
            assumed to be the smallest power of two which is higher than
            ``numPixel``.

        airPixels : scalar
            Number of pixels of air to average at each end of sinogram row.

        ringWidth : scalar
            Number of pixels to smooth by when removing ring artifacts.

        fluorescence : scalar
            0=absorption data, 1=fluorescence.

        reconMethod : scalar
            0=tomoRecon, 1=Gridrec, 2=Backproject.

        numThreads : scalar
            Number of threads.

        slicesPerChunk : scalar
            Number of slices per chunk.

        debug : scalar
            Note: if not 0 there may be some memory leakage problems.

        debugFileName : str

        geom : scalar
            0 if array of angles provided; 1,2 if uniform in half,full circle.

        pswfParam : scalar
            PSWF parameter.

        sampl : scalar
            "Oversampling" ratio.

        MaxPixSiz : scalar
            Max pixel size for reconstruction.

        ROI : scalar
            Region of interest (ROI) relative size.

        X0 : scalar
            (X0,Y0)=Offset of ROI from rotation axis in units of
            center-to-edge distance.

        Y0 : scalar
            (X0,Y0)=Offset of ROI from rotation axis in units of
            center-to-edge distance.

        ltbl : scalar
            No. elements in convolvent lookup tables.

        fname : str {shepp, hann, hamm, ramp}
            Name of filter function.

        BP_Method : scalar
            0=Riemann, 1=Radon.

        BP_filterName : str {none, shepp, hann, hamm, ramp}
            Name of filter function.

        BP_filterSize : scalar
            Length of filter.

        RiemannInterpolation :scalar
            0=none, 1=bilinear, 2=cubic.

        RadonInterpolation : scalar
            0=none, 1=linear.
        """
        self.params = tomoRecon.cStruct()
        self.params.numProjections = dataset.data.shape[0]
        self.params.numSlices = dataset.data.shape[1]
        self.params.numPixels = dataset.data.shape[2]
        self.params.sinoScale = sinoScale
        self.params.reconScale = reconScale
        if paddedSinogramWidth is None:
            paddedSinogramWidth = 0
            powerN = 1
            while (paddedSinogramWidth < dataset.data.shape[2]):
                paddedSinogramWidth = 2 ** powerN
                powerN += 1
        elif paddedSinogramWidth < dataset.data.shape[2]:
            raise ValueError('paddedSinogramWidth must be higher than the number of pixels.')
        self.params.paddedSinogramWidth = paddedSinogramWidth
        self.params.airPixels = airPixels
        self.params.ringWidth = ringWidth
        self.params.fluorescence = fluorescence
        self.params.reconMethod = reconMethod
        self.params.reconMethodTomoRecon = 0
        self.params.reconMethodGridrec = 1
        self.params.reconMethodBackproject = 2
        self.params.numThreads = numThreads
        self.params.slicesPerChunk = slicesPerChunk
        self.params.debug = 0
        for m in range(len(map(ord, debugFileName))):
            self.params.debugFileName[m] = map(ord, debugFileName)[m]
        self.params.geom = geom
        self.params.pswfParam = pswfParam
        self.params.sampl = sampl
        self.params.MaxPixSiz = MaxPixSiz
        self.params.ROI = ROI
        self.params.X0 = X0
        self.params.Y0 = Y0
        self.params.ltbl = ltbl
        for m in range(len(map(ord, fname))):
            self.params.fname[m] = map(ord, fname)[m]
        self.params.BP_Method = BP_Method
        self.params.BP_MethodRiemann = 0
        self.params.BP_MethodRadon = 1
        for m in range(len(map(ord, BP_filterName))):
            self.params.BP_filterName[m] = map(ord, BP_filterName)[m]
        self.params.BP_filterSize = BP_filterSize
        self.params.RiemannInterpolation = RiemannInterpolation
        self.params.RiemannInterpolationNone = 0
        self.params.RiemannInterpolationBilinear = 1
        self.params.RiemannInterpolationCubic = 2
        self.params.RadonInterpolation = RadonInterpolation
        self.params.RadonInterpolationNone = 0
        self.params.RadonInterpolationLinear = 1


    class cStruct(ctypes.Structure):
        _fields_ = [("numPixels", ctypes.c_int),
                    ("numProjections", ctypes.c_int),
                    ("numSlices", ctypes.c_int),
                    ("sinoScale", ctypes.c_float),
                    ("reconScale", ctypes.c_float),
                    ("paddedSinogramWidth", ctypes.c_int),
                    ("airPixels", ctypes.c_int),
                    ("ringWidth", ctypes.c_int),
                    ("fluorescence", ctypes.c_int),
                    ("reconMethod", ctypes.c_int),
                    ("reconMethodTomoRecon", ctypes.c_int),
                    ("reconMethodGridrec", ctypes.c_int),
                    ("reconMethodBackproject", ctypes.c_int),
                    ("numThreads", ctypes.c_int),
                    ("slicesPerChunk", ctypes.c_int),
                    ("debug", ctypes.c_int),
                    ("debugFileName", ctypes.c_byte*256),
                    ("geom", ctypes.c_int),
                    ("pswfParam", ctypes.c_float),
                    ("sampl", ctypes.c_float),
                    ("MaxPixSiz", ctypes.c_float),
                    ("ROI", ctypes.c_float),
                    ("X0", ctypes.c_float),
                    ("Y0", ctypes.c_float),
                    ("ltbl", ctypes.c_int),
                    ("fname", ctypes.c_byte*16),
                    ("BP_Method", ctypes.c_int),
                    ("BP_MethodRiemann", ctypes.c_int),
                    ("BP_MethodRadon", ctypes.c_int),
                    ("BP_filterName", ctypes.c_byte*16),
                    ("BP_filterSize", ctypes.c_int),
                    ("RiemannInterpolation", ctypes.c_int),
                    ("RiemannInterpolationNone", ctypes.c_int),
                    ("RiemannInterpolationBilinear", ctypes.c_int),
                    ("RiemannInterpolationCubic", ctypes.c_int),
                    ("RadonInterpolation", ctypes.c_int),
                    ("RadonInterpolationNone", ctypes.c_int),
                    ("RadonInterpolationLinear", ctypes.c_int)]


    def show(self):
        print 'Reconstruction parameters:'
        print '             numPixels: ' + str(self.params.numPixels)
        print '        numProjections: ' + str(self.params.numProjections)
        print '             numSlices: ' + str(self.params.numSlices)
        print '             sinoScale: ' + str(self.params.sinoScale)
        print '            reconScale: ' + str(self.params.reconScale)
        print '   paddedSinogramWidth: ' + str(self.params.paddedSinogramWidth)
        print '             airPixels: ' + str(self.params.airPixels)
        print '             ringWidth: ' + str(self.params.ringWidth)
        print '          fluorescence: ' + str(self.params.fluorescence)
        print '  reconMethodTomoRecon: ' + str(self.params.reconMethodTomoRecon)
        print '            numThreads: ' + str(self.params.numThreads)
        print '        slicesPerChunk: ' + str(self.params.slicesPerChunk)
        print '                 debug: ' + str(self.params.debug)
        print '                  geom: ' + str(self.params.geom)
        print '             pswfParam: ' + str(self.params.pswfParam)
        print '                 sampl: ' + str(self.params.sampl)
        print '             MaxPixSiz: ' + str(self.params.MaxPixSiz)
        print '                   ROI: ' + str(self.params.ROI)
        print '                    X0: ' + str(self.params.X0)
        print '                    Y0: ' + str(self.params.Y0)
        print '                  ltbl: ' + str(self.params.ltbl)
        print '             BP_Method: ' + str(self.params.BP_Method)
        print '         BP_filterSize: ' + str(self.params.BP_filterSize)
        print '  RiemannInterpolation: ' + str(self.params.RiemannInterpolation)
        print '    RadonInterpolation: ' + str(self.params.RadonInterpolation)


    def run(self, reconInput, sliceNo=None, sharedLibrary=None, printInfo=True):
        """ Performs tomographic reconstruction using the tomoRecon object from
        tomoRecon.cpp and tomoReconPy.cpp.

        tomoRecon uses the "gridrec" algorithm written by Bob Marr
        and Graham Campbell (not sure about authors) at BNL in 1997.
        The basic algorithm is based on FFTs. The source codes are available
        in the current directory at ``./tomoRecon``.

        This file uses ``ctypes`` to call tomoReconPy.cpp which is a thin
        wrapper to tomoRecon.cpp.

        Parameters
        ----------
        reconInput : ndarray, shape(numProjections, numSlices, numPixels)
            An array of normalized projections. It will be converted to
            type ``float32`` if it is another data type.

        sharedLibrary : str, optional
            The shared library of gridrec. If ``None`` it looks at the default
            lib directory in tomoRecon folder.
        """
        if printInfo is True:
            print "Reconstructing..."

        # Assign/calculate values for the projection angles.
        if reconInput.angles is None:
            angles = (np.linspace(0, self.params.numProjections,
                                self.params.numProjections)
                    * 180 / self.params.numProjections).astype('float32')

        ## Assign sliceNo.
        numSlices = self.params.numSlices
        if sliceNo is not None:
           numSlices = 1

        # Get the shared library
        if sharedLibrary is None:
            sharedLibrary = tomoRecon.locateSharedLibrary()

        # Construct the reconstruction object used by the wrapper.
        RECON_LIB = ctypes.CDLL(sharedLibrary)
        RECON_LIB.reconCreate(ctypes.byref(self.params),
                            angles.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

        # Assign/calculate values for the centers of the slices.
        if reconInput.center is None:
            centerArr = np.ones(numSlices,
                                dtype='float32') * \
                                self.params.numPixels/2
        else:
            center = np.array(reconInput.center, dtype='float32')
            if center.size is 1:
                centerArr = np.ones(numSlices, dtype='float32') * center
            elif center.size is numSlices:
                centerArr = np.array(center, dtype='float32')
            else:
                raise ValueError('Center size must be either a scalar or equal to the number of slices.')

        # Prepare input variables for the C-types and feed the wrapper.
        #print reconInput.data.shape
        _numSlices = ctypes.c_int(numSlices)
        reconInput = np.array(reconInput.data[:, sliceNo, :], dtype='float32')
        self.data = np.empty((numSlices,
                                     self.params.numPixels,
                                     self.params.numPixels), dtype='float32')
        RECON_LIB.reconRun(ctypes.byref(_numSlices),
                        centerArr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        reconInput.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        # Wait for the reconstruction to be completed.
        while True:
            reconComplete, slicesRemaining = tomoRecon.poll()
            if reconComplete.value is 1:
                break
            else:
                time.sleep(0.01)

        # Destruct the reconstruction object used by the wrapper.
        RECON_LIB.reconDelete()


    @staticmethod
    def poll(sharedLibrary=None):
        """ Polls the tomoRecon object created by tomo_recon to read the
        reconstruction status and the number of slices remaining.

        Parameters
        ----------
        sharedLibrary : str, optional
            The shared library of gridrec. If ``None`` it looks at the default
            lib directory in tomoRecon folder.

        Returns
        -------
        reconComplete: scalar
            1 if the reconstruction is complete, 0 if it is not yet complete.

        slicesRemaining : scalar
            slicesRemaining is the number of slices remaining to be reconstructed.
        """
        # Get the shared library
        if sharedLibrary is None:
            sharedLibrary = tomoRecon.locateSharedLibrary()
        reconComplete = ctypes.c_int(0)
        slicesRemaining = ctypes.c_int(0)
        RECON_LIB = ctypes.CDLL(sharedLibrary)
        RECON_LIB.reconPoll(ctypes.byref(reconComplete),
                            ctypes.byref(slicesRemaining))
        return reconComplete, slicesRemaining


    def write(self, outputFile, slicesStart=None, slicesEnd=None):
        """ Write reconstructed data into file.
        """
        if outputFile.endswith('hdf'):
            Hdf5.write(self.data, outputFile)
        elif outputFile.endswith('tiff'):
            Tiff.write(self.data,
                       outputFile,
                       slicesStart=slicesStart,
                       slicesEnd=slicesEnd)


    @staticmethod
    def locateSharedLibrary():
        """ Locates the tomoRecon shared library. The library is by default at
        ``./tomoRecon/lib``.

        Returns
        -------
        libPath: str
            Full path of the shared library.
        """
        if os.environ.has_key('TOMO_RECON_SHARE'):
            libPath = os.environ.get('TOMO_RECON_SHARE')
        else:
            folderName = platform.uname()[0].lower() + '-' +platform.uname()[4]
            if platform.uname()[0].lower() == 'linux':
                extension = 'so'
            elif platform.uname()[0].lower() == 'darwin':
                extension = 'dylib'
            else:
                extension = 'dll'
            libPath =  os.getcwd() + '/tomoRecon/lib/' \
                    + folderName + '/libtomoRecon.' + extension
            print libPath
        return libPath


def fftw2d(data, direction='forward', sharedLibrary=None):
    """ Calculate FFT and inverse FFT of the dataset using
    FFTW package.

    This function uses ``ctypes`` to call fftwPy.cpp
    which is a thin wrapper to fftw3.

    Parameters
    ----------
    data : ndarray
        2-D input matrix.

    direction: str, optional
        ``forward`` FFT or ``backward`` (inverse) FFT.

    sharedLibrary : str, optional
        The shared library of gridrec. If ``None`` it looks at the default
        lib directory in tomoRecon folder.

    Returns
    -------
    out : ndarray
        Transformed output matrix.
    """
    # Get the shared library
    if sharedLibrary is None:
        sharedLibrary = tomoRecon.locateSharedLibrary()
    RECON_LIB = ctypes.CDLL(sharedLibrary)
    c_float_p = ctypes.POINTER(ctypes.c_float)
    c_int_p = ctypes.POINTER(ctypes.c_int)

    _data = np.array(data, dtype='complex64')
    dimx = np.array(data.shape[1])
    dimy = np.array(data.shape[0])
    if direction is 'forward':
        direction = np.array(-1)
        RECON_LIB.fftw_2d(_data.ctypes.data_as(c_float_p),
                        dimx.ctypes.data_as(c_int_p),
                        dimy.ctypes.data_as(c_int_p),
                        direction.ctypes.data_as(c_int_p))

    if direction is 'backward':
        direction = np.array(1)
        RECON_LIB.fftw_2d(_data.ctypes.data_as(c_float_p),
                        dimx.ctypes.data_as(c_int_p),
                        dimy.ctypes.data_as(c_int_p),
                        direction.ctypes.data_as(c_int_p))
        _data = _data / (dimx * dimy)
    return _data
