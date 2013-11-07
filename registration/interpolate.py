# -*- coding: utf-8 -*-
# Filename: interpolate.py
""" Module for interpolation.
"""
import ctypes
import numpy as np
import geometry

# Globals
INTERP_LIB = ctypes.CDLL('./registration/interp.so')
c_int_p = ctypes.POINTER(ctypes.c_int)
c_float_p = ctypes.POINTER(ctypes.c_float)

def interp2d(from_img, 
             from_grid, 
             tform=None, 
             to_grid=None, 
             kind='bilinear'):
    """ Calculates interpolation of image ``from_img`` specified by
    the spatial referencing object ``from_grid`` on ``to_grid`` after
    the trasformation, specified by the geometric transformation
    object ``tform``.
    
    This method is the equivalent (and faster) version of the function 
    scipy.interpolate.interp2d() and it uses interp.cpp for the computation.
    (Probably equivalent to RectBivariateSpline).
    
    Parameters
    ----------
    from_img : ndarray
        Input image values on a structure grid specified by ``from_grid``.
        
    from_grid : spatial referencing object
        Specifies the coordinates of the input image ``from_img``.
        
    tform : Geometric transformation object
        The transformation to be applied to ''from_image''.
        
    to_grid : spatial referencing object
        Specifies the desired interpolation coordinates of the input 
        image ``from_image'' after geometric transformation.
 
    kind : str, optional
        The kind of interpolation to use. Default is ``bilinear``.
        
            - Nearest-Neighbor : 'nearest'
            
            - Bilinear : 'bilinear'
            
            - Bicubic : 'bicubic'
        
    Returns
    -------
    to_img : ndarray
        Output interpolated image compatible with ``to_grid``.
    """
    if tform is None:
        tform = geometry.transform2d()
        
    if to_grid is None:
        to_grid = from_grid

    # First transform ``to_grid`` according to the inverse of the
    # ``tform``. This transformation is equaivalent of transforming 
    # ``from_grid`` in the forward direction. Then keep track of 
    # points inside ``from_grid`` using a masked array.
    point = to_grid.transform(tform, 'inverse')
    mask = from_grid.overlap(point)
    
    # Interpolation function inputs.
    _from_img = np.zeros(from_grid.resolution+2, dtype='float32')
    _from_img[1:-1, 1:-1] = from_img
    _from_img = _from_img.flatten(0)
    _point = point[mask]
    _npoint = _point.shape[0]
    _limits = from_grid.limits
    _resolution = from_grid.resolution
    
    # C function call
    _point = np.array(_point, dtype='float32')
    _npoint = np.array(_npoint, dtype='int32')
    _resolution = np.array(_resolution, dtype='int32')
    _limits = np.array(_limits, dtype='float32')
    to_img = np.zeros(to_grid.resolution, dtype='float32') # Output
    if kind is 'nearest':
        INTERP_LIB.nearestInterp2d(_from_img.ctypes.data_as(c_float_p),
                                  _point.ctypes.data_as(c_float_p),  
                                  _npoint.ctypes.data_as(c_int_p),
                                  _resolution.ctypes.data_as(c_int_p),
                                  _limits.ctypes.data_as(c_float_p),
                                  to_img.ctypes.data_as(c_float_p))
    elif kind is 'bilinear':
        INTERP_LIB.bilinearInterp2d(_from_img.ctypes.data_as(c_float_p),
                                   _point.ctypes.data_as(c_float_p),  
                                   _npoint.ctypes.data_as(c_int_p),
                                   _resolution.ctypes.data_as(c_int_p),
                                   _limits.ctypes.data_as(c_float_p),
                                   to_img.ctypes.data_as(c_float_p))
    else:
        raise ValueError('Unknown interpolation type.')
    values = np.zeros(to_grid.resolution[0] * 
                      to_grid.resolution[1], dtype='float32')
    ind = np.where(mask == True)
    values[ind] = to_img
    return np.reshape(values, to_grid.resolution)
    
    
def fineToCoarse(from_img, 
                 from_grid, 
                 tform=None, 
                 to_grid=None, 
                 kind='bilinear'):
    """ Calculates interpolation of image ``from_img`` specified by
    the spatial referencing object ``from_grid`` on ``to_grid`` after
    the trasformation, specified by the geometric transformation
    object ``tform``.

    This transformation is  preserves the information within 
    image if the target transformation grid is coarser than the 
    starting grid. Otherwise, there may be holes (zeros) in the 
    returned image. It is better suited to pyramidal approach for 
    downscaling than conventional interpolations of scipy.
    
    Parameters
    ----------
    from_img : ndarray
        Input image values on a structure grid specified by ``from_grid``.
        
    from_grid : spatial referencing object
        Specifies the coordinates of the input image ``from_img``.
        
    to_grid : spatial referencing object
        Specifies the desired interpolation coordinates of the input 
        image ``from_image'' after geometric transformation.
        
    tform : Geometric transformation object
        The transformation to be applied to ''from_image''.
        
    kind : str, optional
        The kind of interpolation to use. Default is ``bilinear``.
        
            - Nearest-Neighbor : 'nearest'
            
            - Bilinear : 'bilinear'
            
            - Bicubic : 'bicubic'

    Returns
    -------
    to_img : ndarray
        Output interpolated image compatible with ``to_grid``.
    """
    if tform is None:
        tform = geometry.transform2d()
        
    if to_grid is None:
        to_grid = from_grid
        
    # First transform ``from_grid`` according to given ``tform``
    # and keep track of points inside ``to_grid`` using a masked array.
    point = from_grid.transform(tform, 'forward')
    mask = to_grid.overlap(point)
    
    # Interpolation function inputs.
    _from_img = from_img.flatten(0)[mask]
    _point = point[mask]
    _npoint = _point.shape[0]
    _resolution = to_grid.resolution
    _limits = to_grid.limits
    
    # C function call
    _from_img = np.array(_from_img, dtype='float32')
    _point = np.array(_point, dtype='float32')
    _npoint = np.array(_npoint, dtype='int32')
    _resolution = np.array(_resolution, dtype='int32')
    _limits = np.array(_limits, dtype='float32')
    to_img = np.zeros(_resolution+2, dtype='float32') # Output
    counter = np.zeros(_resolution+2, dtype='float32') # Output
    if kind is 'nearest':
        INTERP_LIB.nearestFineToCoarse(_from_img.ctypes.data_as(c_float_p),
                                      _point.ctypes.data_as(c_float_p),  
                                      _npoint.ctypes.data_as(c_int_p),
                                      _resolution.ctypes.data_as(c_int_p),
                                      _limits.ctypes.data_as(c_float_p),
                                      to_img.ctypes.data_as(c_float_p),
                                      counter.ctypes.data_as(c_float_p))
    elif kind is 'bilinear':
        INTERP_LIB.bilinearFineToCoarse(_from_img.ctypes.data_as(c_float_p),
                                        _point.ctypes.data_as(c_float_p),  
                                        _npoint.ctypes.data_as(c_int_p),
                                        _resolution.ctypes.data_as(c_int_p),
                                        _limits.ctypes.data_as(c_float_p),
                                        to_img.ctypes.data_as(c_float_p),
                                        counter.ctypes.data_as(c_float_p))
                                        
    else:
        raise ValueError('Unknown interpolation type.')
    return to_img[1:-1, 1:-1]
    
    
def zoom(img, zoom, kind='bilinear'):
    """ 
    """
    from_grid = geometry.grid2d(resolution=img.shape)
    tform = geometry.transform2d()
    to_grid = geometry.grid2d(resolution=np.multiply(img.shape, zoom))

    # First transform ``to_grid`` according to the inverse of the
    # ``tform``. This transformation is equaivalent of transforming 
    # ``from_grid`` in the forward direction. Then keep track of 
    # points inside ``from_grid`` using a masked array.
    point = to_grid.transform(tform, 'inverse')
    mask = from_grid.overlap(point)

    # Interpolation function inputs.
    _from_img = np.zeros(from_grid.resolution+2, dtype='float32')
    _from_img[1:-1, 1:-1] = img
    _from_img = _from_img.flatten(0)
    _point = point[mask]
    _npoint = _point.shape[0]
    _limits = from_grid.limits
    _resolution = from_grid.resolution
    
    # C function call
    _point = np.array(_point, dtype='float32')
    _npoint = np.array(_npoint, dtype='int32')
    _resolution = np.array(_resolution, dtype='int32')
    _limits = np.array(_limits, dtype='float32')
    to_img = np.zeros(to_grid.resolution, dtype='float32') # Output
    if kind is 'nearest':
        INTERP_LIB.nearestInterp2d(_from_img.ctypes.data_as(c_float_p),
                                  _point.ctypes.data_as(c_float_p),  
                                  _npoint.ctypes.data_as(c_int_p),
                                  _resolution.ctypes.data_as(c_int_p),
                                  _limits.ctypes.data_as(c_float_p),
                                  to_img.ctypes.data_as(c_float_p))

    elif kind is 'bilinear':
        INTERP_LIB.bilinearInterp2d(_from_img.ctypes.data_as(c_float_p),
                                   _point.ctypes.data_as(c_float_p),  
                                   _npoint.ctypes.data_as(c_int_p),
                                   _resolution.ctypes.data_as(c_int_p),
                                   _limits.ctypes.data_as(c_float_p),
                                   to_img.ctypes.data_as(c_float_p))
                                        
    else:
        raise ValueError('Unknown interpolation type.')
    values = np.zeros(to_grid.resolution[0] * 
                      to_grid.resolution[1], dtype='float32')
    ind = np.where(mask == True)
    values[ind] = to_img
    return np.reshape(values, to_grid.resolution)