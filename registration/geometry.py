# -*- coding: utf-8 -*-
# Filename: geometry.py
""" Module for geometric transforms and spatial referencing.
"""
import numpy as np
from scipy import linalg
                            
class grid2d:
    """Reference 2-D spatial referencing object.
    
    ``grid2d`` object encapsulates the spatial positioning of the image. The 
    pixel spacing of each dimension can be different and calculated according 
    to the limits.
    
    Construction
    ------------
    ``R = grid2d(resolution, limits)`` creates an ``grid2d`` object given 
    the resolution of image and the coordinate limits in each dimension,
    specified by ``limits``.

    Axes
    ----
    Coordinate axes are defined as (this is more convenient
    with the imshow):
            y
      .-----> 
      |
      |
    x v
    
    Attributes
    ----------
    resolution : array-like, int, shape (2,)
        Resolution of the image associated with the object.
            >>> resolution = [numRows, numColumns]
    
    limits : array-like, float, shape (4,)
        Minimum and maximum coordinate values in each dimension.
            >>> limits = [xmin, xmax, ymin, ymax]
        
    pixelSize : array-like, float, shape (2,)
        Size of the pixels.
            >>> pixelSize = [xedge, yedge]
    """
    def __init__(self, resolution, limits=[-1, 1, -1, 1]):   
        """ Constructs 2-D spatial reference object.
        
        Parameters
        ----------
        resolution : array-like, int, shape (2,)
            Resolution of the image associated with the object.
            >>> resolution = [numRows, numColumns]
            
        limits : array-like, float, shape (4,), optional
            Minimum and maximum coordinate values in each dimension.
            >>> limits = [xmin, xmax, ymin, ymax]
        """
        self.resolution = np.array(resolution, dtype='int')
        self.limits = np.array(limits, dtype='float')
        self.pixelSize = [self.limits[1] - self.limits[0],
                          self.limits[3] - self.limits[2]]/ self.resolution
        
    def transform(self, tform, option='forward'):
        """ Transform points.
        
        Calculates the transformed points according to the geometric
        transformation object ``tform``. 
        
        Parameters
        ----------
        tform : geometric transformation object

        option : str
            Determines if the forward of inverse transformation will be done.
            
                - 'forward'
                - 'inverse'
        
        Returns
        -------
        point : ndarray, float, shape (,2)
            Transformed points.
        """
        if option is 'forward':
            return tform.transformForward(grid2d.gridPoints(self))
        elif option is 'inverse':
            return tform.transformInverse(grid2d.gridPoints(self))
        
    def overlap(self, point):
        """ Create a mask for the overlapping points.
        
        Returns a boolean mask array for the ``grid2d`` spatial referencing 
        object based on if the points given by ``point`` are inside or outside 
        the grid. 
        
        Parameters
        ----------
        point : ndarray, float, shape (,2)
            Input points. Must be at least 2 points.
        
        Returns
        -------
        out : bool, float, same length as ``point``
            A boolean mask array. The coefficients are ``True`` corresponding 
            to the points inside the grid and ``False`` for points outside the
            grid.
        """
        point = np.array(point, dtype='float')
        rangex = np.logical_and(point[:, 0] >= self.limits[0], 
                                point[:, 0] <= self.limits[1]) 
        rangey = np.logical_and(point[:, 1] >= self.limits[2], 
                                point[:, 1] <= self.limits[3])
        return np.logical_and(rangex, rangey)

    def axesCoords(self):
        """ Coordinates of points on X and Y axes.
        
        Returns the spatial coordinates of the points on the grid in X and 
        Y dimensions from the grid boundaries.
        
        Returns
        -------
        coordx : ndarray, float
            Spatial coordinates in X dimension.
        
        coordy : ndarray, float
            Spatial coordinates in Y dimension.
        """
        coordx = np.linspace(self.limits[0] + self.pixelSize[0]/2, 
                             self.limits[1] - self.pixelSize[0]/2, 
                             self.resolution[0])
        coordy = np.linspace(self.limits[2] + self.pixelSize[1]/2, 
                             self.limits[3] - self.pixelSize[1]/2, 
                             self.resolution[1])
        return coordx, coordy
        
    def gridPoints(self):
        """ Points on the 2-D grid.
        
        Returns a 2-D points array on the grid in (X, Y) notation. 
        
        Returns
        -------
        point : ndarray, float, shape (,2)
            Point array on the grid.
        """
        coordx, coordy = grid2d.axesCoords(self)
        x0, y0 = np.meshgrid(coordx, coordy)
        return np.c_[x0.flatten(1), y0.flatten(1)]
        

class transform2d:
    """ 2-D affine geometric transformation.
    
    Construction
    ------------
    ``tform = transform(A, transformType)`` creates an ``transform`` object
    given parameter array ``A`` that specifies a valid transformation for the 
    specific transformation selected.
        
    Attributes
    ----------
    dim : int
        Describes the dimension of the geometric transformation.
        
    tMatrix : array-like, float
        Geometric transfotmation matrix.
    """
    def __init__(self, transformType=None, a=None):
        """ Construct 2-D affine transformation object.
    
        Parameters
        ----------
        transformType : str, optional
            Type of geometric transformation to be applied to the image:
            
                - 'translation' : translation
                
                - 'rotation' : rotation (in radians)
                
                - 'scale' : scale
            
                - 'rigid' : translation and rotation
            
                - 'similarity' : translation, rotation, and scale
                
        a : ndarray, different shapes corresponding to the ``transformType``
            Specifies a valid geometric transformation.
            If ``transformType`` is:
            
            - 'translation' : ``a = [a1, a2]``
                >>> tform = [[ 0,  0, 0], 
                             [ 0,  0, 0],
                             [a1, a2, 1]])
            
            - 'rotation' : ``a = a1``
                >>> tform = [[ cos(a1), sin(a1), 0], 
                             [-sin(a1), cos(a1), 0],
                             [       0,       0, 1]])
                             
            - 'scale' : ``a = [a1, a2]``
                >>> tform = [[a1,  0, 0], 
                             [ 0, a2, 0],
                             [ 0,  0, 1]])
            
            - 'rigid' : ``a = [a1, a2, a3]``
                >>> tform = [[ cos(a3), sin(a3), 0], 
                             [-sin(a3), cos(a3), 0],
                             [      a1,      a2, 1]])
            
            - 'similarity' : ``a = [a1, a2, a3, a4, a5]``
                >>> tform = [[cos(a3)*a4,    sin(a3), 0], 
                             [  -sin(a3), cos(a3)*a5, 0],
                             [        a1,         a2, 1]])
        """
        self.dim = int(2)

        if transformType is None:
            transformType = 'none'
            self.tMatrix = np.identity(3, dtype='float')

        else:
            a = np.array(a, dtype='float')
            if transformType is 'translation':
                if a.size is not 2:
                    raise ValueError('Length of a must be 2.')
                self.tMatrix = np.array([[1, 0, 0], 
                                        [0, 1, 0],
                                        [a[0], a[1], 1]], dtype='float')
                                        
            elif transformType is 'rotation':
                if a.size is not 1:
                    raise ValueError('Length of a must be 1.')
                self.tMatrix = np.array([[np.cos(a), np.sin(a), 0], 
                                        [-np.sin(a), np.cos(a), 0],
                                        [0, 0, 1]], dtype='float')
                                        
            elif transformType is 'scale':
                if a.size is not 2:
                    raise ValueError('Length of a must be 2.')
                self.tMatrix = np.array([[a[0], 0, 0], 
                                        [0, a[1], 0],
                                        [0, 0, 1]], dtype='float')
                                        
            elif transformType is 'rigid':
                if a.size is not 3:
                    raise ValueError('Length of a must be 3.')
                self.tMatrix = np.array([[np.cos(a[2]), np.sin(a[2]), 0], 
                                        [-np.sin(a[2]), np.cos(a[2]), 0],
                                        [a[0], a[1], 1]], dtype='float')
                                        
            elif transformType is 'similarity':
                if a.size is not 5:
                    raise ValueError('Length of a must be 5.')
                self.tMatrix = np.array([[np.cos(a[2])*a[3], np.sin(a[2])*a[4], 0], 
                                        [-np.sin(a[2])*a[3], np.cos(a[2])*a[4], 0],
                                        [a[0]*a[3], a[1]*a[4], 1]], dtype='float')
                                        
            else:
                raise AssertionError('Unknown geometric transformation type.')
                
    def invert(self):
        """ Invert geometric transformation.
        
        Returns the inverse of the geometric transformation. ``tMatrix`` should
        not contain infinities or NaNs.
        
        Returns
        -------
        out : ndarray, float, same shape as ``tMatrix``
            Inverse of the geometric transformation, returned as an ``affine``
            geometric transformation object. 
        """
        return linalg.inv(self.tMatrix, check_finite=False)
        
    def transformForward(self, point):
        """ Apply forward geometric transformation.
        
        Applies the forward geometric transformation to the input point matrix 
        ``point`` and outputs the transformed point matrix. 
        
        Parameters
        ----------
        point : ndarray, shape (2,)
            Coordinates of points to be transformed. Must be at least 2 points.
        
        Returns
        -------
        out : ndarray, same shape as ``point``
            Transformed points.
        """ 
        point = np.array(point, dtype='float')
        return np.dot(np.hstack((point, np.ones((point.shape[0], 1)))), 
                      self.tMatrix)[:, :2]
        
    def transformInverse(self, point):
        """ Apply inverse geometric transformation.
        
        Applies the inverse geometric transformation to the input point matrix 
        ``point`` and outputs the inverse transformed point matrix. 
        
        Parameters
        ----------
        point : ndarray, shape (2,)
            Coordinates of points to be transformed. Must be at least 2 points.
        
        Returns
        -------
        out : ndarray, same shape as ``point``
            Transformed points.
        """
        point = np.array(point, dtype='float')
        return np.dot(np.hstack((point, np.ones((point.shape[0], 1)))), 
                      self.invert())[:, :2]

















