# -*- coding: utf-8 -*-
# Filename: register.py
""" Module for image registration front-end.
"""
from scipy.optimize import minimize
import geometry
import metrics
import pyramids

def reg2d(from_img, 
          from_grid, 
          to_img, 
          to_grid,
          pyramidLevels=3,
          costFunc='norm2',
          transformType='rigid',
          optimizer='powell',
          tol=None,
          bounds=None,
          options=None):
    """ Register two images.
    
    Transforms the spatially referenced floating image ``from_img`` 
    such that it is registered with the spatially referenced image 
    ``to_img``. ``from_grid`` and ``to_grid`` are spatial referencing 
    objects that describe the global spatial information like 
    position and resolution of the images.
    
    Parameters
    ----------
    from_img : ndarray, float
        Floating image.
    
    from_img : spatial referencing object for ``from_img``
    
    to_img : ndarray, float
        Reference image.
    
    to_grid : spatial referencing object for ``to_img`` 
    
    pyramidLevels : int, optional
        Levels of the Gaussian and Laplace pyramids.
    
    costFunc : metric object, optional
        Image similarity metric to be optimized during registration.
    
    transformType : str, optional
        Type of geometric transformation to be applied to the image:
            
            - 'Translation' : translation
                
            - 'Rotation' : rotation
                
            - 'Scale' : scale
            
            - 'Rigid' : translation and rotation
            
            - 'Similarity' : translation, rotation, and scale
    
    optimizer : optimization type, optional
        Method for optimizing the similarity metric.
        
            - 'Nelder-Mead'
            
            - 'Powell'
            
            - 'CG'
            
            - 'BFGS'
            
            - 'Newton-CG'
            
            - 'Anneal'
            
            - 'L-BFGS-B'
            
            - 'TNC'
            
            - 'COBYLA'
            
            - 'SLSQP'
                
    tol : float, optional
        Tolerance for termination.
        
    bounds : sequence, optional
        Bounds for variables (only for L-BFGS-B, TNC, COBYLA and SLSQP). 
        ``(min, max)`` pairs for each element in ``x``, defining the bounds on
        that parameter. 
            
    options : dict, optional
        A dictionary of solver options. All methods accept the following 
        generic options:
        
            maxiter : int 
                Maximum number of iterations to perform.
                
            disp : bool
                Set to True to print convergence messages.
        
    Returns
    -------
    out : geometric transformation object
        Converged (hopefully!) transformation variables.
    """
    # Initial values for the geometric transformations.
    if transformType   is 'translation': 
        x0 = [0, 0]
    elif transformType is 'rotation': 
        x0 = 0
    elif transformType is 'scale': 
        x0 = [1, 1]
    elif transformType is 'rigid':
        x0 = [0, 0, 0]
    elif transformType is 'similarity': 
        x0 = [0, 0, 0, 1, 1]
    else: 
        raise TypeError('Unknown geometric transform type.')
    
    # For each pyramid level:
    for m in reversed(range(pyramidLevels + 1)):
        print m
        _from_img = pyramids.reduce(from_img, m)
        _to_img = pyramids.reduce(to_img, m)
        _from_grid = geometry.grid2d(_from_img.shape, from_grid.limits)
        _to_grid = geometry.grid2d(_to_img.shape, to_grid.limits)
        
        # Call constructor for the cost function:
        if costFunc is 'mutualInformation':
            costFunc = metrics.mutualInformation(_from_img, 
                                                _from_grid, 
                                                _to_img, 
                                                _to_grid,
                                                transformType)
        if costFunc is 'crossCorrelation':
            costFunc = metrics.crossCorrelation(_from_img, 
                                               _from_grid, 
                                               _to_img, 
                                               _to_grid,
                                               transformType)
        if costFunc is 'norm2':
            costFunc = metrics.norm2(_from_img, 
                                    _from_grid, 
                                    _to_img, 
                                    _to_grid,
                                    transformType)
        if costFunc is 'norm1':
            costFunc = metrics.norm1(_from_img, 
                                    _from_grid, 
                                    _to_img, 
                                    _to_grid,
                                    transformType)

        # Solve minimization problem.
        res = minimize(costFunc.calc, 
                       x0, 
                       method=optimizer, 
                       tol=tol,
                       bounds=bounds,
                       options=options)
        x0 = res.x
    return geometry.transform2d(transformType, x0)
    
    
    
    
    
    