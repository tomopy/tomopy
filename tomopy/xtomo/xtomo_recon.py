# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers to 
hook the methods in recon package to X-ray 
absorption tomography data object.
"""

import numpy as np
import os
import shutil

# Import main TomoPy object.
from tomopy.xtomo.xtomo_dataset import XTomoDataset

# Import available reconstruction functons in the package.
from tomopy.algorithms.recon.art import art
from tomopy.algorithms.recon.gridrec import Gridrec
from tomopy.algorithms.recon.mlem import mlem

# Import helper functons in the package.
from tomopy.algorithms.recon.diagnose_center import diagnose_center
from tomopy.algorithms.recon.optimize_center import optimize_center
from tomopy.algorithms.recon.upsample import upsample2d, upsample3d

# --------------------------------------------------------------------

def _diagnose_center(xtomo, dir_path=None, slice_no=None,
		     center_start=None, center_end=None, center_step=None, 
		     mask=True, ratio=1):
	
    # Dimensions:
    num_slices = xtomo.data.shape[1]
    num_pixels = xtomo.data.shape[2]

    # Set default parameters.
    if dir_path is None: # Create one at at data location for output images.
        dir_path = 'tmp/center_diagnose/'
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Define diagnose region.
    if slice_no is None:
        slice_no = num_slices/2
        
    if center_start is None: # Search +-20 pixels near center
        center_start = (num_pixels/2)-20
    if center_end is None: 
        center_end = (num_pixels/2)+20
    if center_step is None:
        center_step = 1

    # Call function.
    diagnose_center(xtomo.data, xtomo.theta, dir_path, slice_no, 
                     center_start, center_end, center_step, mask, ratio)

    # Update log.
    xtomo.logger.debug("diagnose_center: dir_path: " + str(dir_path))
    xtomo.logger.debug("diagnose_center: slice_no: " + str(slice_no))
    xtomo.logger.debug("diagnose_center: center_start: " + str(center_start))
    xtomo.logger.debug("diagnose_center: center_end: " + str(center_end))
    xtomo.logger.debug("diagnose_center: center_step: " + str(center_step))
    xtomo.logger.debug("diagnose_center: mask: " + str(mask))
    xtomo.logger.debug("diagnose_center: ratio: " + str(ratio))
    xtomo.logger.info("diagnose_center [ok]")

# --------------------------------------------------------------------

def _optimize_center(xtomo, slice_no=None, center_init=None, 
                     tol=0.5, overwrite=True, mask=True, ratio=1):
                    
    # Dimensions:
    num_slices = xtomo.data.shape[1]
    num_pixels = xtomo.data.shape[2]

    # Set default parameters.
    if slice_no is None:
        slice_no = num_slices/2
    
    if center_init is None:
        center_init = num_pixels/2
                          
    # Make check.                      
    if not isinstance(center_init, np.float32):
        center_init = np.array(center_init, dtype='float32')

    # All set, give me center now.
    center = optimize_center(xtomo.data, xtomo.theta, slice_no, 
                              center_init, tol, mask, ratio)
    
    # Update log.
    xtomo.logger.debug("optimize_center: slice_no: " + str(slice_no))
    xtomo.logger.debug("optimize_center: center_init: " + str(center_init))
    xtomo.logger.debug("optimize_center: tol: " + str(tol))
    xtomo.logger.debug("optimize_center: mask: " + str(mask))
    xtomo.logger.debug("optimize_center: ratio: " + str(ratio))
    xtomo.logger.info("optimize_center [ok]")
    
    # Update returned values.
    if overwrite: xtomo.center = center
    else: return center
	    
# --------------------------------------------------------------------

def _upsample2d(xtomo, level=1,
                num_cores=None, chunk_size=None,
                overwrite=True):

    # Check input.
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data_recon = upsample2d(xtomo.data_recon, level)
    
    # Update log.
    xtomo.logger.debug("upsample2d: level: " + str(level))
    xtomo.logger.info("upsample2d [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon
	    
# --------------------------------------------------------------------

def _upsample3d(xtomo, level=1,
                num_cores=None, chunk_size=None,
                overwrite=True):

    # Check input.
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data_recon = upsample3d(xtomo.data_recon, level)
    
    # Update log.
    xtomo.logger.debug("upsample3d: level: " + str(level))
    xtomo.logger.info("upsample3d [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon 
    else: return data_recon
    
# --------------------------------------------------------------------
    
def _art(xtomo, iters=1, num_grid=None, init_matrix=None, overwrite=True):
    
    # Dimensions:
    num_pixels = xtomo.data.shape[2]
        
    # This works with radians.
    if np.max(xtomo.theta) > 90: # then theta is obviously in radians.
        xtomo.theta *= np.pi/180

    # Pad data first.
    data = xtomo.apply_padding(overwrite=False)
    data = -np.log(data);
    
    # Adjust center according to padding.
    if not hasattr(xtomo, 'center'):
        xtomo.center = xtomo.data.shape[2]/2
    center = xtomo.center + (data.shape[2]-num_pixels)/2.

    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
        xtomo.logger.debug("art: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:   
        init_matrix = np.zeros((data.shape[1], num_grid, num_grid), dtype='float32')
        xtomo.logger.debug("art: init_matrix set to zeros [ok]")
        
        
    # Check inputs.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(xtomo.theta, np.float32):
        theta = np.array(xtomo.theta, dtype='float32')

    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype='int32')

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype='int32')
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype='float32', copy=False)

    # Initialize and perform reconstruction.
    data_recon = art(data, theta, center, num_grid, iters, init_matrix)
    
    # Update log.
    xtomo.logger.debug("art: iters: " + str(iters))
    xtomo.logger.debug("art: center: " + str(center))
    xtomo.logger.debug("art: num_grid: " + str(num_grid))
    xtomo.logger.info("art [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon
    
# --------------------------------------------------------------------
    
def _mlem(xtomo, iters=1, num_grid=None, init_matrix=None, overwrite=True):

    # Dimensions:
    num_pixels = xtomo.data.shape[2]
        
    # This works with radians.
    if np.max(xtomo.theta) > 90: # then theta is obviously in radians.
        xtomo.theta *= np.pi/180

    # Pad data first.
    data = xtomo.apply_padding(overwrite=False)
    data = np.abs(-np.log(data));

    # Adjust center according to padding.
    if not hasattr(xtomo, 'center'):
        xtomo.center = xtomo.data.shape[2]/2
    center = xtomo.center + (data.shape[2]-num_pixels)/2.
   
    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
        xtomo.logger.debug("mlem: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:
        init_matrix = np.ones((data.shape[1], num_grid, num_grid), dtype='float32')
        xtomo.logger.debug("mlem: init_matrix set to ones [ok]")
    

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(xtomo.theta, np.float32):
        theta = np.array(xtomo.theta, dtype='float32')

    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype='int32')

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype='int32')
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype='float32', copy=False)
    
    # Initialize and perform reconstruction.
    data_recon = mlem(data, theta, center, num_grid, iters, init_matrix)

    # Update log.
    xtomo.logger.debug("mlem: iters: " + str(iters))
    xtomo.logger.debug("mlem: center: " + str(center))
    xtomo.logger.debug("mlem: num_grid: " + str(num_grid))
    xtomo.logger.info("mlem [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------
    
def _gridrec(xtomo, overwrite=True, *args, **kwargs):

    # Check input.
    if not hasattr(xtomo, 'center'):
        xtomo.center = xtomo.data.shape[2]/2
    if not isinstance(xtomo.center, np.float32):
        xtomo.center = np.array(xtomo.center, dtype='float32')
    
    # Initialize and perform reconstruction.    
    recon = Gridrec(xtomo.data, *args, **kwargs)
    data_recon = recon.reconstruct(xtomo.data, xtomo.center, xtomo.theta)
    
    # Update provenance and log.
    xtomo.logger.info("gridrec [ok]")
    
    # Update returned values.
    if overwrite: xtomo.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

# Hook all these methods to TomoPy.
setattr(XTomoDataset, 'diagnose_center', _diagnose_center)
setattr(XTomoDataset, 'optimize_center', _optimize_center)
setattr(XTomoDataset, 'upsample2d', _upsample2d)
setattr(XTomoDataset, 'upsample3d', _upsample3d)
setattr(XTomoDataset, 'art', _art)
setattr(XTomoDataset, 'gridrec', _gridrec)
setattr(XTomoDataset, 'mlem', _mlem)
