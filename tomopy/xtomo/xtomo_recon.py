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
from tomopy.algorithms.recon.sirt import sirt
from tomopy.algorithms.recon.gridrec import Gridrec
from tomopy.algorithms.recon.mlem import mlem
from tomopy.algorithms.recon.pml import pml

# Import helper functons in the package.
from tomopy.algorithms.recon.diagnose_center import diagnose_center
from tomopy.algorithms.recon.optimize_center import optimize_center
from tomopy.algorithms.recon.upsample import upsample2d, upsample3d

# --------------------------------------------------------------------

def _diagnose_center(self, dir_path=None, slice_no=None,
		     center_start=None, center_end=None, center_step=None, 
		     mask=True, ratio=1, 
                     dtype='float32', data_min=None, data_max=None):
	
    # Dimensions:
    num_slices = self.data.shape[1]
    num_pixels = self.data.shape[2]

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
    dtype, data_max, data_min = diagnose_center(
                     self.data, self.theta, dir_path, slice_no, 
                     center_start, center_end, center_step, mask, ratio, 
                     dtype, data_min, data_max)

    # Update log.
    self.logger.debug("diagnose_center: dir_path: " + str(dir_path))
    self.logger.debug("diagnose_center: slice_no: " + str(slice_no))
    self.logger.debug("diagnose_center: center_start: " + str(center_start))
    self.logger.debug("diagnose_center: center_end: " + str(center_end))
    self.logger.debug("diagnose_center: center_step: " + str(center_step))
    self.logger.debug("diagnose_center: mask: " + str(mask))
    self.logger.debug("diagnose_center: ratio: " + str(ratio))
    self.logger.debug("diagnose_center: dtype: " + str(dtype))
    self.logger.debug("diagnose_center: data_max: " + str(data_max))
    self.logger.debug("diagnose_center: data_min: " + str(data_min))
    self.logger.info("diagnose_center [ok]")

# --------------------------------------------------------------------

def _optimize_center(self, slice_no=None, center_init=None, 
                     tol=0.5, overwrite=True, mask=True, ratio=1):
                    
    # Dimensions:
    num_slices = self.data.shape[1]
    num_pixels = self.data.shape[2]

    # Set default parameters.
    if slice_no is None:
        slice_no = num_slices/2
    
    if center_init is None:
        center_init = num_pixels/2
                          
    # Make check.                      
    if not isinstance(center_init, np.float32):
        center_init = np.array(center_init, dtype='float32')

    # All set, give me center now.
    center = optimize_center(self.data, self.theta, slice_no, 
                              center_init, tol, mask, ratio)
    
    # Update log.
    self.logger.debug("optimize_center: slice_no: " + str(slice_no))
    self.logger.debug("optimize_center: center_init: " + str(center_init))
    self.logger.debug("optimize_center: tol: " + str(tol))
    self.logger.debug("optimize_center: mask: " + str(mask))
    self.logger.debug("optimize_center: ratio: " + str(ratio))
    self.logger.info("optimize_center [ok]")
    
    # Update returned values.
    if overwrite: self.center = center
    else: return center
	    
# --------------------------------------------------------------------

def _upsample2d(self, level=1,
                num_cores=None, chunk_size=None,
                overwrite=True):

    # Check input.
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data_recon = upsample2d(self.data_recon, level)
    
    # Update log.
    self.logger.debug("upsample2d: level: " + str(level))
    self.logger.info("upsample2d [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon
	    
# --------------------------------------------------------------------

def _upsample3d(self, level=1,
                num_cores=None, chunk_size=None,
                overwrite=True):

    # Check input.
    if not isinstance(level, np.int32):
        level = np.array(level, dtype='int32')

    data_recon = upsample3d(self.data_recon, level)
    
    # Update log.
    self.logger.debug("upsample3d: level: " + str(level))
    self.logger.info("upsample3d [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon 
    else: return data_recon
    
# --------------------------------------------------------------------
    
def _art(self, emission=False, 
         iters=1, num_grid=None, 
         init_matrix=None, overwrite=True):
    
    # Dimensions:
    num_pixels = self.data.shape[2]
        
    # This works with radians.
    if np.max(self.theta) > 90: # then theta is obviously in radians.
        self.theta *= np.pi/180

    # Pad data.
    if emission:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=0)
    else:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=1)
        data = -np.log(data);
    
    # Adjust center according to padding.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[2]/2
    center = self.center + (data.shape[2]-num_pixels)/2.

    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
    if init_matrix is None:   
        init_matrix = np.zeros((data.shape[1], num_grid, num_grid), dtype='float32')
        
    # Check inputs.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(self.theta, np.float32):
        theta = np.array(self.theta, dtype='float32')

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
    self.logger.debug("art: emission: " + str(emission))
    self.logger.debug("art: iters: " + str(iters))
    self.logger.debug("art: center: " + str(center))
    self.logger.debug("art: num_grid: " + str(num_grid))
    self.logger.info("art [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon
    
# --------------------------------------------------------------------

def _sirt(self, emission=False, 
         iters=1, num_grid=None, 
         init_matrix=None, overwrite=True):
    
    # Dimensions:
    num_pixels = self.data.shape[2]
        
    # This works with radians.
    if np.max(self.theta) > 90: # then theta is obviously in radians.
        self.theta *= np.pi/180

    # Pad data.
    if emission:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=0)
    else:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=1)
        data = -np.log(data);
    
    # Adjust center according to padding.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[2]/2
    center = self.center + (data.shape[2]-num_pixels)/2.

    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
    if init_matrix is None:   
        init_matrix = np.zeros((data.shape[1], num_grid, num_grid), dtype='float32')
        
    # Check inputs.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(self.theta, np.float32):
        theta = np.array(self.theta, dtype='float32')

    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype='int32')

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype='int32')
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype='float32', copy=False)

    # Initialize and perform reconstruction.
    data_recon = sirt(data, theta, center, num_grid, iters, init_matrix)
    
    # Update log.
    self.logger.debug("sirt: emission: " + str(emission))
    self.logger.debug("sirt: iters: " + str(iters))
    self.logger.debug("sirt: center: " + str(center))
    self.logger.debug("sirt: num_grid: " + str(num_grid))
    self.logger.info("sirt [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon
    
# --------------------------------------------------------------------
    
def _mlem(self, emission=False, 
         iters=1, num_grid=None, 
         init_matrix=None, overwrite=True):

    # Dimensions:
    num_pixels = self.data.shape[2]
        
    # This works with radians.
    if np.max(self.theta) > 90: # then theta is obviously in radians.
        self.theta *= np.pi/180

    # Pad data.
    if emission:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=0)
    else:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=1)
        data = -np.log(data);

    # Adjust center according to padding.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[2]/2
    center = self.center + (data.shape[2]-num_pixels)/2.
   
    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
    if init_matrix is None:
        init_matrix = np.ones((data.shape[1], num_grid, num_grid), dtype='float32')

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(self.theta, np.float32):
        theta = np.array(self.theta, dtype='float32')

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
    self.logger.debug("mlem: emission: " + str(emission))
    self.logger.debug("mlem: iters: " + str(iters))
    self.logger.debug("mlem: center: " + str(center))
    self.logger.debug("mlem: num_grid: " + str(num_grid))
    self.logger.info("mlem [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon
    
# --------------------------------------------------------------------
    
def _pml(self, emission=False, 
         iters=1, num_grid=None, beta=1,
         init_matrix=None, overwrite=True):

    # Dimensions:
    num_pixels = self.data.shape[2]
        
    # This works with radians.
    if np.max(self.theta) > 90: # then theta is obviously in radians.
        self.theta *= np.pi/180

    # Pad data.
    if emission:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=0)
    else:
        data, white, dark = self.apply_padding(overwrite=False, pad_val=1)
        data = -np.log(data);

    # Adjust center according to padding.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[2]/2
    center = self.center + (data.shape[2]-num_pixels)/2.
   
    # Set default parameters.
    if num_grid is None or num_grid > num_pixels:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
    if init_matrix is None:
        init_matrix = np.ones((data.shape[1], num_grid, num_grid), dtype='float32')

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype='float32', copy=False)

    if not isinstance(self.theta, np.float32):
        theta = np.array(self.theta, dtype='float32')

    if not isinstance(center, np.float32):
        center = np.array(center, dtype='float32')
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype='int32')

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype='int32')
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype='float32', copy=False)
    
    # Initialize and perform reconstruction.
    data_recon = pml(data, theta, center, num_grid, iters, beta, init_matrix)

    # Update log.
    self.logger.debug("pml: emission: " + str(emission))
    self.logger.debug("pml: iters: " + str(iters))
    self.logger.debug("pml: center: " + str(center))
    self.logger.debug("pml: num_grid: " + str(num_grid))
    self.logger.debug("pml: beta: " + str(beta))
    self.logger.info("pml [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------
    
def _gridrec(self, overwrite=True, *args, **kwargs):

    # Check input.
    if not hasattr(self, 'center'):
        self.center = self.data.shape[2]/2
    if not isinstance(self.center, np.float32):
        self.center = np.array(self.center, dtype='float32')
    
    # Initialize and perform reconstruction.    
    recon = Gridrec(self.data, *args, **kwargs)
    data_recon = recon.reconstruct(self.data, self.center, self.theta)
    
    # Update provenance and log.
    self.logger.info("gridrec [ok]")
    
    # Update returned values.
    if overwrite: self.data_recon = data_recon
    else: return data_recon

# --------------------------------------------------------------------

# Hook all these methods to TomoPy.
setattr(XTomoDataset, 'diagnose_center', _diagnose_center)
setattr(XTomoDataset, 'optimize_center', _optimize_center)
setattr(XTomoDataset, 'upsample2d', _upsample2d)
setattr(XTomoDataset, 'upsample3d', _upsample3d)
setattr(XTomoDataset, 'art', _art)
setattr(XTomoDataset, 'sirt', _sirt)
setattr(XTomoDataset, 'gridrec', _gridrec)
setattr(XTomoDataset, 'mlem', _mlem)
setattr(XTomoDataset, 'pml', _pml)

