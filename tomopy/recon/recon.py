# -*- coding: utf-8 -*-
"""
This module containes a set of thin wrappers for the other
modules in recon package to link them to TomoPy session.
Each wrapper first checks the arguments and then calls the method.
The linking is mostly realized through the multiprocessing module.
"""
import numpy as np
import os
import shutil
import logging
logger = logging.getLogger("tomopy")

# Import main TomoPy object.
from tomopy.dataio.reader import Session

# Import available reconstruction functons in the package.
from art import Art
from gridrec import Gridrec
from mlem import Mlem

# Import helper functons in the package.
from diagnose_center import _diagnose_center
from optimize_center import _optimize_center



def diagnose_center(tomo, dir_path=None, slice_no=None,
		    center_start=None, center_end=None, center_step=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("diagnose rotation center " +
                       "(data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        logger.warning("diagnose rotation center " +
                       "(angles missing) [bypassed]")
        return
    

    # Set default parameters.
    if dir_path is None: # Create one at at data location for output images.
        dir_path = os.path.dirname(tomo.file_name) + '/data_center/'
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        logger.debug("data_center: dir_path set " +
                       "to ", dir_path, " [ok]")
    
    # Define diagnose region.
    if slice_no is None:
        slice_no = tomo.data.shape[1] / 2
        
    if center_start is None: # Search +-20 pixels near center
        center_start = (tomo.data.shape[2] / 2) - 20
    if center_end is None: 
        center_end = (tomo.data.shape[2] / 2) + 20
    if center_step is None:
        center_step = 1
        

    # Call function.
    _diagnose_center(tomo.data, tomo.theta, dir_path, slice_no, 
                     center_start, center_end, center_step)
    

    # Update provenance and log.
    tomo.provenance['diagnose_center'] = {'dir_path':dir_path,
                                          'slice_no':slice_no,
                                   	      'center_start':center_start,
                                   	      'center_end':center_end,
                                   	      'center_step':center_step}
    logger.debug("data_center directory create [ok]")


def optimize_center(tomo, slice_no=None, center_init=None, tol=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("optimize rotation center " +
                       "(data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        logger.warning("optimize rotation center " +
                       "(angles missing) [bypassed]")
        return


    # Set default parameters.
    if slice_no is None: # Use middle slice.
        slice_no = tomo.data.shape[1] / 2
        logger.debug("optimize_center: slice_no is " +
                       "set to " + str(slice_no) + " [ok]")
    
    if center_init is None: # Use middle point of the detector area.
        center_init = tomo.data.shape[2] / 2
        logger.debug("optimize_center: center_init " +
                       "is set to " + str(center_init) + " [ok]")
    
    if tol is None:
        tol = 0.5
        logger.debug("optimize_center: tol is set " +
                       "to " + str(tol) + " [ok]")
    

    # All set, give me center now.
    tomo.center = _optimize_center(tomo.data, tomo.theta, 
                                   slice_no, center_init, tol)
    

    # Update provenance and log.
    tomo.provenance['optimize_center'] = {'theta':tomo.theta,
                                          'slice_no':slice_no,
                                   	      'center_init':center_init,
                                   	      'tol':tol}
    logger.info("optimize rotation center [ok]")
    

    
def art(tomo, iters=None, num_grid=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("art (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        logger.warning("art (angles missing) [bypassed]")
        return

    if not hasattr(tomo, 'center'):
        logger.warning("art (center missing) [bypassed]")
        return
        
    # Set default parameters.
    if iters is None:
        iters = 1
        logger.debug("art: iters set to 1 [ok]")

    if num_grid is None:
        num_grid = tomo.data.shape[2]
        logger.debug("art: num_grid set to " + str(num_grid) + " [ok]")


    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)
    
    # This works with radians.
    tomo.theta *= np.pi/180

    # Intensity to line integral according to Beer's Law.
#    data = -np.log(tomo.data)
    data = tomo.data

    # Initialize and perform reconstruction.    
    recon = Art(data, num_grid)
    tomo.data_recon = recon.reconstruct(iters, tomo.center, tomo.theta)
    
    
    # Update provenance and log.
    tomo.provenance['art'] =  {'iters':iters}
    tomo.FLAG_DATA_RECON = True
    logger.info("art reconstruction [ok]")
    
    
    
    
    
def mlem(tomo, iters=None, num_grid=None):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("mlem (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        logger.warning("mlem (angles missing) [bypassed]")
        return
        
    if not hasattr(tomo, 'center'):
        logger.warning("mlem (center missing) [bypassed]")
        return
        
        
    # Set default parameters.
    if iters is None:
        iters = 1
        logger.debug("mlem: iters set to " + str(iters) + " [ok]")
    
    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)

    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)
    
    if num_grid is None:
        num_grid = tomo.data.shape[2]
        logger.debug("mlem: num_grid set to " + str(num_grid) + " [ok]")

    # This works with radians.
    tomo.theta *= np.pi/180

    # Intensity to line integral according to Beer's Law.
#    data = -np.log(tomo.data)
    data = tomo.data


    # Initialize and perform reconstruction.    
    recon = Mlem(data, num_grid)
    tomo.data_recon = recon.reconstruct(iters, tomo.center, tomo.theta)
    
    
    # Update provenance and log.
    tomo.provenance['mlem'] = {'iters':iters}
    tomo.FLAG_DATA_RECON = True
    logger.info("mlem reconstruction [ok]")


    
def gridrec(tomo, *args, **kwargs):
    # Make checks first. 
    if not tomo.FLAG_DATA:
        logger.warning("gridrec (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        logger.warning("gridrec (angles missing) [bypassed]")
        return

    if not hasattr(tomo, 'center'):
        logger.warning("gridrec (center missing) [bypassed]")
        return


    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)

    
    # Initialize and perform reconstruction.    
    recon = Gridrec(tomo.data, *args, **kwargs)
    tomo.data_recon = recon.reconstruct(tomo.data, tomo.center, tomo.theta)
    
    # Update provenance and log.
    tomo.provenance['gridrec'] = (args, kwargs)
    tomo.FLAG_DATA_RECON = True
    logger.info("gridrec reconstruction [ok]")



# Hook all these methods to TomoPy.
setattr(Session, 'diagnose_center', diagnose_center)
setattr(Session, 'optimize_center', optimize_center)
setattr(Session, 'art', art)
setattr(Session, 'gridrec', gridrec)
setattr(Session, 'mlem', mlem)

# Use original function docstrings for the wrappers.
diagnose_center.__doc__ = _diagnose_center.__doc__
optimize_center.__doc__ = _optimize_center.__doc__
art.__doc__ = Art.__doc__
gridrec.__doc__ = Gridrec.__doc__
mlem.__doc__ = Mlem.__doc__