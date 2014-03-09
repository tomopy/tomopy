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

# Import main TomoPy object.
from tomopy.dataio.reader import Session

# Import available reconstruction functons in the package.
from art import _art
from gridrec import Gridrec
from mlem import _mlem

# Import helper functons in the package.
from diagnose_center import _diagnose_center
from optimize_center import _optimize_center
from upsample import _upsample


# --------------------------------------------------------------------

def diagnose_center(tomo, dir_path=None, slice_no=None,
		    center_start=None, center_end=None, center_step=None):

    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("diagnose rotation center " +
                            "(data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("diagnose rotation center " +
                            "(angles missing) [bypassed]")
        return
    

    # Set default parameters.
    if dir_path is None: # Create one at at data location for output images.
        dir_path = os.path.dirname(tomo.file_name) + '/data_center/'
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        tomo.logger.debug("data_center: dir_path set " +
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
    tomo.logger.debug("data_center directory create [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon

# --------------------------------------------------------------------

def optimize_center(tomo, slice_no=None, center_init=None, tol=None, overwrite=True):

    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("optimize rotation center " +
                            "(data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("optimize rotation center " +
                            "(angles missing) [bypassed]")
        return


    # Set default parameters.
    if slice_no is None: # Use middle slice.
        slice_no = tomo.data.shape[1] / 2
        tomo.logger.debug("optimize_center: slice_no is " +
                          "set to " + str(slice_no) + " [ok]")
    
    if center_init is None: # Use middle point of the detector area.
        center_init = tomo.data.shape[2] / 2
        tomo.logger.debug("optimize_center: center_init " +
                          "is set to " + str(center_init) + " [ok]")
    
    if tol is None:
        tol = 0.5
        tomo.logger.debug("optimize_center: tol is set " +
                          "to " + str(tol) + " [ok]")
    

    # All set, give me center now.
    center = _optimize_center(tomo.data, tomo.theta, slice_no, center_init, tol)
    

    # Update provenance and log.
    tomo.provenance['optimize_center'] = {'theta':tomo.theta,
                                          'slice_no':slice_no,
                                   	      'center_init':center_init,
                                   	      'tol':tol}
    tomo.logger.info("optimize rotation center [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon
	    
# --------------------------------------------------------------------

def upsample(tomo, level=None,
             num_cores=None, chunk_size=None,
             overwrite=True):
    
    # Set default parameters.
    if level is None:
        level = 1
        tomo.logger.debug("upsample: level is " +
                          "set to " + str(level) + " [ok]")


    # Check inputs.
    if not isinstance(tomo.data_recon, np.float32):
        tomo.data_recon = np.array(tomo.data_recon, dtype=np.float32, copy=False)

    if not isinstance(level, np.int32):
        level = np.array(level, dtype=np.int32, copy=False)

    data_recon = _upsample(tomo.data_recon, level)
    
    ## Distribute jobs.
    #_func = _upsample
    #_args = ()
    #_axis = 1 # Slice axis
    #tomo.data = distribute_jobs(tomo.data, _func, _args, _axis,
    #                            num_cores, chunk_size)
    
    # Update provenance and log.
    tomo.provenance['upsample'] = {'level':level}
    tomo.logger.info("data upsampling [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon
    
# --------------------------------------------------------------------
    
def art(tomo, iters=None, num_grid=None, init_matrix=None, overwrite=True):

    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("art (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("art (angles missing) [bypassed]")
        return

    if not hasattr(tomo, 'center'):
        tomo.logger.warning("art (center missing) [bypassed]")
        return
        

    # Set default parameters.
    if iters is None:
        iters = 1
        tomo.logger.debug("art: iters set to " + str(iters) + " [ok]")

    if num_grid is None or num_grid > tomo.data.shape[2]:
        num_grid = np.floor(tomo.data.shape[2] / np.sqrt(2))
        tomo.logger.debug("art: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:   
        init_matrix = np.zeros((tomo.data.shape[1], num_grid, num_grid), dtype='float32')
        tomo.logger.debug("mlem: init_matrix set to zeros [ok]")
        
        
    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        tomo.theta *= np.pi/180
    
    # For transmission data use Beer's Law.
    data = -np.log(tomo.data);

    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)

    # Initialize and perform reconstruction.
    data_recon = _art(data, tomo.theta, tomo.center, num_grid, iters, init_matrix)
    
    
    # Update provenance and log.
    tomo.provenance['art'] =  {'iters':iters}
    tomo.FLAG_DATA_RECON = True
    tomo.logger.info("art reconstruction [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon
    
# --------------------------------------------------------------------
    
def mlem(tomo, iters=None, num_grid=None, init_matrix=None, overwrite=True):
    
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("mlem (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("mlem (angles missing) [bypassed]")
        return
        
    if not hasattr(tomo, 'center'):
        tomo.logger.warning("mlem (center missing) [bypassed]")
        return
        
        
    # Set default parameters.
    if iters is None:
        iters = 1
        tomo.logger.debug("mlem: iters set to " + str(iters) + " [ok]")

    if num_grid is None or num_grid > tomo.data.shape[2]:
        num_grid = np.floor(tomo.data.shape[2] / np.sqrt(2))
        tomo.logger.debug("mlem: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:
        init_matrix = np.ones((tomo.data.shape[1], num_grid, num_grid), dtype='float32')
        tomo.logger.debug("mlem: init_matrix set to ones [ok]")
        
        
    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        tomo.theta *= np.pi/180
    
    # For transmission data use Beer's Law.
    data = np.abs(-np.log(tomo.data));

    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)

    # Initialize and perform reconstruction.
    data_recon = _mlem(data, tomo.theta, tomo.center, num_grid, iters, init_matrix)

    
    # Update provenance and log.
    tomo.provenance['mlem'] = {'iters':iters}
    tomo.FLAG_DATA_RECON = True
    tomo.logger.info("mlem reconstruction [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon
	    
# --------------------------------------------------------------------
    
def mlem_multilevel(tomo, iters=None, num_grid=None, level=None, init_matrix=None, overwrite=True):
    
    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("mlem_multilevel (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("mlem_multilevel (angles missing) [bypassed]")
        return
        
    if not hasattr(tomo, 'center'):
        tomo.logger.warning("mlem_multilevel (center missing) [bypassed]")
        return
        
        
    # Set default parameters.
    if iters is None:
        iters = 1
        tomo.logger.debug("mlem_multilevel: iters set to " + str(iters) + " [ok]")
        
    if level is None:
        level = 2
        tomo.logger.debug("mlem_multilevel: iters set to " + str(level) + " [ok]")

    if num_grid is None or num_grid > tomo.data.shape[2]:
        num_grid = np.floor(tomo.data.shape[2] / np.sqrt(2)) / np.power(2, level)
        tomo.logger.debug("mlem_multilevel: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:  
        init_matrix = np.ones((tomo.data.shape[1], num_grid, num_grid), dtype='float32') 
        tomo.logger.debug("mlem_multilevel: init_matrix set to ones [ok]")
            

    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        tomo.theta *= np.pi/180
    
    # For transmission data use Beer's Law.
    tomo.data = np.abs(-np.log(tomo.data));


    # Check again.
    if not isinstance(tomo.data, np.float32):
        tomo.data = np.array(tomo.data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        tomo.theta = np.array(tomo.theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        tomo.center = np.array(tomo.center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)
    
    data = tomo.downsample(level=level, overwrite=False)    
#     tomo.data_to_tiff('test_')
    	
    center = np.array(tomo.center/np.power(2, level), dtype=np.float32, copy=False)
#     print center, num_grid, tomo.data.shape
	
    data_recon = _mlem(data, tomo.theta, tomo.center, num_grid, iters, init_matrix)
    tomo.FLAG_DATA_RECON = True
#     	tomo.recon_to_tiff('test_')
    	
#     data_recon = init_matrix.copy()
    
    # Update provenance and log.
    tomo.provenance['mlem'] = {'iters':iters}
    tomo.FLAG_DATA_RECON = True
    tomo.logger.info("mlem reconstruction [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon

# --------------------------------------------------------------------
    
def gridrec(tomo, overwrite=True, *args, **kwargs):

    # Make checks first. 
    if not tomo.FLAG_DATA:
        tomo.logger.warning("gridrec (data missing) [bypassed]")
        return
   
    if not tomo.FLAG_THETA:
        tomo.logger.warning("gridrec (angles missing) [bypassed]")
        return

    if not hasattr(tomo, 'center'):
        tomo.logger.warning("gridrec (center missing) [bypassed]")
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
    data_recon = recon.reconstruct(tomo.data, tomo.center, tomo.theta)
    
    
    # Update provenance and log.
    tomo.provenance['gridrec'] = (args, kwargs)
    tomo.FLAG_DATA_RECON = True
    tomo.logger.info("gridrec reconstruction [ok]")
    
    # Update returned values.
    if overwrite:
	    tomo.data_recon = data_recon
    else:
	    return data_recon

# --------------------------------------------------------------------

# Hook all these methods to TomoPy.
setattr(Session, 'diagnose_center', diagnose_center)
setattr(Session, 'optimize_center', optimize_center)
setattr(Session, 'upsample', upsample)
setattr(Session, 'art', art)
setattr(Session, 'gridrec', gridrec)
setattr(Session, 'mlem', mlem)
setattr(Session, 'mlem_multilevel', mlem_multilevel)

# Use original function docstrings for the wrappers.
diagnose_center.__doc__ = _diagnose_center.__doc__
upsample.__doc__ = _upsample.__doc__
diagnose_center.__doc__ = _diagnose_center.__doc__
art.__doc__ = _art.__doc__
gridrec.__doc__ = Gridrec.__doc__
mlem.__doc__ = _mlem.__doc__