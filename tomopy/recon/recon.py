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
from upsample import _upsample2d, _upsample3d
from tomopy.preprocess.downsample import _downsample2d, _downsample3d


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

# --------------------------------------------------------------------

def optimize_center(tomo, slice_no=None, center_init=None, 
                    tol=None, overwrite=True):

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
                          
                          
    if not isinstance(center_init, np.float32):
        center_init = np.array(center_init, dtype=np.float32, copy=False)

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
	tomo.center = center
    else:
	return center
	    
# --------------------------------------------------------------------

def upsample2d(tomo, level=None,
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

    data_recon = _upsample2d(tomo.data_recon, level)
    
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

def upsample3d(tomo, level=None,
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

    data_recon = _upsample3d(tomo.data_recon, level)
    
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

        
    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        theta = tomo.theta * np.pi/180

    # Pad data first.
    data = tomo.apply_padding(overwrite=False)
    data = -np.log(data);
    
    # Adjust center according to padding.
    center = tomo.center + (data.shape[2]-tomo.data.shape[2])/2.
    

    # Set default parameters.
    if iters is None:
        iters = 1
        tomo.logger.debug("art: iters set to " + str(iters) + " [ok]")

    if num_grid is None or num_grid > tomo.data.shape[2]:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
        tomo.logger.debug("art: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:   
        init_matrix = np.zeros((data.shape[1], num_grid, num_grid), dtype='float32')
        tomo.logger.debug("art: init_matrix set to zeros [ok]")
        

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype=np.float32, copy=False)

    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype=np.float32, copy=False)

    if not isinstance(center, np.float32):
        center = np.array(center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)

    # Initialize and perform reconstruction.
    data_recon = _art(data, theta, center, num_grid, iters, init_matrix)
    
    
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
        

    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        theta = tomo.theta * np.pi/180

    # Pad data first.
    data = tomo.apply_padding(overwrite=False)
    data = np.abs(-np.log(data));

    # Adjust center according to padding.
    center = tomo.center + (data.shape[2]-tomo.data.shape[2])/2.

        
    # Set default parameters.
    if iters is None:
        iters = 1
        tomo.logger.debug("mlem: iters set to " + str(iters) + " [ok]")

    if num_grid is None or num_grid > tomo.data.shape[2]:
        num_grid = np.floor(data.shape[2] / np.sqrt(2))
        tomo.logger.debug("mlem: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:
        init_matrix = np.ones((data.shape[1], num_grid, num_grid), dtype='float32')
        tomo.logger.debug("mlem: init_matrix set to ones [ok]")
    

    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype=np.float32, copy=False)

    if not isinstance(theta, np.float32):
        theta = np.array(theta, dtype=np.float32, copy=False)

    if not isinstance(center, np.float32):
        center = np.array(center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)

    # Initialize and perform reconstruction.
    data_recon = _mlem(data, theta, center, num_grid, iters, init_matrix)

    
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
    
def mlem_multilevel(tomo, iters=None, num_grid=None, level=None, 
                    init_matrix=None, overwrite=True):
    
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




    # This works with radians.
    if np.max(tomo.theta) > 90: # then theta is obviously in radians.
        theta = tomo.theta * np.pi/180
        
    # Level size.
    lvl = np.power(2, level)

    # Pad data first.
    num_padded_pixels = tomo.data.shape[2]
    if num_padded_pixels % lvl > 0:
        num_padded_pixels +=  lvl - tomo.data.shape[2] % lvl
    
    data = tomo.apply_padding(num_pad=num_padded_pixels, overwrite=False) 
    data = np.abs(-np.log(data));
    
    # Adjust center according to padding.
    center = tomo.center + (data.shape[2]-tomo.data.shape[2])/2.
    


    if num_grid is None or num_grid > data.shape[2]:
        num_grid = num_padded_pixels/lvl
        tomo.logger.debug("mlem_multilevel: num_grid set to " + str(num_grid) + " [ok]")
        
    if init_matrix is None:  
        #init_matrix = np.ones((data.shape[1]/lvl, num_grid, num_grid), dtype='float32') 
        init_matrix = np.ones((data.shape[1], num_grid, num_grid), dtype='float32') 
        tomo.logger.debug("mlem_multilevel: init_matrix set to ones [ok]")


    # Check again.
    if not isinstance(data, np.float32):
        data = np.array(data, dtype=np.float32, copy=False)

    if not isinstance(tomo.theta, np.float32):
        theta = np.array(theta, dtype=np.float32, copy=False)

    if not isinstance(tomo.center, np.float32):
        center = np.array(center, dtype=np.float32, copy=False)
        
    if not isinstance(iters, np.int32):
        iters = np.array(iters, dtype=np.int32, copy=False)

    if not isinstance(num_grid, np.int32):
        num_grid = np.array(num_grid, dtype=np.int32, copy=False)
        
    if not isinstance(init_matrix, np.float32):
        init_matrix = np.array(init_matrix, dtype=np.float32, copy=False)

      
    for m in reversed(range(level+1)):
        x = _downsample2d(data, level=m)
        cen = np.array(center/np.power(2, m), dtype=np.float32)
        num_grid = np.array(data.shape[2]/np.power(2, m), dtype=np.int32)
        it = np.array(iters, dtype=np.float32)
        y = _mlem(x, theta, cen, num_grid, it, init_matrix)

        if m != 0:
            init_matrix = _upsample2d(y, level=1)
        else:
            init_matrix = y
    data_recon = init_matrix
    
    
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
setattr(Session, 'upsample2d', upsample2d)
setattr(Session, 'upsample3d', upsample3d)
setattr(Session, 'art', art)
setattr(Session, 'gridrec', gridrec)
setattr(Session, 'mlem', mlem)
setattr(Session, 'mlem_multilevel', mlem_multilevel)

# Use original function docstrings for the wrappers.
diagnose_center.__doc__ = _diagnose_center.__doc__
upsample2d.__doc__ = _upsample2d.__doc__
upsample3d.__doc__ = _upsample3d.__doc__
diagnose_center.__doc__ = _diagnose_center.__doc__
art.__doc__ = _art.__doc__
gridrec.__doc__ = Gridrec.__doc__
mlem.__doc__ = _mlem.__doc__
