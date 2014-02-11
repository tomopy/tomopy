# -*- coding: utf-8 -*-
import os
import shutil
from tomopy.dataio.reader import Dataset
from gridrec import Gridrec
from diagnose_center import diagnose_center
from optimize_center import optimize_center
import logging
logger = logging.getLogger("tomopy")


def diagnose_center_wrapper(TomoObj,
			    dir_path,
			    slice_no=None,
			    center_start=None,
			    center_end=None,
			    center_step=None):
    dir_path = os.path.dirname(TomoObj.file_name) + '/tmp/'
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    logger.debug("tmp directory create [ok]")
    if TomoObj.FLAG_DATA and TomoObj.FLAG_THETA:
        diagnose_center(TomoObj.data, 
			TomoObj.theta, 
			dir_path=dir_path,
			slice_no=slice_no,
			center_start=center_start,
			center_end=center_end,
			center_step=center_step)
        #TomoObj.provenance['diagnose_center'] = (args, kwargs)
        logger.debug("save diagnostic images at %s [ok]", dir_path)
    else:
        logger.warning("diagnose rotation center (data missing) [bypassed]")

def optimize_center_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA and TomoObj.FLAG_THETA:
        TomoObj.center = optimize_center(TomoObj.data, TomoObj.theta, *args, **kwargs)
        TomoObj.provenance['optimize_center'] = (args, kwargs)
        logger.info("optimize rotation center [ok]")
    else:
        logger.warning("optimize rotation center (data missing) [bypassed]")
    
def gridrec_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA and TomoObj.FLAG_THETA:
        # Find center if center is absent.
        if not hasattr(TomoObj, 'center'):
            TomoObj.center = optimize_center(TomoObj.data, TomoObj.theta)
        recon = Gridrec(TomoObj.data, *args, **kwargs)
        recon.run(TomoObj.data, center=TomoObj.center, theta=TomoObj.theta)
        TomoObj.data_recon = recon.data_recon
        TomoObj.gridrec_pars = recon.params
        TomoObj.FLAG_DATA_RECON = True
        TomoObj.provenance['gridrec'] = (args, kwargs)
        logger.info("gridrec reconstruction [ok]")
    else:
        logger.warning("gridrec reconstruction (data missing) [bypassed]")


setattr(Dataset, 'diagnose_center', diagnose_center_wrapper)
setattr(Dataset, 'optimize_center', optimize_center_wrapper)
setattr(Dataset, 'gridrec', gridrec_wrapper)

diagnose_center_wrapper.__doc__ = diagnose_center.__doc__
optimize_center_wrapper.__doc__ = optimize_center.__doc__
gridrec_wrapper.__doc__ = Gridrec.__doc__