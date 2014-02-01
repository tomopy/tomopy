# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from gridrec import Gridrec
from find_center import find_center
import logging
logger = logging.getLogger("tomopy")


def find_center_wrapper(TomoObj, *args, **kwargs):
    logger.info("finding rotation center")
    TomoObj.center = find_center(TomoObj.data, *args, **kwargs)
    
def gridrec_wrapper(TomoObj, *args, **kwargs):
    logger.info("performing reconstruction with gridrec")
    # Find center if center is absent.
    if not hasattr(TomoObj, 'center'):
        TomoObj.center = find_center(TomoObj.data)
    recon = Gridrec(TomoObj.data, *args, **kwargs)
    recon.run(TomoObj.data, center=TomoObj.center, theta=TomoObj.theta)
    TomoObj.data_recon = recon.data_recon
    TomoObj.gridrec_pars = recon.params
    

setattr(Dataset, 'find_center', find_center_wrapper)
setattr(Dataset, 'gridrec', gridrec_wrapper)

find_center_wrapper.__doc__ = find_center.__doc__
gridrec_wrapper.__doc__ = Gridrec.__doc__