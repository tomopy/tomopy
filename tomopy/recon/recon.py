# -*- coding: utf-8 -*-
from tomopy.dataio.reader import Dataset
from gridrec import Gridrec
from optimize_center import optimize_center
import logging
logger = logging.getLogger("tomopy")


def optimize_center_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        TomoObj.center = optimize_center(TomoObj.data, *args, **kwargs)
        logger.info("optimize rotation center [ok]")
    else:
        logger.warning("optimize rotation center [bypassed]")
    
def gridrec_wrapper(TomoObj, *args, **kwargs):
    if TomoObj.FLAG_DATA:
        # Find center if center is absent.
        if not hasattr(TomoObj, 'center'):
            TomoObj.center = optimize_center(TomoObj.data)
        recon = Gridrec(TomoObj.data, *args, **kwargs)
        recon.run(TomoObj.data, center=TomoObj.center, theta=TomoObj.theta)
        TomoObj.data_recon = recon.data_recon
        TomoObj.gridrec_pars = recon.params
        logger.info("gridrec reconstruction [ok]")
    else:
        logger.warning("gridrec reconstruction [bypassed]")


setattr(Dataset, 'optimize_center', optimize_center_wrapper)
setattr(Dataset, 'gridrec', gridrec_wrapper)

optimize_center_wrapper.__doc__ = optimize_center.__doc__
gridrec_wrapper.__doc__ = Gridrec.__doc__