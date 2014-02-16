# -*- coding: utf-8 -*-
import numpy as np
from tomopy.tools.multiprocess import worker


@worker
def threshold_segment(args):
    """
    Threshold based segmentation.
    """
    data, args, ind_start, ind_end = args
    cutoff = args
    
    data[data >= cutoff] = 1
    data[data < cutoff] = 0
    return ind_start, ind_end, data