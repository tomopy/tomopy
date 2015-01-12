# -*- coding: utf-8 -*-
try:
    import pkg_resources  # part of setuptools
    __version__ = pkg_resources.require("tomopy")[0].version
except:
    pass

import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/tools')
os.environ['LD_LIBRARY_PATH'] = os.path.abspath(os.path.dirname(__file__)+'/lib')
os.environ['C_INCLUDE_PATH'] = os.path.abspath(os.path.dirname(__file__)+'/lib')

# ---------X-ray absorption tomography imports---------

# Reader/Writer functions for xtomo data.
from xtomo.xtomo_io import xtomo_reader
from xtomo.xtomo_io import xtomo_writer

# Main xtomo object constructor.
from xtomo.xtomo_dataset import XTomoDataset as xtomo_dataset

# Hooks to other functions.
import xtomo.xtomo_preprocess
import xtomo.xtomo_recon
import xtomo.xtomo_postprocess

# ---------X-ray fluorescence tomography imports---------

# Reader/Writer functions for xftomo data.
#from xtomo.xtomo_io import xtomo_reader
#from xtomo.xtomo_io import xtomo_writer

# Main xtomo object constructor.
from xftomo.xftomo_dataset import XFTomoDataset as xftomo_dataset

# Hooks to other functions.
#import xtomo.xtomo_preprocess
#import xtomo.xtomo_recon
#import xtomo.xtomo_postprocess
