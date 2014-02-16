# -*- coding: utf-8 -*-
try:
    import pkg_resources  # part of setuptools
    __version__ = pkg_resources.require("tomopy")[0].version
except:
    pass

# Main function to create TomoObj.
from dataio.reader import Dataset

# Hooks to other functions.
import dataio.writer
import preprocess.preprocess
import recon.recon
import postprocess.postprocess
