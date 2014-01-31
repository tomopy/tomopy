# -*- coding: utf-8 -*-
try:
    import pkg_resources  # part of setuptools
    __version__ = pkg_resources.require("tomopy")[0].version
except:
    pass

# Main function to create TomoObj.
from dataio.reader import Dataset
from recon.gridrec import tomoRecon