#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read the Swiss Light Source tomcat tomography log file
"""

import tomopy
import os.path

# Set path to the micro-CT data to reconstruct.
fname = 'data_dir/sample_name_prefix'

fname = os.path.abspath(fname)
log_fname = fname + '.log'

# Read SLS tomcat log file.
contents = open(log_fname, 'r')
for line in contents:
    ls = line.split()
    if len(ls)>1:
        if (ls[0]=="Number" and ls[2]=="darks"):
            ndark = int(ls[4])
        elif (ls[0]=="Number" and ls[2]=="flats"):
            nflat = int(ls[4])
        elif (ls[0]=="Number" and ls[2]=="projections"):
            nproj = int(ls[4])
        elif (ls[0]=="Rot" and ls[2]=="min"):
            angle_start = float(ls[6])
        elif (ls[0]=="Rot" and ls[2]=="max"):
            angle_end = float(ls[6])
        elif (ls[0]=="Angular" and ls[1]=="step"):
            angle_step = float(ls[4])
contents.close()

dark_start = 1
dark_end = ndark + 1
flat_start = dark_end
flat_end = flat_start + nflat
proj_start = flat_end
proj_end = proj_start + nproj


