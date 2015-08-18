#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read the Swiss Light Source tomcat tomography log file
"""

from __future__ import print_function
import tomopy
import os.path

if __name__ == '__main__':

    # Set path to the CT data to reconstruct.
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
