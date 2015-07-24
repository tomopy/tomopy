# -*- coding: utf-8 -*-
"""
Read the APS 1-ID tomography log file
"""
import tomopy
import os.path

# set the path to the micro-CT data set to reconstruct
fname = 'data_dir/sample_name_prefix'

fname = os.path.abspath(fname)
log_file = os.path.dirname(fname) + os.path.sep + 'TomoStillScan.dat'

# Read APS 1-ID log file.
contents = open(log_file, 'r')
for line in contents:
    ls = line.split()
    if len(ls)>1:
        if (ls[0]=="Tomography" and ls[1]=="scan"):
            prj_start = int(ls[6])
        elif (ls[0]=="Number" and ls[2]=="scan"):
            nprj = int(ls[4])
        elif (ls[0]=="Dark" and ls[1]=="field"):
            dark_start = int(ls[6])
        elif (ls[0]=="Number" and ls[2]=="dark"):
            ndark = int(ls[5])
        elif (ls[0]=="White" and ls[1]=="field"):
            flat_start = int(ls[6])
        elif (ls[0]=="Number" and ls[2]=="white"):
            nflat = int(ls[5])
contents.close()
