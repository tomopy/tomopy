#!/usr/bin/env python

import os
import sys
import shutil
import subprocess as sp

ret = 0
output = "No output generated"

try:
    cwd = os.getcwd()
    nwd = os.path.realpath(os.path.dirname(cwd))
    os.chdir(nwd)
    cmd = [sys.executable, "setup.py"] + sys.argv[1:]
    print("\nExecuting:\n\n\t'{}'\n".format(" ".join(cmd)))
    output = sp.check_output(cmd)
except Exception as e:
    print("Exception: {}".format(e))
    print("stdout:\n\n{}\n".format(output))
    ret = 1

sys.exit(ret)
