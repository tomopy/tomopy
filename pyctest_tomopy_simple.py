#!/usr/bin/env python

import os
import sys
import platform
import pyctest.pyctest as pyctest
import pyctest.helpers as helpers

parser = helpers.ArgumentParser("TomoPy", os.getcwd(), os.getcwd())
parser.add_argument("--build", help="Build name", required=True)
args = parser.parse_args()

pyctest.BUILD_NAME = "{}".format(args.build)
pyctest.BUILD_COMMAND = "python setup.py build_ext --inplace"
pyctest.UPDATE_COMMAND = "git"

test = pyctest.test()
test.SetName("unittest")
test.SetCommand(["nosetests"])

pyctest.generate_config()
pyctest.generate_test_file()
pyctest.run(pyctest.ARGUMENTS)
