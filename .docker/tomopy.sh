#!/bin/bash

set -o errexit

cd /root

if [ -f /opt/intel/bin/icc ]; then
    : ${CC:=/opt/intel/bin/icc}
    : ${CXX:=/opt/intel/bin/icpc}
else
    : ${CC:=$(which cc)}
    : ${CXX:=$(which c++)}
fi

export CC
export CXX

: ${TEST:=0}
: ${BRANCH:="gpu-devel"}

################################################################################
#
#           tomopy
#
################################################################################

cd /root

if [ -d /opt/intel ]; then
    export C_INCLUDE_PATH=/opt/intel/compiler_and_libraries/linux/mkl/include:/opt/conda/include:${C_INCLUDE_PATH}
else
    export C_INCLUDE_PATH=/opt/conda/include:${C_INCLUDE_PATH}
fi

echo -e "Cloning tomopy branch: ${BRANCH}..."
if [ ! -d "/root/tomopy-src" ]; then
    git clone -b ${BRANCH} https://github.com/jrmadsen/tomopy /root/tomopy-src
fi

cd /root/tomopy-src

if [ -f "/root/setup.py" ]; then
    mv /root/setup.py /root/tomopy-src/setup.py
fi

pip install -v .

if [ "${TEST}" -gt 0 ]; then

    python setup.py build_ext --inplace
    nosetests test

fi
