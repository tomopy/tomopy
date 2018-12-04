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
: ${BRANCH:="master"}

################################################################################
#
#           TiMemory
#
################################################################################

echo -e "Cloning TiMemory branch: ${BRANCH}..."
git clone -b ${BRANCH} https://github.com/jrmadsen/TiMemory.git /root/timemory-src
cd /root/timemory-src

export TIMEMORY_BUILD_TYPE=RelWithDebInfo

pip install -v .

if [ "${TEST}" -gt 0 ]; then

    mkdir build-test
    cd build-test
    cmake \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_INSTALL_PREFIX=/opt/local \
        -DPYTHON_EXECUTABLE=/opt/conda/bin/python \
        -DTIMEMORY_BUILD_TESTING=ON \
        -DTIMEMORY_BUILD_EXAMPLES=ON \
        -DTIMEMORY_TEST_MPI=OFF \
        -DCTEST_SITE=docker \
        -DCTEST_MODEL=Continuous \
        -DCTEST_LOCAL_CHECKOUT=ON \
        ${PWD}/..
    ctest -V --output-on-failure
    cd /root/timemory-src
    rm -rf build-test

fi
