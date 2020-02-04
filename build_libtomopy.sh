#!/bin/bash

REPODIR=$(cd $(dirname $0); pwd)

TOMOPY_BUILD_DIR=${REPODIR}/build

mkdir -p ${TOMOPY_BUILD_DIR}
cd  ${TOMOPY_BUILD_DIR}

cmake .. \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-std=c++11" \
  -DCMAKE_INSTALL_LIBDIR="lib" \
  -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} \
  ..

make -j 4
