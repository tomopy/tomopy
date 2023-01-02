#!/bin/bash
echo "Building tomopy software using CMake..."
echo "<<< Make sure you've got cython and nvcc compiler installed >>>"
rm -r build
mkdir build
cd build

# conifigure:
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_INSTALL_LIBDIR=lib
# build 
cmake --build .
# install 
cmake --install .