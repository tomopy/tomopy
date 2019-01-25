#!/bin/bash -e

#git clone https://github.com/opencv/opencv.git opencv
#cd opencv
#git checkout 4.0.1

exit 0

wget https://github.com/opencv/opencv/archive/4.0.1.tar.gz -O opencv.tar.gz
mkdir -p opencv
cd opencv
tar --strip-components=1 -xzvf ../opencv.tar.gz

SOURCE_DIR=${PWD}
BINARY_DIR=${SOURCE_DIR}/build-opencv

mkdir -p ${BINARY_DIR}
cd ${BINARY_DIR}

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_CUDA_STUBS=ON \
    -DBUILD_TBB=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_WEBP=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_DOCS=OFF \
    -DWITH_CUDA=ON \
    -DWITH_EIGEN=ON \
    -DWITH_QT=OFF \
    -DWITH_WEBP=OFF \
    -DBUILD_opencv_cudev=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    ${SOURCE_DIR} \
    -G Ninja

# build
cmake --build ${PWD} --target all
# install
cmake --build ${PWD} --target install

cd ${SOURCE_DIR}
rm -rf ${BINARY_DIR}


