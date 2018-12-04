#!/bin/bash

set -o errexit

apt-get update

apt-get -y install build-essential software-properties-common

add-apt-repository ppa:ubuntu-toolchain-r/test -y

apt-get update

apt-get dist-upgrade -y

apt-get install -y --reinstall \
    cmake build-essential git-core apt-utils libtiff5-dev libtiff-opengl \
    wget libtiff-tools libtiff5-dev tcllib libpng-dev libjpeg-dev pngtools \
    python emacs-nox vim bash-completion man-db \
    libgomp1 libgomp1-dbg libtbb-dev libomp-dev \
    environment-modules libnetcdf-dev \
    gcc-${GCC_VERSION} gcc-${GCC_VERSION}-doc g++-${GCC_VERSION} \
    gcc-${GCC_VERSION}-multilib gcc-${GCC_VERSION}-offload-nvptx \
    clang-5.0 clang-${CLANG_VERSION}.0 libc++-dev libc++abi-dev \
    google-perftools libgoogle-perftools-dev
    #valgrind kcachegrind gdb
    #gcc-6 gcc-6-doc g++-6 gcc-6-multilib \
    #xserver-xorg eog qiv
    #libnetcdf-dev
    #curl openssh-server keychain \
    #libopenmpi-dev openmpi-bin openmpi-common  \

priority=10
for i in 5 6 7 8
do
    if [ -n "$(which gcc-${i})" ]; then
        update-alternatives --install $(which gcc) gcc $(which gcc-${i}) ${priority} \
            --slave $(which g++) g++ $(which g++-${i})
        priority=$(( ${priority}+10 ))
    fi
done

priority=10
for i in 5 6 7
do
    if [ -n "$(which clang-${i}.0)" ]; then
        update-alternatives --install /usr/bin/clang clang $(which clang-${i}.0) 10 \
            --slave /usr/bin/clang++ clang++ $(which clang++-${i}.0)
        priority=$(( ${priority}+10 ))
    fi
done

update-alternatives --install $(which cc)  cc  $(which clang)   10
update-alternatives --install $(which c++) c++ $(which clang++) 10
update-alternatives --install $(which cc)  cc  $(which gcc)     20
update-alternatives --install $(which c++) c++ $(which g++)     20

apt-get -y autoclean

rm -rf /var/lib/apt/lists/*
