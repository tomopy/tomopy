#!/bin/bash -e

#-----------------------------------------------------------------------------#
#   GENERAL CONFIG
#-----------------------------------------------------------------------------#
apt-get update
apt-get -y install build-essential software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update
apt-get dist-upgrade -y

#-----------------------------------------------------------------------------#
#   PACKAGES
#-----------------------------------------------------------------------------#
CORE_PACKAGES="cmake build-essential git-core apt-utils curl wget ninja-build \
    python emacs-nox vim bash-completion man-db environment-modules"

COMPILER_PACKAGES="gcc-${GCC_VERSION} gcc-${GCC_VERSION}-doc g++-${GCC_VERSION} \
    gcc-${GCC_VERSION}-multilib gcc-${GCC_VERSION}-offload-nvptx \
    clang-${CLANG_VERSION}.0 libc++-dev libc++abi-dev \
    libgomp1 libgomp1-dbg libtbb-dev libomp-dev clang-format clang-format-${CLANG_VERSION}.0"

IMAGE_PACKAGES="libtiff5-dev libtiff-opengl libtiff-tools libtiff5-dev tcllib \
    libpng-dev libjpeg-dev pngtools libnetcdf-dev eog qiv zlib1g-dev"

PROFILE_PACKAGES="google-perftools libgoogle-perftools-dev"

MATH_PACKAGES="libblas-dev libopenblas-dev liblapack-dev libeigen3-dev"

EXTRA_PACKAGES="valgrind kcachegrind gdb\
    xserver-xorg openssh-server keychain"

MPI_PACKAGES="libopenmpi-dev openmpi-bin openmpi-common"

apt-get install -y --reinstall ${CORE_PACKAGES} ${COMPILER_PACKAGES} \
    ${IMAGE_PACKAGES} ${PROFILE_PACKAGES} ${MATH_PACKAGES}


#-----------------------------------------------------------------------------#
#   ALTERNATIVES
#-----------------------------------------------------------------------------#
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

#-----------------------------------------------------------------------------#
#   CLEANUP
#-----------------------------------------------------------------------------#
apt-get -y autoclean
rm -rf /var/lib/apt/lists/*
