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
    clang-format clang-format-${CLANG_VERSION}.0 libtbb-dev \
    libgomp1 libomp-dev libiomp-dev"

IMAGE_PACKAGES="libtiff5-dev libtiff-opengl libtiff-tools libtiff5-dev tcllib \
    libpng-dev libjpeg-dev pngtools libnetcdf-dev eog qiv zlib1g-dev"

PROFILE_PACKAGES="google-perftools libgoogle-perftools-dev"

MATH_PACKAGES="libblas-dev libopenblas-dev liblapack-dev libeigen3-dev"

VIZ_PACKAGES="xserver-xorg freeglut3-dev libx11-dev libx11-xcb-dev libxpm-dev \
    libxft-dev libxmu-dev libxv-dev libxrandr-dev libglew-dev libftgl-dev \
    libxkbcommon-x11-dev libxrender-dev libxxf86vm-dev libxinerama-dev \
    qt5-default"

EXTRA_PACKAGES="valgrind kcachegrind gdb openssh-server keychain"

MPI_PACKAGES="libopenmpi-dev openmpi-bin openmpi-common"

NVIDIA_PACKAGES="nvidia-nsight cuda-visual-tools-10-0 cuda-nsight-compute-10-0 \
    cuda-nsight-10-0"

apt-get install -y --reinstall ${CORE_PACKAGES} ${COMPILER_PACKAGES} \
    ${IMAGE_PACKAGES} ${PROFILE_PACKAGES} ${MATH_PACKAGES} ${VIZ_PACKAGES} \
    ${NVIDIA_PACKAGES}


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

# allow to fail
set +e

# set the java to java-8-openjdk (for nvvp)
echo 2 | update-alternatives --config java
