#!/bin/bash -l

# make sure files are not owned by root
umask 0000

#------------------------------------------------------------------------------#
#   init
#------------------------------------------------------------------------------#
for j in profile bashrc; do
    if [ -d /etc/${j}.d ]; then
        for i in /etc/${j}.d/*.sh; do
            if [ -f ${i} ]; then source ${i}; fi
        done
    fi
done

#------------------------------------------------------------------------------#
#   tomopy
#------------------------------------------------------------------------------#
if [ ! -d /home/tomopy ]; then
    git clone https://github.com/jrmadsen/tomopy.git /home/tomopy
    cd /home/tomopy
    git checkout gpu-devel 2> /dev/null
else
    cd /home/tomopy
fi

#------------------------------------------------------------------------------#
#   compiler setup
#------------------------------------------------------------------------------#
if [ -n "${COMPILER}" ]; then
    COMPILER=$(echo ${COMPILER} | awk -F ' %HDFADC% ' '{print tolower($1)}')
    case "${COMPILER}" in
        "pgi")
            module load pgi
        ;;
        "pgi-llvm")
            module load pgi-llvm
            module load pgi
        ;;
        "clang")
            echo 1 | update-alternatives --config cc
            echo 1 | update-alternatives --config c++
        ;;
        "gcc")
            echo 2 | update-alternatives --config cc
            echo 2 | update-alternatives --config c++
        ;;
        *)
            echo "Invalid option \"${COMPILER}\". Valid options: pgi, pgi-llvm, clang, gcc"
        ;;
    esac
fi

#------------------------------------------------------------------------------#
#   build
#------------------------------------------------------------------------------#
if [ -f setup.py ]; then
    echo -e "Running \"$(which python) setup.py install\" with ARGS=\"${ARGS}\"..."
    $(which python) setup.py install ${ARGS} 1> /dev/null \
        && echo "Build success" || echo "Build failure"
fi

#------------------------------------------------------------------------------#
#   command
#------------------------------------------------------------------------------#
if [ -z "${1}" ]; then
    exec /bin/bash
else
    exec $@
fi
