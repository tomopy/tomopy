#!/bin/bash -l

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
#   command
#------------------------------------------------------------------------------#
if [ -z "${1}" ]; then
    exec /bin/bash
else
    exec $@
fi
