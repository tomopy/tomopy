#!/bin/bash

: ${NUM_ITER:=35}
: ${FORMAT:=png}

export NUM_ITER FORMAT

move-folder()
{
    mv ${1}/sirt ${1}/${1}_m${2}_p${3}_sirt
}

exec-scripts()
{
    ./pyctest_tomopy_phantom.py -i ${NUM_ITER} -p shepp2d -f ${FORMAT}
    ./pyctest_tomopy_phantom.py -i ${NUM_ITER} -p cameraman -f ${FORMAT}
}

export TOMOPY_USE_C_SIRT=0
export TOMOPY_USE_C_PROJECT=0
exec-scripts
move-folder shepp2d rot rot
move-folder cameraman rot rot



export TOMOPY_USE_C_SIRT=1
export TOMOPY_USE_C_PROJECT=0
exec-scripts
move-folder shepp2d ray rot
move-folder cameraman ray rot



export TOMOPY_USE_C_SIRT=0
export TOMOPY_USE_C_PROJECT=1
exec-scripts
move-folder shepp2d rot ray
move-folder cameraman rot ray



export TOMOPY_USE_C_SIRT=1
export TOMOPY_USE_C_PROJECT=1
exec-scripts
move-folder shepp2d ray ray
move-folder cameraman ray ray
