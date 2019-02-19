#!/bin/bash

if [ "${NERSC_HOST}" = "edison" ]; then
    module load edison
    module load python/3.6-anaconda-4.4
    source activate tomopy-cpu
else
    module load python/3.6-anaconda-4.4
    module load gcc
    module load cuda
    source activate tomopy-gpu
fi

# algorithm settings
: ${TOMOPY_INTER:=1}
: ${TOMOPY_NUM_ITERATION:=50}
: ${DATA_TYPE:=partial}

if [ -n "${1}" ]; then TOMOPY_INTER=${1}; shift; fi
if [ -n "${1}" ]; then TOMOPY_NUM_ITERATION=${1}; shift; fi

# data settings
: ${BLOCK_BEG:=0}
: ${BLOCK_END:=24}
: ${NUM_SLICES:=$((${BLOCK_END} - ${BLOCK_BEG}))}

# parallelism settings
: ${PTL_CPU_AFFINITY:=0}
: ${PTL_VERBOSE:=1}
: ${TOMOPY_NUM_GPU:=4}
if [ "${NERSC_HOST}" = "edison" ]; then
    : ${TOMOPY_PYTHON_THREADS:=24}
else
    : ${TOMOPY_PYTHON_THREADS:=${TOMOPY_NUM_GPU}}
fi
: ${TOMOPY_NUM_THREADS:=$(( 80 / ${TOMOPY_PYTHON_THREADS} + 4 ))}
: ${CUDA_BLOCK_SIZE:=32}
: ${CUDA_GRID_SIZE:=0}

PTL_NUM_THREADS=${TOMOPY_NUM_THREADS}
NUMEXPR_MAX_THREADS=$(nproc)

export TOMOPY_INTER TOMOPY_NUM_ITERATION DATA_TYPE
export BLOCK_BEG BLOCK_END NUM_SLICES
export PTL_CPU_AFFINITY PTL_VERBOSE PTL_NUM_THREADS NUMEXPR_MAX_THREADS
export TOMOPY_NUM_GPU TOMOPY_PYTHON_THREADS TOMOPY_NUM_THREADS
export CUDA_BLOCK_SIZE CUDA_GRID_SIZE

configure-out()
{
    echo "${TOMOPY_DEVICE_TYPE}-${DATA_TYPE}-${TOMOPY_NUM_GPU}-${TOMOPY_INTER}-${CUDA_BLOCK_SIZE}-${CUDA_GRID_SIZE}-${TOMOPY_PYTHON_THREADS}-${TOMOPY_NUM_THREADS}-${TOMOPY_USE_C_ALGORITHMS}-${TOMOPY_USE_CPU}"
}

run-verbose()
{
    echo -e "\n### Running : '$@'... ###\n"
    eval $@
}
