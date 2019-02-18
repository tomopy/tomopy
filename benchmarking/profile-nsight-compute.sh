#!/bin/bash

: ${TOMOPY_NUM_THREADS:=8}
: ${CUDA_BLOCK_SIZE:=32}
: ${CUDA_GRID_SIZE:=16}
: ${N:=0}
: ${NITR:=10}
: ${GLOBUS_PATH:=${HOME}/devel/globus}
: ${ALGORITHM:=mlem}

if [ -n "${1}" ]; then N=${1}; shift; fi
if [ -n "${1}" ]; then NITR=${1}; shift; fi

export TOMOPY_NUM_THREADS
export CUDA_BLOCK_SIZE
export CUDA_GRID_SIZE

python setup.py install

mkdir -p tomo_00001_output/${ALGORITHM}

nv-nsight-cu-cli --nvtx -s 10 -c 100 \
                 -o tomo_00001_output/${ALGORITHM}/block_${CUDA_BLOCK_SIZE}_thread_${TOMOPY_NUM_THREADS}.${N} \
		 $(which python) \
		 ./pyctest_tomopy_rec.py \
		 -o tomo_00001_output/${ALGORITHM}/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} \
		 -i ${NITR} -a ${ALGORITHM} --type=slice \
		 ${GLOBUS_PATH}/tomo_00001/tomo_00001.h5
