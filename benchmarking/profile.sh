#!/bin/bash

: ${TOMOPY_NUM_THREADS:=8}
: ${CUDA_BLOCK_SIZE:=16}
: ${N:=0}
: ${NITR:=10}

if [ -n "${1}" ]; then N=${1}; shift; fi
if [ -n "${1}" ]; then NITR=${1}; shift; fi

export TOMOPY_NUM_THREADS
export CUDA_BLOCK_SIZE

python setup.py install

nvprof -s -u col --log-file block_${CUDA_BLOCK_SIZE}_thread_${TOMOPY_NUM_THREADS}.${N}.nvprof-log \
       $(which python) \
       ./pyctest_tomopy_rec.py \
       -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} \
       -i ${NITR} \
       /home/globus/tomo_00001/tomo_00001.h5

nv-nsight-cu-cli --nvtx -s 10 -c 100 \
                 -o block_${CUDA_BLOCK_SIZE}_thread_${TOMOPY_NUM_THREADS}.${N} \
		 $(which python) \
		 ./pyctest_tomopy_rec.py \
		 -o tomo_00001_sirt/block_${CUDA_BLOCK_SIZE}_threads_${TOMOPY_NUM_THREADS} \
		 -i ${NITR} \
		 /home/globus/tomo_00001/tomo_00001.h5
