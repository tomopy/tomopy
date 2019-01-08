#!/bin/bash -l

cd /home/tomopy
python setup.py install --build-type=Release -- -DTOMOPY_USE_TIMEMORY=ON &> /dev/null
cd benchmarking

: ${NUM_ITER:=5}
: ${NUM_THREADS:=8}

if [ -n "${1}" ]; then NUM_ITER=$1; fi
if [ -n "${2}" ]; then NUM_THREADS=$2; fi

export TOMOPY_NUM_THREADS=${NUM_THREADS}
export TOMOPY_USE_C_SIRT=1
export TOMOPY_USE_CPU=1

python pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -a sirt -o tomo_00001_c -i ${NUM_ITER}

export TOMOPY_USE_C_SIRT=0

python pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -a sirt -o tomo_00001_cxx -i ${NUM_ITER}

export TOMOPY_USE_CPU=0

python pyctest_tomopy_rec.py /home/globus/tomo_00001/tomo_00001.h5 -a sirt -o tomo_00001_cuda -i ${NUM_ITER}
