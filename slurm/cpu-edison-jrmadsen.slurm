#!/bin/bash -l

#SBATCH -q premium
#SBATCH -A m1759
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH --job-name=cpu-edison
#SBATCH --output=cpu-edison-%j.log

module load edison
source activate tomopy-gpu-dev

set -o errexit

# algorithm settings
export TOMOPY_USE_C_SIRT=1
export TOMOPY_USE_CPU=1
export TOMOPY_NUM_ITERATION=10

# parallelism settings
export PTL_CPU_AFFINITY=1
export TOMOPY_PYTHON_THREADS=24
export TOMOPY_NUM_THREADS=1
export BLOCK_BEGIN=884
export BLOCK_END=908

cd /global/cscratch1/sd/jrmadsen/software/tomopy/tomopy-edison-gpu

python setup.py install --disable-gpu --disable-cuda --disable-openmp -- -DTOMOPY_USE_PTL=OFF

cd /global/cscratch1/sd/jrmadsen/software/tomopy/tomopy-edison-gpu/benchmarking

$(which python) \
        ./pyctest_tomopy_rec.py \
        -o cpu-partial-${TOMOPY_PYTHON_THREADS}-${TOMOPY_NUM_THREADS} \
        --type=partial \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a sirt \
        -b ${BLOCK_BEGIN} \
        -e ${BLOCK_END} \
	/global/cscratch1/sd/jrmadsen/globus/tomo_00001/tomo_00001.h5
