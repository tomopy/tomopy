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

export TOMOPY_USE_C_SIRT=1 
export TOMOPY_NUM_THREADS=1

cd /global/cscratch1/sd/jrmadsen/software/tomopy/tomopy-edison-gpu

python setup.py install --disable-gpu --disable-cuda --enable-openmp -- -DTOMOPY_USE_PTL=OFF

cd /global/cscratch1/sd/jrmadsen/software/tomopy/tomopy-edison-gpu/benchmarking

$(which python) ./pyctest_tomopy_rec.py /global/cscratch1/sd/jrmadsen/globus/tomo_00001/tomo_00001.h5 \
    -o cpu-c-partial-24 --type=partial \
    -n 24 -i 10 -b 0 -e 24 \
    -a sirt

#python ./pyctest_tomopy_rec.py /global/cscratch1/sd/jrmadsen/globus/tomo_00001/tomo_00001.h5 \
#    -o cpu-compare-slice-24 --type=slice -n 1 -i 1 \
#    -a sirt -g 4
