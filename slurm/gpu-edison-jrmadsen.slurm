#!/bin/bash -l

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --job-name=gpu-edison
#SBATCH --output=gpu-edison-%j.log

echo -e "Starting at $(date)"

module load python/3.6-anaconda-4.4
module load gcc
module load cuda

set -o errexit

srun nvidia-smi

source activate tomopy-gpu

cd /project/projectdirs/m1759/jrmadsen/tomopy-edison-gpu

#python setup.py clean 
python setup.py install --disable-openmp --disable-nvtx --enable-gpu --enable-cuda --enable-arch -- -DTOMOPY_USE_PYBIND11=OFF -DTOMOPY_USE_PTL=ON

export NUMEXPR_MAX_THREADS=80

# algorithm settings
export TOMOPY_USE_C_SIRT=0
export TOMOPY_USE_CPU=0
export TOMOPY_GPU_TYPE=cuda
export TOMOPY_INTER=1
export TOMOPY_NUM_ITERATION=10

# parallelism settings
export PTL_CPU_AFFINITY=1
export PTL_VERBOSE=1
export TOMOPY_NUM_GPU=4
export TOMOPY_PYTHON_THREADS=8
export TOMOPY_NUM_THREADS=10
export TOMOPY_STREAM_SYNC=1
export CUDA_BLOCK_SIZE=32
export BLOCK_BEGIN=872
export BLOCK_END=920
export PTL_NUM_THREADS=${TOMOPY_NUM_THREADS}

cd /project/projectdirs/m1759/jrmadsen/tomopy-edison-gpu/benchmarking

srun -c 80 \
    $(which python) \
    ./pyctest_tomopy_rec.py \
    -o gpu-cuda-partial-${TOMOPY_NUM_GPU}-${TOMOPY_INTER}-${CUDA_BLOCK_SIZE}-${TOMOPY_PYTHON_THREADS}-${TOMOPY_NUM_THREADS} \
    --type=partial \
    -n ${TOMOPY_PYTHON_THREADS} \
    -i ${TOMOPY_NUM_ITERATION} \
    -a sirt \
    -b ${BLOCK_BEGIN} \
    -e ${BLOCK_END} \
    /project/projectdirs/m1759/jrmadsen/globus/tomo_00001/tomo_00001.h5

#python ./pyctest_tomopy_rec.py /global/cscratch1/sd/jrmadsen/globus/tomo_00001/tomo_00001.h5 \
#    -o gpu-compare-slice-24 --type=slice -n 1 -i 1 \
#    -a sirt -g 4

echo -e "Completed at $(date)"
