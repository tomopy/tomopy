#!/bin/bash -l

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --reservation=N9_COE
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
python setup.py install --disable-openmp --disable-nvtx --enable-gpu --enable-cuda -- -DTOMOPY_USE_PYBIND11=OFF -DTOMOPY_USE_PTL=ON

#export TOMOPY_USE_C_SIRT=0
#export TOMOPY_USE_CPU=0
#export TOMOPY_GPU_TYPE=cuda
export TOMOPY_NUM_THREADS=8
export TOMOPY_INTER=1
export CUDA_BLOCK_SIZE=32
export NUMEXPR_MAX_THREADS=80

cd /project/projectdirs/m1759/jrmadsen/tomopy-edison-gpu/benchmarking

srun -c 64 $(which python) ./pyctest_tomopy_rec.py /project/projectdirs/m1759/jrmadsen/globus/tomo_00001/tomo_00001.h5 \
    -o gpu-cuda-partial-1-8-64 --type=partial \
    -n 8 -i 10 -b 0 -e 24 \
    -a sirt

#python ./pyctest_tomopy_rec.py /global/cscratch1/sd/jrmadsen/globus/tomo_00001/tomo_00001.h5 \
#    -o gpu-compare-slice-24 --type=slice -n 1 -i 1 \
#    -a sirt -g 4

echo -e "Completed at $(date)"
