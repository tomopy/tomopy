#!/bin/bash -le

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --job-name=gpu-edison
#SBATCH --output=gpu-edison-%j.log

run-verbose()
{
    echo -e "\nRunning : '$@'...\n"
    eval $@
}

echo -e "Starting at $(date)"

module load python/3.6-anaconda-4.4
module load gcc
module load cuda

source activate tomopy-gpu

BASE_DIR=/global/cscratch1/sd/jrmadsen
GLOBUS_DIR=${BASE_DIR}/globus
SOURCE_DIR=${BASE_DIR}/software/tomopy/tomopy-edison-gpu

cd ${BASE_DIR}

LOG_DIR=${PWD}/logs
mkdir -p ${LOG_DIR}

# algorithm settings
export TOMOPY_USE_C_ALGORITHMS=1
export TOMOPY_USE_CPU=1
export TOMOPY_DEVICE_TYPE=cpu
export TOMOPY_INTER=1
export TOMOPY_NUM_ITERATION=50
export DATA_TYPE=partial

if [ -n "${1}" ]; then TOMOPY_INTER=${1}; shift; fi
if [ -n "${1}" ]; then TOMOPY_NUM_ITERATION=${1}; shift; fi

# data settings
export BLOCK_BEGIN=0
export BLOCK_END=24
export NUM_SLICES=$((${BLOCK_END} - ${BLOCK_BEGIN}))

# parallelism settings
export PTL_CPU_AFFINITY=1
export PTL_VERBOSE=1
export TOMOPY_NUM_GPU=8
export TOMOPY_PYTHON_THREADS=8
export TOMOPY_NUM_THREADS=10
export TOMOPY_STREAM_SYNC=1
export CUDA_BLOCK_SIZE=32
export PTL_NUM_THREADS=${TOMOPY_NUM_THREADS}
export NUMEXPR_MAX_THREADS=80

OUT="gpu-${TOMOPY_DEVICE_TYPE}-${DATA_TYPE}-${TOMOPY_NUM_GPU}-${TOMOPY_INTER}-${CUDA_BLOCK_SIZE}-${TOMOPY_PYTHON_THREADS}-${TOMOPY_NUM_THREADS}-${TOMOPY_USE_C_SIRT}-${TOMOPY_USE_CPU}"

# echo -e "\n### cleaning... ###\n"
# python setup.py clean 

echo -e "\n### building... ###\n"
run-verbose python setup.py install --enable-cuda --enable-arch --disable-openmp --disable-nvtx --enable-gpu --disable-timemory --disable-avx512 -- -DTOMOPY_USE_PYBIND11=OFF -DTOMOPY_USE_PTL=ON | tee ${LOG_DIR}/build-${OUT}.log

run-verbose cd ${SOURCE_DIR}/benchmarking

echo -e "\n### printing GPU info... ###\n"
run-verbose srun nvidia-smi | tee ${LOG_DIR}/smi-${OUT}.log

echo -e "\n### running... ###\n"
run-phantom()
{
    PHANTOM=${1}
    shift
    run-verbose srun -c 80 \
	$(which python) \
	./pyctest_tomopy_phantom.py \
	--output-dir ${OUT} \
	-n ${TOMOPY_PYTHON_THREADS} \
	-i ${TOMOPY_NUM_ITERATION} \
	-a sirt \
        -f png \
	-p ${PHANTOM} $@ | tee ${LOG_DIR}/run-${PHANTOM}-${OUT}.log
}

run-phantom shepp3d --partial -r ${BLOCK_BEGIN} ${BLOCK_END}
run-phantom shepp2d
run-phantom cameraman
run-phantom lena
run-phantom peppers

echo -e "Completed at $(date)"
