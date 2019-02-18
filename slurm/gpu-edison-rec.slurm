#!/bin/bash -le

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --job-name=gpu-edison
#SBATCH --output=gpu-edison-%j.log

echo -e "Starting at $(date)"

BASE_DIR=/project/projectdirs/m1759/jrmadsen
GLOBUS_DIR=${BASE_DIR}/globus
SOURCE_DIR=${BASE_DIR}/tomopy-edison-gpu
SCRIPT_DIR=$(realpath $(dirname ${BASH_SOURCE[0]}))
. ${SCRIPT_DIR}/env-common-settings.sh $@
cd ${SOURCE_DIR}

LOG_DIR=${PWD}/logs
mkdir -p ${LOG_DIR}

# algorithm settings
export TOMOPY_USE_C_ALGORITHMS=0
export TOMOPY_USE_CPU=0
export TOMOPY_DEVICE_TYPE=cuda

OUT="$(configure-out)"

run-verbose python setup.py install --enable-arch --enable-cuda --enable-gpu | tee ${LOG_DIR}/build-${OUT}.log
run-verbose cd ${SOURCE_DIR}/benchmarking
run-verbose srun nvidia-smi | tee ${LOG_DIR}/smi-${OUT}.log

run-recon()
{
    export DATA_TYPE=${1}
    OUT="$(configure-out)"
    run-verbose srun -c 80 \
        $(which python) \
        ./pyctest_tomopy_rec.py \
        -o ${OUT}-${ALGORITHM} \
        --type=${DATA_TYPE} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -r ${BLOCK_BEGIN} ${BLOCK_END} \
        -f png \
        ${GLOBUS_DIR}/tomo_00001/tomo_00001.h5 | tee ${LOG_DIR}/run-${OUT}.log
}

for i in 32 128 256
do
    export CUDA_BLOCK_SIZE=${i}
    for j in 0 16 32
    do
        export CUDA_GRID_SIZE=${j}

        export ALGORITHM=sirt
        run-recon partial
        run-recon slice

        export ALGORITHM=mlem
        run-recon partial
        run-recon slice
    done
done

echo -e "Completed at $(date)"
