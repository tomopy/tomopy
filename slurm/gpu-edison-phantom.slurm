#!/bin/bash -le

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --job-name=gpu-edison
#SBATCH --output=gpu-edison-%j.log

echo -e "Starting at $(date)"

BASE_DIR=/project/projectdirs/m1759/jrmadsen
GLOBUS_DIR=${BASE_DIR}/globus
SOURCE_DIR=${BASE_DIR}/tomopy-edison-gpu
SCRIPT_DIR=${SOURCE_DIR}/slurm
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

run-phantom()
{
    PHANTOM=${1}
    shift
    run-verbose srun -n 1 -c 80 \
        $(which python) \
        ./pyctest_tomopy_phantom.py \
        -o ${OUT}-${ALGORITHM} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -f png \
        -p ${PHANTOM} $@ | tee ${LOG_DIR}/run-${PHANTOM}-${ALGORITHM}-${OUT}.log
}

for j in 0 32 64
do
    for i in 32 128
    do
        export CUDA_BLOCK_SIZE=${i}
        export CUDA_GRID_SIZE=${j}
        export ALGORITHM=sirt
        run-phantom shepp3d --partial -r ${BLOCK_BEG} ${BLOCK_END}
        export ALGORITHM=mlem
        run-phantom shepp3d --partial -r ${BLOCK_BEG} ${BLOCK_END}
    done
done

for j in 0 32 64
do
    for i in 32 128
    do
        export CUDA_BLOCK_SIZE=${i}
        export CUDA_GRID_SIZE=${j}
        export ALGORITHM=sirt
        run-phantom cameraman
        run-phantom lena
        export ALGORITHM=mlem
        run-phantom cameraman
        run-phantom lena
    done
done

echo -e "Completed at $(date)"
