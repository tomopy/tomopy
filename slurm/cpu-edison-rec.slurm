#!/bin/bash -le

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH --job-name=cpu-edison
#SBATCH --output=cpu-edison-%j.log

echo -e "Starting at $(date)"

BASE_DIR=/global/cscratch1/sd/jrmadsen
GLOBUS_DIR=${BASE_DIR}/globus
SOURCE_DIR=${BASE_DIR}/software/tomopy/tomopy-edison-gpu
SCRIPT_DIR=${SOURCE_DIR}/slurm
. ${SCRIPT_DIR}/env-common-settings.sh $@
cd ${SOURCE_DIR}

LOG_DIR=${PWD}/logs
mkdir -p ${LOG_DIR}

# algorithm settings
export TOMOPY_USE_C_ALGORITHMS=1
export TOMOPY_USE_CPU=1
export TOMOPY_DEVICE_TYPE=cpu

OUT="$(configure-out)"

run-verbose python setup.py install --enable-arch --disable-cuda --disable-gpu | tee ${LOG_DIR}/build-${OUT}.log
run-verbose cd ${SOURCE_DIR}/benchmarking

run-recon()
{
    export DATA_TYPE=${1}
    OUT="$(configure-out)"
    run-verbose srun -c ${TOMOPY_PYTHON_THREADS} \
        $(which python) \
        ./pyctest_tomopy_rec.py \
        -o ${OUT}-${ALGORITHM} \
        --type=${DATA_TYPE} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -r ${BLOCK_BEG} ${BLOCK_END} \
        -f png \
        ${GLOBUS_DIR}/tomo_00001/tomo_00001.h5 | tee ${LOG_DIR}/run-${ALGORITHM}-${OUT}.log &
    echo $!
}

export ALGORITHM=sirt
run-recon partial

export ALGORITHM=mlem
run-recon partial

export ALGORITHM=sirt
run-recon slice

export ALGORITHM=mlem
run-recon slice

echo -e "Completed at $(date)"
