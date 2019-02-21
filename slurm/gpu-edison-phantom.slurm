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

. ${PWD}/slurm/env-common-settings.sh $@

cd ${SOURCE_DIR}

LOG_DIR=${SOURCE_DIR}/logs
mkdir -p ${LOG_DIR}
OUT="$(configure-out)"

run-verbose python setup.py install ${BUILD_ARGS} | tee ${LOG_DIR}/build-${OUT}.log
run-verbose cd ${SOURCE_DIR}/benchmarking
run-verbose-smi

: ${ALGORITHM:="sirt"}

for j in shepp2d cameraman shepp3d
do
    for i in nn linear cubic
    do
        run-recon-phantom ${i} ${ALGORITHM} ${j}
    done
done

echo -e "Completed at $(date)"
