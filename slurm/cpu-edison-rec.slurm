#!/bin/bash -le

#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH --job-name=cpu-edison
#SBATCH --output=cpu-edison-%j.log

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

for j in slice partial
do
    for i in nn linear cubic
    do
        run-recon-rec ${i} ${ALGORITHM} ${j}
    done
done

echo -e "Completed at $(date)"
