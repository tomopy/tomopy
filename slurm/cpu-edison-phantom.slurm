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

run-phantom()
{
    PHANTOM=${1}
    shift
    run-verbose srun -c ${TOMOPY_PYTHON_THREADS} \
        $(which python) \
        ./pyctest_tomopy_phantom.py \
        -o ${OUT}-${ALGORITHM} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -f png \
        -p ${PHANTOM} $@ | tee ${LOG_DIR}/run-${PHANTOM}-${ALGORITHM}-${OUT}.log &
    echo $!
}

export ALGORITHM=sirt
pid=$(run-phantom shepp3d --partial -r ${BLOCK_BEG} ${BLOCK_END} | awk '{print $NF}')
wait $pid

export ALGORITHM=mlem
pid=$(run-phantom shepp3d --partial -r ${BLOCK_BEG} ${BLOCK_END} | awk '{print $NF}')
wait $pid

export ALGORITHM=sirt
pids[0]=$(run-phantom cameraman | awk '{print $NF}')
sleep 10
pids[1]=$(run-phantom lena | awk '{print $NF}')
sleep 10

pids[2]=$(run-phantom cameraman | awk '{print $NF}')
sleep 10
pids[3]=$(run-phantom lena | awk '{print $NF}')
sleep 10

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo -e "Completed at $(date)"
