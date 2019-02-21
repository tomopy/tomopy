#!/bin/bash

BUILD_CPU_ARGS="--enable-arch --disable-cuda --disable-gpu"
BUILD_GPU_ARGS="--enable-arch --enable-cuda --enable-gpu"

: ${TEST_MODE:="gpu"}

if [ "${NERSC_HOST}" = "edison" ]; then
    TEST_MODE="c"
fi

if [ -n "${1}" ]; then TEST_MODE=${1}; shift; fi

if [ "${TEST_MODE}" = "c" ]; then

    module load edison
    module load python/3.6-anaconda-4.4

    : ${CONDA_ENVIRON:=tomopy-cpu}
    : ${BASE_DIR:=/global/cscratch1/sd/jrmadsen}
    : ${GLOBUS_DIR:=${BASE_DIR}/globus}
    : ${SOURCE_DIR:=${BASE_DIR}/software/tomopy/tomopy-edison-gpu}
    : ${SCRIPT_DIR:=${SOURCE_DIR}/slurm}
    # algorithm settings
    export TOMOPY_USE_C_ALGORITHMS=0
    export TOMOPY_DEVICE=cpu
    export BUILD_ARGS="${BUILD_CPU_ARGS}"

elif [ "${TEST_MODE}" = "gpu" ]; then

    module load python/3.6-anaconda-4.4
    module load gcc
    module load cuda

    : ${CONDA_ENVIRON:=tomopy-gpu}
    : ${BASE_DIR:=/project/projectdirs/m1759/jrmadsen}
    : ${GLOBUS_DIR:=${BASE_DIR}/globus}
    : ${SOURCE_DIR:=${PWD}}
    : ${SCRIPT_DIR:=${SOURCE_DIR}/slurm}
    # algorithm settings
    export TOMOPY_USE_C_ALGORITHMS=0
    export TOMOPY_DEVICE=gpu
    export BUILD_ARGS="${BUILD_GPU_ARGS}"

elif [ "${TEST_MODE}" = "cpu" ]; then

    module load python/3.6-anaconda-4.4
    module load gcc
    module load cuda

    : ${CONDA_ENVIRON:=tomopy-cpu}
    : ${BASE_DIR:=/project/projectdirs/m1759/jrmadsen}
    : ${GLOBUS_DIR:=${BASE_DIR}/globus}
    : ${SOURCE_DIR:=${PWD}}
    : ${SCRIPT_DIR:=${SOURCE_DIR}/slurm}
    # algorithm settings
    export TOMOPY_USE_C_ALGORITHMS=0
    export TOMOPY_DEVICE=cpu
    export BUILD_ARGS="${BUILD_CPU_ARGS}"

else

    echo "\nError! Unknown test mode: '${TEST_MODE}'"
    echo "\tValid options: 'c', 'cpu', 'gpu'\n"
    exit 1

fi

export BASE_DIR GLOBUS_DIR SOURCE_DIR SCRIPT_DIR

if [ -n "${CONDA_ENVIRON}" ];
    source activate ${CONDA_ENVIRON}
fi

# algorithm settings
: ${TOMOPY_INTER:=nn}
: ${TOMOPY_NUM_ITERATION:=50}
: ${DATA_TYPE:=partial}

# data settings
: ${BLOCK_BEG:=0}
: ${BLOCK_END:=24}
: ${NUM_SLICES:=$((${BLOCK_END} - ${BLOCK_BEG}))}

# parallelism settings
: ${PTL_CPU_AFFINITY:=0}
: ${PTL_VERBOSE:=1}
: ${TOMOPY_NUM_GPU:=8}
if [ "${NERSC_HOST}" = "edison" ]; then
    : ${TOMOPY_PYTHON_THREADS:=24}
else
    : ${TOMOPY_PYTHON_THREADS:=${TOMOPY_NUM_GPU}}
fi
: ${TOMOPY_NUM_THREADS:=$(( $(nproc) / ${TOMOPY_PYTHON_THREADS} + 2 ))}
: ${TOMOPY_BLOCK_SIZE:=32}
: ${TOMOPY_GRID_SIZE:=0}

PTL_NUM_THREADS=${TOMOPY_NUM_THREADS}
NUMEXPR_MAX_THREADS=$(nproc)

export TOMOPY_INTER TOMOPY_NUM_ITERATION DATA_TYPE
export BLOCK_BEG BLOCK_END NUM_SLICES
export PTL_CPU_AFFINITY PTL_VERBOSE PTL_NUM_THREADS NUMEXPR_MAX_THREADS
export TOMOPY_NUM_GPU TOMOPY_PYTHON_THREADS TOMOPY_NUM_THREADS
export TOMOPY_BLOCK_SIZE TOMOPY_GRID_SIZE

configure-out()
{
    echo "${TOMOPY_DEVICE}-${DATA_TYPE}-${TOMOPY_NUM_GPU}-${TOMOPY_INTER}-${TOMOPY_BLOCK_SIZE}-${TOMOPY_GRID_SIZE}-${TOMOPY_PYTHON_THREADS}-${TOMOPY_NUM_THREADS}-${TOMOPY_USE_C_ALGORITHMS}-${TOMOPY_USE_CPU}"
}

run-verbose()
{
    echo -e "\n### Running : '$@'... ###\n"
    eval $@
}

run-recon-phantom()
{
    export TOMOPY_INTER=${1}
    shift
    ALGORITHM=${2}
    shift
    PHANTOM=${3}
    shift
    EXTRA=""
    if [ "${PHANTOM}" = "shepp3d" ]; then EXTRA="--partial"; fi

    OUT="$(configure-out)"
    run-verbose srun -n 1 -c $(nproc) \
        $(which python) \
        ./pyctest_tomopy_phantom.py \
        -o ${OUT}-${ALGORITHM} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -r ${BLOCK_BEG} ${BLOCK_END} \
        -f png ${EXTRA} \
        -p ${PHANTOM} $@ | tee ${LOG_DIR}/run-${PHANTOM}-${ALGORITHM}-${OUT}.log
}

run-recon-rec()
{
    export TOMOPY_INTER=${1}
    shift
    ALGORITHM=${2}
    shift
    export DATA_TYPE=${3}
    shift

    OUT="$(configure-out)"
    run-verbose srun -n 1 -c $(nproc) \
        $(which python) \
        ./pyctest_tomopy_rec.py \
        -o ${OUT}-${ALGORITHM} \
        --type=${DATA_TYPE} \
        -n ${TOMOPY_PYTHON_THREADS} \
        -i ${TOMOPY_NUM_ITERATION} \
        -a ${ALGORITHM} \
        -r ${BLOCK_BEG} ${BLOCK_END} \
        -f png \
        $@ ${GLOBUS_DIR}/tomo_00001/tomo_00001.h5 | tee ${LOG_DIR}/run-${ALGORITHM}-${OUT}.log
}

run-verbose-smi()
{
    if [ "${BUILD_ARGS}" = "${BUILD_GPU_ARGS}" ]; then
        run-verbose srun nvidia-smi | tee ${LOG_DIR}/smi-${OUT}.log
    fi
}
