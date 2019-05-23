#!/bin/bash -e

set -v

SOURCE_DIR=${1}
GCOV_CMD=$(which gcov)
COVERAGE_CMD=$(which coverage)
SKBUILD_OS="linux"
OUTPUT_DIR=${SOURCE_DIR}/CMakeFiles

if [ -z "${COVERAGE_CMD}" ]; then exit 0; fi
if [ -z "${SOURCE_DIR}" ]; then exit 0; fi
if [ "$(uname)" = "Darwin" ]; then SKBUILD_OS="macosx"; fi

cd ${SOURCE_DIR}

# Python coverage
coverage xml

if [ -f "${PWD}/coverage.xml" ]; then
    mv coverage.xml .coverage.xml
fi

if [ -z "${GCOV_CMD}" ]; then exit 0; fi

# GNU coverage
mkdir -p ${OUTPUT_DIR}

shopt -s nullglob
for i in ${PWD}/_skbuild/*
do
    if [ -n "$(echo ${i} | grep ${SKBUILD_OS})" ]; then
        find ${i} -type f | grep "TargetDirectories" | xargs cat > ${OUTPUT_DIR}/TargetDirectories.txt
    fi
done

if [ ! -d "${SOURCE_DIR}/Testing" ]; then mkdir -p ${SOURCE_DIR}/Testing ; fi
cp -r ${SOURCE_DIR}/source ${SOURCE_DIR}/Testing/
