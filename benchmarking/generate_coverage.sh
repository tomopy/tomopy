#!/bin/bash

set -o errexit

SOURCE_DIR=${1}
GCOV_CMD=$(which gcov)
COVERAGE_CMD=$(which coverage)

if [ -z "${COVERAGE_CMD}" ]; then exit 0; fi
if [ -z "${SOURCE_DIR}" ]; then exit 0; fi

cd ${SOURCE_DIR}

coverage xml

if [ -z "${GCOV_CMD}" ]; then exit 0; fi

cd ${SOURCE_DIR}/config

mkdir -p ${SOURCE_DIR}/CMakeFiles
cat << EOF > ${SOURCE_DIR}/CMakeFiles/TargetDirectories.txt
${SOURCE_DIR}/src
${SOURCE_DIR}/config
EOF

if [ ! -d "${SOURCE_DIR}/Testing" ]; then mkdir -p ${SOURCE_DIR}/Testing ; fi
cp -r ${SOURCE_DIR}/src ${SOURCE_DIR}/Testing
