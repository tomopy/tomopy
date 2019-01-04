#!/bin/bash

set -o errexit

SOURCE_DIR=${1}
GCOV_CMD=$(which gcov)
COVERAGE_CMD=$(which coverage)

if [ -z "${COVERAGE_CMD}" ]; then exit 0; fi
if [ -z "${SOURCE_DIR}" ]; then exit 0; fi

cd ${SOURCE_DIR}

coverage xml
mv coverage.xml .coverage.xml
