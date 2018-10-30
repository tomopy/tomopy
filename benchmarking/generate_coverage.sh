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

if [ -n "$(find . | grep '\.gcda')" ]; then
    for i in ${PWD}/*.o
    do
        ${GCOV_CMD} ${i} -m
    done

    for i in ${PWD}/*.gcov
    do
        mv ${i} ${SOURCE_DIR}/
    done
fi
