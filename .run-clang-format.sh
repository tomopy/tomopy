#!/bin/bash -l

: ${CLANG_FORMATTER:=$(which clang-format)}

if [ -n "${1}" ]; then CLANG_FORMATTER=${1}; fi

for i in $(find src include -type f | egrep -v 'pybind11|PTL' | egrep '\.h$|\.hh$|\.hpp$|\.c$|\.cc$|\.cpp$|\.cu$|\.icc$')
do
    ${CLANG_FORMATTER} -i ${i}
done
