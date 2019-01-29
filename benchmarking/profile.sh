#!/bin/bash -l

LOG=${1}
shift
nvprof -s -u col --log-file ${LOG} $@
