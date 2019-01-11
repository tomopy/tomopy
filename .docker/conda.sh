#!/bin/bash -e

: ${PYTHON_VERSION:=3.6}

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

bash miniconda.sh -b -p /opt/conda

export PATH="/opt/conda/bin:${PATH}"

conda config --add channels jrmadsen
conda config --add channels conda-forge
conda config --set always_yes yes --set changeps1 yes

conda update conda

PYTHON_TAG=$(echo ${PYTHON_VERSION} | sed 's/\.//g')
conda env create -n tomopy -f /work/envs/tomopy-python${PYTHON_TAG}.yml

conda clean -a -y
