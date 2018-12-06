#!/bin/bash

set -o errexit

: ${PYTHON_VERSION:=3.6}

if [[ "$PYTHON_VERSION" = "2.7" ]]; then
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi

bash miniconda.sh -b -p /opt/conda

export PATH="/opt/conda/bin:${PATH}"
mkdir -p /etc/bashrc.d
cat << EOF > /etc/profile.d/conda.sh
#!/bin/bash

if [ -z "$(which conda)" ]; then PATH="/opt/conda/bin:${PATH}"; fi
export PATH
source deactivate
source activate tomopy
EOF

conda config --add channels conda-forge
conda config --add channels jrmadsen
conda config --set always_yes yes --set changeps1 yes
conda update conda

conda create -n tomopy python=${PYTHON_VERSION} nose six numpy h5py scipy \
    scikit-image scikit-build pywavelets mkl-devel mkl_fft python-coveralls \
    dxchange numexpr matplotlib pillow cython timemory pyctest
