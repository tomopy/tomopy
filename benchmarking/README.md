# TomoPy Benchmarking with PyCTest

## Overview

- Using PyCTest, the process of building and testing TomoPy is captured and submitted to [CDash @ NERSC](https://cdash.nersc.gov)
- Results can be found at the [TomoPy Testing Dashboard](https://cdash.nersc.gov/index.php?project=TomoPy)
- All build warnings and errors are parsed and reported by CDash
- In addition to running `nosetests`, several other tests are generated and executed
    - CDash captures the command executed and all of the output
    - Detailed timing and memory reports on TomoPy are integrated into these tests and submitted to CDash in both plotted and ASCII forms
        - Timing and memory plots analysis is done through [TiMemory](https://github.com/jrmadsen/TiMemory)
    - Additionally, the results of the TomoPy tests (i.e. image reconstructions) are also uploaded tp CDash

## Installation

- Anaconda
```shell
$ conda install -c jrmadsen -n <env> pyctest timemory
```

- PyPi
```shell
$ pip install -v pyctest timemory
```

## Execution

- Submission is invoke by running `./pyctest_tomopy.py` from the main directory
- Run `./pyctest_tomopy.py --help` to view the list of available configuration options
- Supplementary files
    - `./benchmarking/pyctest_tomopy_rec.py` is a TomoPy reconstruction script for TomoBank data
        - When the option `--globus-path` is specified, PyCTest will reconstruct `tomo_00001/tomo_00001.h5` with all the specified algorithms
    - `./benchmarking/pyctest_tomopy_phantom.py` is a TomoPy reconstruction script for built-in phantoms
- The default assumption is that the current Python environment has all of the required packages for TomoPy

### Example

```shell
$ export PYTHON_VERSION=3.6

$ export CONDA_ENV="tomopy-pyctest"

$ conda install -n ${CONDA_ENV} -c conda-forge -c jrmadsen python=${PYTHON_VERSION} nose six numpy h5py scipy scikit-image pywavelets mkl-devel mkl_fft python-coveralls dxchange numexpr coverage timemory pyctest

$ source activate ${CONDA_ENV}

$ ./pyctest_tomopy.py --globus-path=${HOME}/devel/globus --num-iter=10 --pyctest-site="Cori-Haswell" --pyctest-token-file="${HOME}/.tokens/nersc-cdash"

```
