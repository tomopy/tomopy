# TomoPy Benchmarking via PyCTest

[PyCTest](https://github.com/jrmadsen/pyctest) is a Python package under development at NERSC.
This package creates Python bindings to CMake and CTest to easily run
tests for benchmarking (via CTest) and display the results on a public web-server (via CDash).

TomoPy is a perfect example of why [PyCTest](https://github.com/jrmadsen/pyctest) was created:
TomoPy is a Python package and has no need for a CMake build system but it can greatly benefit
from using CTest and CDash.

[CDash Testing Dashboard @ NERSC documentation](https://docs.nersc.gov/services/cdash/)

[TomoPy Testing Dashboard @ NERSC](https://cdash.nersc.gov/index.php?project=TomoPy)

## Concept

This implementation provides a demonstration on how to publish benchmarking results to a
public web-server in a manner than can be automated -- such as, setting up a cron job
on [Cori @ NERSC](http://www.nersc.gov/users/computational-systems/cori/) that runs
a set of tests when updates to the master branch of TomoPy are detected.

The set of tests provide a history of the performance and validity of TomoPy

- Timing (real and CPU) and memory consumptions
  - Plots
  - ASCII results
- Display of reconstruction images
- Comparison of the reconstruction vs. known solution
  - Display images of known solution alongside reconstruction
  - Display images of difference between known solution alongside reconstruction
  - Quantification of differences
    - L1 and L2 norm of pixel difference
    - L1 and L2 norm of image gradient difference

## Implementation

### `PyCTestPreInit.cmake`

- General purpose:
  - Configure or download any packages (e.g. dependent packages)
  - Setup general environment: `set(ENV{PATH} "/some/bin/path:$ENV{PATH}")`
- TomoPy purpose:
  - Downloads and installs Miniconda if not `CONDA_EXE` not found in environment
  - Installs dependent packages for TomoPy
  - Ensures proper conda environment variables are set

### `tomopy_benchmark.py`

- Main `PyCTest` driver
- `pyctest.helpers.ParseArgs(...)`
  - Adds PyCTest `argparse` options and used to specify 3 key components:
    - `project_name` : the name of the project
    - `source_dir` : location of the source code
    - `binary_dir` : location of build and testing output
      - In general, this should be different from source directory for easy removal
      - Be careful about placing the binary directory inside the source directory: copying the source directory to the binary directory will result in a recursive copy
- Adds additional `argparse` options for tests
  - Number of cores
  - Number of iterations
  - Phantoms to test
  - Algorithms to test
  - Globus data path
- Checks out source code: `pyctest.git_checkout(...)`
- Copies files `PyCTestPreInit.cmake`, `tomopy_phantom.py` and `tomopy_rec.py`
- Specifies how to build TomoPy C bindings
  - Warnings/errors are logged in ___Build___ section of dashboard
- Tests
  - Generates test for checking that `import tomopy` is importing from correct path
  - Generates test for `nosetests` unit testing
  - Generates per-algorithm tests for `tomo_00001/tomo_00001.h5` using `tomopy_rec.py` script
    - If `--globus-path` option provides a path to this dataset
  - Generates per-phantom algorithm comparison tests using `tomopy_phantom.py` script
- Submits to [TomoPy CDash Testing Dashboard @ NERSC](https://cdash.nersc.gov/index.php?project=TomoPy)
  - Timing and memory plots provided via [TiMemory](https://github.com/jrmadsen/TiMemory)
  - Timing and memory ASCII results attached as CTest notes
  - Attaches images from reconstruction (e.g. Dart measurement files)

### `tomopy_rec.py`

- Used by per-algorithm tests configured in `tomopy_benchmark.py`
- Customized implementation of `tomopy_rec.py` from [TomoBank documentation](http://tomobank.readthedocs.io/en/latest/index.html)
  - Enables [TiMemory](https://github.com/jrmadsen/TiMemory) for timing + memory results
  - Enables selection of algorithms other than `gridrec`
  - Enables image output in non-TIFF formats

### `tomopy_phantom.py`

- Used by per-phantom algorithm comparison tests in `tomopy_benchmark.py`

## Usage

The [TomoPy Testing Dashboard @ NERSC](https://cdash.nersc.gov/index.php?project=TomoPy)
is restricted to users with a CDash token. Please contact [Jonathan Madsen](mailto:jrmadsen@lbl.gov)
to request a token or refer to the
[CDash Testing Dashboard @ NERSC documentation](https://docs.nersc.gov/services/cdash/)
for instructions for registering an account and configuring a new project as a public or private dashboard.

Install `PyCTest` via Anaconda (pre-compiled, recommended) or PyPi (source dist).

### Anaconda

[![Anaconda-Server Badge](https://anaconda.org/jrmadsen/pyctest/badges/installer/conda.svg)](https://conda.anaconda.org/jrmadsen)
[![Anaconda-Server Badge](https://anaconda.org/jrmadsen/pyctest/badges/platforms.svg)](https://anaconda.org/jrmadsen/pyctest)

- Python versions available
  - `2.7`
  - `3.5`
  - `3.6`

```bash
$ conda config --add channels jrmadsen
$ conda create -n pyctest python=3.6 pyctest
$ source activate pyctest
$ ./benchmark_tomopy.py
```

### PyPi

```bash
$ pip install pyctest
```
