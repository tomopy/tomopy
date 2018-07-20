# TomoPy Example

## Overview

- Uses CMake macros provided by package for Miniconda/Anaconda installation
  - Downloads Miniconda in `PyCTestPreInit.cmake`
  - Installs Miniconda and required packages in `PyCTestPreInit.cmake`
  - Ensures proper conda environment variables are set
- Checks out source code of Python package via `pyctest.pyctest.git_checkout("https://github.com/tomopy/tomopy.git", "tomopy-src")`
- Build source code via `python setup.py install` when running CTest
  - Warnings are logged in "Build" section of dashboard
- Generates a test around the `nosetests` unit testing
- If `--globus-path` options is specified for a path to `tomo_00001/tomo_00001.h5`, generates tests calling `tomopy_rec.py`
- Generates tests around several algorithms by calling `run_tomopy.py`
- Submits to CDash dashboard at [NERSC CDash Testing Dashboard](https://cdash.nersc.gov)
- Timing and memory plots provided via [TiMemory](https://github.com/jrmadsen/TiMemory)
- Attaches CTest notes (e.g. ASCII results)
- Attaches images from reconstruction (e.g. Dart measurement files)

Results from running the TomoPy example can be found at the [TomoPy CDash Testing Dashboard @ NERSC](https://cdash.nersc.gov/index.php?project=TomoPy)

### Dependencies
- Python packages
  - gitpython (optional)

### Optional setup
$ git clone https://github.com/tomopy/tomopy.git tomopy-src

### Run and submit to dashboard
```bash
$ ./tomopy.py
```
