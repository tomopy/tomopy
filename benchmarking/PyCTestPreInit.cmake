#------------------------------------------------------------------------------#
#
#   The PyCTestPreInit.cmake script is called before any build, tests, etc.
#   are run.
#
#   This script will be copied over to pyctest.pyctest.BINARY_DIRECTORY
#   so relative paths should be w.r.t. to the BINARY_DIRECTORY
#
#   The intention of this script is to setup the build + testing environment
#
#   - J. Madsen (July 5th, 2018)
#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
#   echo message for debug
#
message(STATUS "Including ${CMAKE_CURRENT_LIST_FILE}...")


#------------------------------------------------------------------------------#
# provides macros:
#   download_conda(...)
#   find_conda(...)
#   configure_conda(...)
#
include("${CMAKE_CURRENT_LIST_DIR}/Utilities.cmake")


#------------------------------------------------------------------------------#
# determine if already using conda
#------------------------------------------------------------------------------#
set(EXISTING_CONDA_EXE "$ENV{CONDA_EXE}")
set(EXISTING_CONDA_PYTHON_EXE "$ENV{CONDA_PYTHON_EXE}")
if(EXISTING_CONDA_EXE)
    # get the base prefix
    execute_process(COMMAND ${EXISTING_CONDA_EXE} info --base
        OUTPUT_VARIABLE MINICONDA_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # get a random environment
    execute_process(COMMAND ${EXISTING_CONDA_PYTHON_EXE} -c
        "import random; import string; random = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)]); print('tomopy-pyctest-{}'.format(random))"
        OUTPUT_VARIABLE MINICONDA_ENVIRONMENT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # get the python version of conda install
    execute_process(COMMAND ${EXISTING_CONDA_PYTHON_EXE} -c
        "import sys; print('{}.{}'.format(sys.version_info[0], sys.version_info[1]))"
        OUTPUT_VARIABLE MINICONDA_PYTHON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    # set PYTHON_VERSION in environment
    set(ENV{PYTHON_VERSION} ${MINICONDA_PYTHON_VERSION})
    set(ENV{MINICONDA_PACKAGE_INSTALL} ON)
    message(STATUS "Using ${MINICONDA_PREFIX}...")
else(EXISTING_CONDA_EXE)
    # prefix for miniconda install
    set(MINICONDA_PREFIX "${CMAKE_CURRENT_LIST_DIR}/Miniconda")
    # environment to install to
    set(MINICONDA_ENVIRONMENT "root")
    # control whether to download + install when not necessary
    set(MINICONDA_PACKAGE_INSTALL OFF)
endif(EXISTING_CONDA_EXE)


#------------------------------------------------------------------------------#
# Determine Python version
#------------------------------------------------------------------------------#
macro(SET_PYTHON_VERSION VAL)
    if(NOT DEFINED PYTHON_VERSION OR "${PYTHON_VERSION}" STREQUAL "")
        if(NOT "${VAL}" STREQUAL "")
            set(PYTHON_VERSION "${VAL}")
        endif(NOT "${VAL}" STREQUAL "")
    endif(NOT DEFINED PYTHON_VERSION OR "${PYTHON_VERSION}" STREQUAL "")
endmacro(SET_PYTHON_VERSION VAL)

set_python_version("$ENV{PYTHON_VERSION}")
set_python_version("$ENV{TRAVIS_PYTHON_VERSION}")
set_python_version("2.7")


#------------------------------------------------------------------------------#
# set some values
#------------------------------------------------------------------------------#
set(ENV{PYTHONPATH} "") # conda doesn't like PYTHONPATH
# packages
set(MINICONDA_PACKAGES
    nose six numpy h5py scipy scikit-image pywavelets mkl-devel
    mkl_fft python-coveralls dxchange numexpr timemory)

if(NOT "$ENV{CONDA_PACKAGE_INSTALL}" STREQUAL "" OR
   NOT "$ENV{MINICONDA_PACKAGE_INSTALL}" STREQUAL "")
    set(MINICONDA_PACKAGE_INSTALL ON)
endif()

if("${PYTHON_VERSION}" VERSION_EQUAL "2.7")
    list(APPEND MINICONDA_PACKAGES "futures")
endif()


#------------------------------------------------------------------------------#
#   download Miniconda if not already exists
#------------------------------------------------------------------------------#
# if not already installed
if(NOT EXISTS "${MINICONDA_PREFIX}/bin/conda")

    download_conda(
        VERSION "latest"
        PYTHON_VERSION ${PYTHON_VERSION}
        INSTALL_PREFIX ${MINICONDA_PREFIX}
        ARCH "x86_64"
        DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}")

    set(MINICONDA_PACKAGE_INSTALL ON)

endif(NOT EXISTS "${MINICONDA_PREFIX}/bin/conda")


#------------------------------------------------------------------------------#
#   setup
#------------------------------------------------------------------------------#
find_conda(${MINICONDA_PREFIX} ${MINICONDA_ENVIRONMENT})

execute_process(COMMAND
    ${CONDA_EXE} create -n ${MINICONDA_ENVIRONMENT} python=${PYTHON_VERSION}
    ERROR_QUIET)

set(MINICONDA_PREFIX ${MINICONDA_PREFIX}/envs/${MINICONDA_ENVIRONMENT})

configure_conda(
    PYTHON_VERSION      ${PYTHON_VERSION}
    PREFIX              ${MINICONDA_PREFIX}
    ENVIRONMENT         ${MINICONDA_ENVIRONMENT}
    PACKAGES            ${MINICONDA_PACKAGES}
    CHANNELS            conda-forge jrmadsen)

find_conda(${MINICONDA_PREFIX} ${MINICONDA_ENVIRONMENT})
# tomopy has errors if install is not called a second time on KNL
execute_process(COMMAND ${CONDA_EXE} install -n ${MINICONDA_ENVIRONMENT}
    python=${PYTHON_VERSION} ${MINICONDA_PACKAGES})

configure_conda(
    PYTHON_VERSION      ${PYTHON_VERSION}
    PREFIX              ${MINICONDA_PREFIX}
    ENVIRONMENT         ${MINICONDA_ENVIRONMENT}
    PACKAGES            scipy
    CHANNELS            conda-forge jrmadsen)

find_conda(${MINICONDA_PREFIX} ${MINICONDA_ENVIRONMENT})
