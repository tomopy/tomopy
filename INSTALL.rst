Installing dependencies
=======================
To build and run TomoPy, you will need to install at least the dependencies
listed in ``envs/{platform}-{version}.yml`` plus additional dependencies based
on your platform. For convenience, installing requirements for building
the Python 3.6 version on Linux can be accomplished as follows::

    $ conda env create -f envs/linux-36.yml [-n ENVIRONMENT]
    
This will create a new conda environment named tomopy with the build
dependencies. If you already have a conda environment named tomopy. Use the
`-n` option to name it something else.

Additional Windows Requirements
-------------------------------
The Windows VC++2017 compiler cannot be distributed through conda. The conda
package for this compiler creates link from the system provided compiler
to the conda environment into which it is installed. Install VC++2017 using
the Windows Build Tools installer; Visual Studio (the IDE) is not required
to build TomoPy.

Additional CUDA Requirements
----------------------------
The CUDA compiler cannot be distributed through conda. Building TomoPy with
GPU features requires the CUDA Toolkit and NVCC.

Building TomoPy
===============

After navigating to inside the `tomopy` directory, you can install TomoPy by
running the install script in the typical Python way::

    $ python setup.py install
