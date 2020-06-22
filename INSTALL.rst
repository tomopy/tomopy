Installing dependencies
=======================
To build and run TomoPy, you will need to install at least the dependencies
listed in ``envs/{platform}-{version}.yml`` plus additional dependencies based
on your platform. For example, installing requirements for building
the Python 3.6 version on Linux can be accomplished as follows::

    $ conda env create -f envs/linux-36.yml [-n ENVIRONMENT]


Additional Windows Requirements
-------------------------------
The Windows VC++2017 compiler cannot be distributed through conda. You must
install it using the Windows Build Tools installer. The conda package for this
compiler merely searches for this compiler and links it to the conda
environment into which it is installed.


Building TomoPy
===============

After navigating to inside the `tomopy` directory, you can install TomoPy by
running the install script in the typical Python way::

    $ python setup.py install
