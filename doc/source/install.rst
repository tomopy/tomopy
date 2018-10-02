==================
Install directions
==================

This section covers the basics of how to download and install TomoPy.

.. contents:: Contents:
   :local:

Supported Environments
======================

TomoPy is tested, built, and distributed for python 2.7 3.5 3.6 on Linux/macOS
and python 3.5 3.6 on Windows 10.

Installing from Conda (Recommended)
===================================

If you only want to run TomoPy, not develop it, then you should install through
a package manager. Conda, our supported package manager, can install TomoPy and
its dependencies for you.

First, you must have `Conda <http://continuum.io/downloads>`_ installed,
then open a terminal or a command prompt window and run::

    $ conda install -c conda-forge tomopy

This will install TomoPy and all the dependencies from the conda-forge channel.

Updating the installation
-------------------------

TomoPy is an active project, so we suggest you update your installation
frequently. To update the installation run::

    $ conda update -c conda-forge tomopy

For some more information about using Conda, please refer to the
`docs <http://conda.pydata.org/docs>`__.

Installing from source with Conda
=================================

Sometimes an adventurous user may want to get the source code, which is
always more up-to-date than the one provided by Conda (with more bugs of
course!).

For this you need to get the source from the
`TomoPy repository <https://github.com/tomopy/tomopy>`_ on GitHub.
Download the source to your local computer using git by opening a
terminal and running::

    $ git clone https://github.com/tomopy/tomopy.git

in the folder where you want the source code. This will create a folder called
`tomopy` which contains a copy of the source code.


Installing dependencies
-----------------------

You will need to install all the dependencies listed in
``requirements.txt`` or ``meta.yaml`` files. For example, requirements can be
installed using Conda by running::

    $ conda install --file requirements.txt

After navigating to inside the `tomopy` directory, you can install TomoPy by
building/compiling the shared libraries and running the install script::

    $ python build.py
    $ pip install .

Common issues
-------------

No issues with the current build system have been reported.

Importing TomoPy
================

When importing, it is best to import TomoPy before importing numpy.
See `this thread <https://github.com/tomopy/tomopy/issues/178>`_ for details.
