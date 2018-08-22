==================
Install directions
==================

This section covers the basics of how to download and install TomoPy.

.. contents:: Contents:
   :local:

Supported Operating Systems
===========================

Since most of the developers use Unix systems (Linux and macOS), TomoPy is only
tested and officially supported on Unix systems. Windows installation and
functionality is not guaranteed.

Installing from Conda (Recommended)
===================================

If you only want to run TomoPy, not develop it, then you should install through
a package manager. Conda, our supported package manager, can install TomoPy and
its dependencies for you.

First, you must have `Conda <http://continuum.io/downloads>`_ installed,
then open a terminal or a command prompt window and run::

    $ conda install -c conda-forge tomopy

This will install TomoPy from the conda-forge channel. Conda should also handle
installing all of the dependencies.

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
always more up-to-date than the one provided by Conda
(with more bugs of course!).

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
activating the Conda base environment and running the install script::

    $ source activate
    (base) $ python setup.py install

Activating the conda environment ensures that `setup.py` will know where to
look for the mkl headers. If you didn't use Conda to install the mkl library,
you may need to read the section below.

Common issues
-------------

The following error from `gcc` states that it cannot find the `mkl` header file
it needs to compile the c extensions of TomoPy::

    src/gridrec.c:56:21: fatal error: mkl.h: No such file or directory

You will need to add the
location of your `mkl` header file to the `C_INCLUDE_PATH` by setting an
environmental variable or changing line 30 of `setup.py <https://github.com/tomopy/tomopy/blob/29949bee02a1b620c6af247faded01cffa3d336f/setup.py#L28-L30>`_

If you installed `mkl-devel` using conda, the header files should be located
in the `foo/conda/include` directory where `foo` is the install location of
conda on your computer. By default this folder is your Home directory.

Importing TomoPy
================

When importing, it is best to import TomoPy before importing numpy.
See `this thread <https://github.com/tomopy/tomopy/issues/178>`_ for details.
