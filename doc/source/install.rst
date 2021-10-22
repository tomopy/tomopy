==================
Install directions
==================

This section covers the basics of how to download and install TomoPy.


Installing from Conda
---------------------

If you only want to run TomoPy, not develop it, then you should install through
Conda, our supported package and environment manager. Conda can install TomoPy
and its dependencies for you.

First, you must have `Conda <https://docs.conda.io/en/latest/miniconda.html>`_
installed.

Next, install TomoPy and all its runtime dependencies into a new Conda
environment called ``tomopy`` by running::

    $ conda create --name tomopy --channel conda-forge tomopy

Use this TomoPy installation by activating this environment::

    $ conda activate tomopy


Updating the installation
-------------------------

TomoPy is an active project, so we suggest you update your installation
frequently. To update the installation, activate the Conda environment
containing TomoPy and run::

    $ conda update --channel conda-forge tomopy

For some more information about using Conda, please refer to the `docs
<https://conda.io/projects/conda>`__.


Build from Source
-----------------

Please read the development guide for directions on how to build TomoPy from
the source code.
