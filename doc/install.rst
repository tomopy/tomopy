==================
Install Directions
==================

:Page Status: Incomplete
:Last Reviewed: 2015-03-23


This section covers the basics of how to install TomoPy.

.. contents:: Contents
   :local:


Installing from Conda/Binstar
=============================

To use this you must have `Conda <https://store.continuum.io/>`_
installed first. Open a terminal and run::

    conda install -c dgursoy tomopy


Installing from GitHub
======================

You need to have `Git <http://git-scm.com>`_ installed. Then:

1. Go to a directory where you would like to put the package.

2. Clone the project from GitHub repository::

    git clone https://github.com/dgursoy/tomopy.git

3. Go to the root directory of the project and run the 
   following command from a terminal::

    python setup.py install


Updating the Installation
=========================

To update the installation use::

    conda update -c dgursoy tomopy
