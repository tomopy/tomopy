=================
Developer's Guide
=================

:Page Status: Incomplete
:Last Reviewed: 2015-03-14


This section is for TomoPy developers.

.. contents:: Contents
   :local:


For Beginners
=============

Short descriptions of some of the files in the root folder of the project:

* .gitignore : Specifies intentionally untracked files that 
  `Git <http://git-scm.com>`_ should ignore.

* bld.bat : Windows build script which is executed using ``cmd``.

* build.sh : Unix build script which is executed using bash. 

* MANIFEST.in : Specifies the files in the project that will go into the 
  final distribution.

* meta.yaml : Contains metadata for the 
  `Conda <http://conda.pydata.org/docs/build.html>`_ build.

* requirements.txt : Contains a list of items to be installed using 
  `pip install <https://pip.pypa.io/en/latest/user_guide.html#requirements-files>`_

* setup.cfg : Contains the distribution's optional metadata and build 
  configuration which are not provided in ``setup.py``

* setup.py : Contains metadata and configuration for building and 
  installing the package distribution. 


Coding Syntax
=============

TomoPy uses the following style guides for code development:

1. `PEP7 <https://www.python.org/dev/peps/pep-0007/>`_ for Python 
   codes.

2. `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ for C codes.

3. `PEP287 <https://www.python.org/dev/peps/pep-0287/>`_ for 
   Python docstring

It is recommended to use the Python packages 
`pep8 <https://pypi.python.org/pypi/pep8>`_ and 
`pyFlakes <https://pypi.python.org/pypi/pyflakes>`_ to check for
syntax and warnings. They are available in
`Conda <http://docs.continuum.io/anaconda/pkg-docs.html>`_
as well as in `PyPI <https://pypi.python.org>`_.


Conda Packaging
===============

Run the following command from a terminal to build the 
`Conda <https://store.continuum.io>`_  package for TomoPy::

    conda build /path/to/tomopy/folder


Using Git
=========

Here is a list of descriptions for commit messages:

* API: api related commits

* BLD: changes related to building

* BUG: bug fixes

* DOC: documentation

* ENH: enhancement

* MNT: maintenance

* STR: code restructuring, moving files

* STY: style fixes

* TST: addition or modification of tests

* WIP: for work in progress


Testing
=======

Add the test scripts for the package or module in ``tomopy/tests``. To see
the test coverage use the following command::

    nosetests

See ``setup.cfg`` for `nose <http://nose.readthedocs.org/en/latest/index.html>`_ configuration.
