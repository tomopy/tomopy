=================
Developer's Guide
=================

:Page Status: Incomplete
:Last Reviewed: 2015-03-14


This section is for TomoPy developers.

.. contents:: Contents
   :local:


Coding Syntax
=============

TomoPy uses the following style guides for code development:

1. `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ for Python 
   codes.

2. `PEP7 <https://www.python.org/dev/peps/pep-0007/>`_ for C codes.

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
