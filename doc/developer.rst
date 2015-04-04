=================
Developer's Guide
=================

:Page Status: Incomplete
:Last Reviewed: 2015-03-14


This section is for TomoPy developers.

.. contents:: Contents
   :local:


How to contribute?
==================

The project is maintained on GitHub, which is a version control and a 
collaboration platform for software developers. To start first register 
on `GitHub <https://github.com>`_ and fork the TomoPy repository by 
clicking the **Fork** button in the header of the 
`TomoPy repository <https://github.com/dgursoy/tomopy>`__: 

.. image:: img/fork-repo.png

At this point you've successfully forked the project to your personal
GitHub account. The next thing you want to do is to clone it to your 
local machine. You can do this either by opening a terminal window and
running::

    git clone https://github.com/YOUR-USERNAME/tomopy.git

or clicking the **Clone in Desktop** button in the bottom of the right 
hand side bar and following the instructions on downloading and using the
GitHub Desktop Application: 

.. image:: img/clone-in-desktop.png

Git commit messages
------------------- 

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

Coding Syntax
=============

TomoPy uses the following style guides for code development:

1. `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__ for Python 
   codes.

2. `PEP7 <https://www.python.org/dev/peps/pep-0007/>`_ for C codes.

3. `PEP287 <https://www.python.org/dev/peps/pep-0287/>`_ for 
   Python docstring

It is recommended to use the Python packages 
`pep8 <https://pypi.python.org/pypi/pep8>`__ and 
`pyFlakes <https://pypi.python.org/pypi/pyflakes>`_ to check for
syntax and warnings. They are available in
`Conda <http://docs.continuum.io/anaconda/pkg-docs.html>`__
as well as in `PyPI <https://pypi.python.org>`_.


Conda Packaging
===============

Run the following command from a terminal to build the 
`Conda <https://store.continuum.io>`__  package for TomoPy::

    conda build /path/to/tomopy/folder


Testing
=======

Add the test scripts for the package or module in ``tomopy/tests``. To see
the test coverage use the following command::

    nosetests

See ``setup.cfg`` for `nose <http://nose.readthedocs.org/en/latest/index.html>`_ configuration.
