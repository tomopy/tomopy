==========================
Frequently asked questions
==========================

.. contents::
   :local:


How can I report bugs?
======================

The easiest way to report bugs or get help is to open an issue on GitHub.
Simply go to the `project GitHub page <https://github.com/tomopy/tomopy>`_,
click on `Issues <https://github.com/tomopy/tomopy/issues>`_  in the
right menu tab and submit your report or question.


Are there any video tutorials?
==============================

We currently do not have specific plans in this direction, but we agree
that it would be very helpful.


Are there any segmentation routines?
====================================

No. The package does not include segmentation algorithms, since we are currently
focused on tomography, while we feel that segmentation are more part of the
application­ specific data analysis that occurs after tomographic processing. An
important exception is when segmentation steps are used as part of the
tomographic reconstruction algorithm, such as in the DART algorithm.


Are there any tools for aligning projections?
=============================================

Yes. Please check the `alignment
<https://tomopy.readthedocs.io/en/latest/api/tomopy.prep.alignment.html>`_
module for details.


What is the ASTRA toolbox?
==========================

The ASTRA toolbox provides highly efficient tomographic reconstruction methods
by implementing them on graphic processing units (GPUs). It includes advanced
iterative methods and allows for very flexible scanning geometries. The ASTRA
toolbox also includes building blocks which can be used to develop new
reconstruction methods, allowing for easy and efficient implementation and
modification of advanced reconstruction methods. However, the toolbox is only
focused on reconstruction, and does not include pre-processing or
post-processing methods that are typically required for correctly processing
synchrotron data. Furthermore, no routines to read data from disk are provided
by the toolbox.


Why were TomoPy and ASTRA integrated?
=====================================

The TomoPy toolbox is specifically designed to be easy to use and deploy at a
synchrotron facility beamline. It supports reading many common synchrotron data
formats from disk through the companion project, DXChange :cite:`DeCarlo:14a`,
and includes several other processing algorithms commonly used for synchrotron
data. Integrating the ASTRA toolbox into the TomoPy framework, brought optimized
GPU-based reconstruction methods to synchrotron beamline users at a time when
TomoPy had no GPU-based methods. Even today, the ASTRA toolbox offers faster
iterative methods than TomoPy.


What is UFO?
============

UFO is a general purpose image processing framework, optimized for heterogeneous
compute systems and streams of data. Arbitrary data processing tasks are plugged
together to form larger processing pipelines. These pipelines are then mapped to
the hardware resources available at run-time, i.e. all available multiple GPUs
and CPUs.

One specific use case that has been integrated into the TomoPy is fast
reconstruction using the filtered backprojection and direct Fourier inversion
methods although others for pre- and post-processing might be added in the
future.


Which platforms are supported?
==============================

TomoPy supports Windows, Linux, and macOS. The ASTRA toolbox supports Linux
and Windows. Project UFO supports Linux and macOS.


Do TomoPy, ASTRA, and UFO support all GPUs?
===========================================

The GPU algorithms in TomoPy and the ASTRA toolbox are all implemented using
nVidia's CUDA. As a result, only nVidia CUDA­ enabled GPUs can be used to run
them. UFO uses OpenCL, so it supports both AMD and nVidia OpenCL compatible
GPUs.


Can I run this on a HPC cluster?
================================

Maybe. In their default installation packages, TomoPy and the ASTRA toolbox are
limited to running on a single multi-core and multi-GPU machine. Both toolboxes
can be run on a HPC cluster through parallelization using MPI, but since
installation and running on a HPC cluster is often cluster­ specific, the
default installation packages do not include these capabilities.

As such, the integrated packages that is presented in the manuscript currently
does not support running on a HPC cluster. Note that the ASTRA toolbox provides
a separate MPI­ enabled package for use on a HPC cluster. We refer to
:cite:`Bicer:15` for more details about TomoPy's planned HPC implementation. It
is a MapReduce type MPI implementation layer, which was successfully used on
many clusters,  i.e. Stampede, Cori, Mira. There are plans to allow user access
to TomoPy on a HPC cluster (e.g. through a client or web­portal), but these
projects will take some time before they are being matured for user’s use.


Why can't I install TomoPy from PyPI using pip?
===============================================

pip wasn't designed to manage non-Python packages, and TomoPy has non-Python
dependencies. Our preferred package and environment manager, conda, makes it
easier for us (the developers) to build and distribute TomoPy.
