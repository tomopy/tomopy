==========================
Frequently asked questions
==========================

Here's a list of questions.

.. contents:: Questions
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


Can I run this on a HPC cluster?
================================

In their default installation packages, TomoPy and the ASTRA toolbox are 
limited to running on a multicore single machine. The ASTRA toolbox, and 
TomoPy through the presented ASTRA integration, are able to use multiple 
GPUs that are installed in a single machine. Both toolboxes can be run on 
a HPC cluster through parallelization using MPI, but since installation 
and running on a HPC cluster is often cluster­ specific, the default 
installation packages do not include these capabilities.   

As such, the integrated packages that is presented in the manuscript 
currently does not support running on a HPC cluster. Note that the ASTRA 
toolbox provides a separate MPI­ enabled package for use on a HPC cluster. 
We refer to :cite:`Bicer:15` for more details about TomoPy's planned HPC 
implementation. It is a MapReduce type MPI implementation layer, which 
was succesfully used on many clusters,  i.e. Stampede, Cori, Mira. 
There are plans to allow user access to TomoPy on a HPC cluster 
(e.g. through a client or web­portal), but these projects will 
take some time before they are being matured for user’s use.


Are there any segmentation routines?
====================================

Some data processing operations can be applied after reconstruction. 
Examples of these type of operations are image­ based ring­ removal methods, 
and gaussian­ filtering or median­ filtering the reconstructed image. Typicaly, 
these methods are called "post­processing algorithms, since they occur after 
the reconstruction.

The package does not include segmentation algorithms, since we are currently
focused on tomography, while we feel that segmentation are more part of the 
application­ specific data analysis that occurs after tomographic processing. 
An important exception is when segmentation steps are used as part of the
tomographic reconstruction algorithm, such as in the DART algorithm.


Are there any tools for aligning projections?
=============================================

Yes we have. Please check the 
`Examples <http://tomopy.readthedocs.io/en/latest/demo.html#>`_ 
section for details.


What is ASTRA toolbox?
======================

The ASTRA toolbox provides highly efficient tomographic reconstruction 
methods by implementing them on graphic processing units (GPUs). It 
includes advanced iterative methods and allows for very flexible scanning 
geometries. The ASTRA toolbox also includes building blocks which can be 
used to develop new reconstruction methods, allowing for easy and efficient 
implementation and modification of advanced reconstruction methods. 
However, the toolbox is only focused on reconstruction, and does not 
include pre-processing or post-processing methods that are typically 
required for correctly processing synchrotron data. Furthermore, no
routines to read data from disk are provided by the toolbox.


Why TomoPy and ASTRA were integrated?
=====================================

The TomoPy toolbox is specifically designed to be easy to use and deploy 
at a synchrotron facility beamline. It supports reading many common 
synchrotron data formats from disk :cite:`DeCarlo:14a`, and includes 
several other processing algorithms commonly used for synchrotron data. 
TomoPy also includes several reconstruction algorithms, which can be run 
on multi-core workstations and large-scale computing facilities. The 
algorithms in TomoPy are all CPU-based, however, which can make them 
prohibitively slow in the case of iterative methods, which are often 
required for advanced tomographic experiments.

By integrating the ASTRA toolbox in the TomoPy framework, the optimized 
GPU-based reconstruction methods become easily available for synchrotron 
beamline users, and users of the ASTRA toolbox can more easily read data 
and use TomoPy’s other functionality for data filtering and cleaning.


Which platforms are supported?
==============================

TomoPy supports Linux and Mac OS X, and the ASTRA toolbox supports Linux 
and Windows. As such, the combined package currently supports only Linux, 
but we are working on supporting more operating systems.


Does ASTRA support all GPUs? 
============================

The GPU algorithms are all implemented used nVidia CUDA. As a result, 
only nVidia CUDA­ enabled video cards can be used to run them.
