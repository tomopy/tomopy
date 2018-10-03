
LPrec
-----

Here is an example on how to use the log-polar based method
(https://github.com/math-vrn/lprec) for reconstruction in Tomopy

.. code:: ipython2

    %pylab inline


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


Install lprec from github, then

.. code:: ipython2

    import tomopy

`DXchange <http://dxchange.readthedocs.io>`__ is installed with tomopy
to provide support for tomographic data loading. Various data format
from all major
`synchrotron <http://dxchange.readthedocs.io/en/latest/source/demo.html>`__
facilities are supported.

.. code:: ipython2

    import dxchange

matplotlib provide plotting of the result in this notebook.
`Paraview <http://www.paraview.org/>`__ or other tools are available for
more sophisticated 3D rendering.

.. code:: ipython2

    import matplotlib.pyplot as plt

Set the path to the micro-CT data to reconstruct.

.. code:: ipython2

    fname = '../../tomopy/data/tooth.h5'

Select the sinogram range to reconstruct.

.. code:: ipython2

    start = 0
    end = 2

This data set file format follows the `APS <http://www.aps.anl.gov>`__
beamline `2-BM and 32-ID <https://www1.aps.anl.gov/Imaging>`__
definition. Other file format readers are available at
`DXchange <http://dxchange.readthedocs.io/en/latest/source/api/dxchange.exchange.html>`__.

.. code:: ipython2

    proj, flat, dark, theta = dxchange.read_aps_32id(fname, sino=(start, end))

Plot the sinogram:

.. code:: ipython2

    plt.imshow(proj[:, 0, :], cmap='Greys_r')
    plt.show()



.. image:: lprec_files/output_15_0.png


If the angular information is not avaialable from the raw data you need
to set the data collection angles. In this case theta is set as equally
spaced between 0-180 degrees.

.. code:: ipython2

    theta = tomopy.angles(proj.shape[0])

Perform the flat-field correction of raw data:

.. math::  \frac{proj - dark} {flat - dark} 

.. code:: ipython2

    proj = tomopy.normalize(proj, flat, dark)

Select the rotation center manually

.. code:: ipython2

    rot_center = 296

Calculate

.. math::  -log(proj) 

.. code:: ipython2

    proj = tomopy.minus_log(proj)

Reconstruction using FBP method with the log-polar coordinates

.. code:: ipython2

    recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='lpfbp', filter_name='parzen')

Mask each reconstructed slice with a circle.

.. code:: ipython2

    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

.. code:: ipython2

    plt.imshow(recon[0, :,:], cmap='Greys_r')
    plt.show()



.. image:: lprec_files/output_28_0.png


Reconstruction using the gradient descent method with the log-polar
coordinates

.. code:: ipython2

    recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='lpgrad', ncore=1, num_iter=64, reg_par=-1)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :,:], cmap='Greys_r')
    plt.show()



.. image:: lprec_files/output_30_0.png


Reconstruction using the TV method with the log-polar coordinates

.. code:: ipython2

    recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='lptv', ncore=1, num_iter=256, reg_par=2e-4)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :,:], cmap='Greys_r')
    plt.show()



.. image:: lprec_files/output_32_0.png


Reconstruction using the MLEM method with the log-polar coordinates

.. code:: ipython2

    recon = tomopy.recon(proj, theta, center=rot_center, algorithm=tomopy.lprec, lpmethod='lpem', ncore=1, num_iter=64, reg_par=0.05)
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :,:], cmap='Greys_r')
    plt.show()



.. image:: lprec_files/output_34_0.png

