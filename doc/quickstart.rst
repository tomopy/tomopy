================
Quickstart Guide
================

This section gives a quick start on how to simulate data and reconstruct 
objects.

.. contents:: Contents
   :local:


Data simulation
===============



Let us first import the `Shepp-Logan phantom <http://en.wikipedia.org/wiki/Shepp–Logan_phantom>`_ ::

    >>> import tomopy
    >>> obj = tomopy.shepp2d()
    >>> print(obj.shape)
    (1, 512, 512)

This creates a single slice of the 3-D phantom::

    >>> import matplotlib.pylab as plt
    >>> plt.imshow(obj[0], cmap='gray')
    >>> plt.show()

.. image:: img/shepp.png
    :height: 300px
    :width: 300px

We can then define the projection angles and pass it with the generated 
object to the ``simulate`` function::

    >>> import numpy as np
    >>> theta = np.linspace(0, np.pi, 360)
    >>> data = tomopy.simulate(obj, theta)
    >>> print(data.shape)
    (90, 1, 725)

Then we can image the sinogram::

    >>> plt.imshow(data[:,0,:], cmap='gray')
    >>> plt.xlabel('Pixels')
    >>> plt.ylabel('Projections')
    >>> plt.show()

.. image:: img/sinogram.png
    :height: 250px
    :width: 500px

Image reconstruction
====================

Tomographic reconstruction creates three-dimensional views of an object 
by combining two-dimensional projection images taken from multiple 
directions, for example in how a CAT (computer-aided tomography) 
scanner allows 3D views of the heart or brain.

For reconstruction we require the sinogram and the projection angles::

    >>> recon = tomopy.art(data, theta, num_iter=1, num_gridx=512, num_gridy=512)
    >>> plt.imshow(recon[0], cmap='gray')
    >>> plt.show()

.. image:: img/art.png
    :height: 300px
    :width: 300px

The reconstruction grid size can be adjusted using the ``num_gridx`` and 
``num_gridy`` arguments. For more information about arguments please read
the function docs.

