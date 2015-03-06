# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('ggplot')

# --------------------------------------------------------------------


def to_png(dataset, output_filename='/tmp/projection_{:d}.png'):
    """
    Dump projections to png.

    Parameters
    ----------
    dataset : ndarray
        1 or more 3-D tomographic data with dimensions:
        [projections, slices, pixels]

    Returns
    -------
    None

    Examples
    --------
    """
    fig, ax = plt.subplots(ncols=1, nrows=1)
    for i in range(dataset.shape[0]):
        ax.imshow(dataset[i,:,:])
        plt.savefig(fig, output_filename)
        plt.cla()

