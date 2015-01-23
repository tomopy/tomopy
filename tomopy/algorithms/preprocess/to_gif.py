# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('ggplot')

# --------------------------------------------------------------------


def to_gif(datasets, title='Projection: {:d}',
        output_filename='/tmp/projections.gif',
        fps=4):
    """
    Convert projections to GIF.

    Parameters
    ----------
    datasets : list(ndarray)
        1 or more 3-D tomographic data with dimensions:
        [projections, slices, pixels]

    title : string
        String which will be passed to each frame of the gif
        with the integer frame number

    fps : integer
        Frames per second in the output gif


    Returns
    -------
    None

    Examples
    --------
    """
    if not isinstance(datasets, list):
        datasets = list(datasets)

    n_datasets = len(datasets)

    def animate(nframe):
        plt.cla()
        for i, ax in enumerate(axes):
            ax.imshow(datasets[i][nframe,:,:])
            if i==0:
                plt.title(title.format(nframe))
        plt.savefig(output_filename.split('.')[0]+'_{:d}.png'.format(nframe))


    fig, axes = plt.subplots(ncols = n_datasets, nrows=1)
    anim = animation.FuncAnimation(fig, animate, frames=datasets[0].shape[0])
    anim.save(output_filename, writer='imagemagick', fps=4)

