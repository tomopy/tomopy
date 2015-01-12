"""Interact with X-ray fluorescence tomography data.

Implements a container for X-ray fluorescence tomography data. Interact with the data to perform preprocessing, reconstruction and postprocessing.

:Author:
    David J. Vine <http://www.djvine.com>

:Organization:
    Argonne National Laboratory

:Version:
    2015.01.10

Requires
--------
Numpy

Examples
--------

"""
import logging
import numpy as np


class XFTomoDataset(object):

    def __init__(self, data, theta=None, log='INFO', color_log=True, stream_handler=True):
        """

        An XFTomo_Dataset instance has a ``data`` attribute with dimensions [channel, projections, slices, pixels]. This is analogous with an XTomo_Dataset object which is use for phase/absoprtion tomography but with an additional dimension to contain multiple fluorescence channels or scalers.

        Attributes
        ----------
        log : str, optional
            Determines the logging level.
            Available arguments: {'DEBUG' 'INFO' 'WARN' 'WARNING' 'ERROR'}.

        color_log : bool, optional
            If ``True`` command line logging is colored.
            You may want to set it ``False`` if you will use
            file logging only.
        """
        # Logging init.
        if color_log:
            # enable colored logging
            from tomopy.tools import colorer

        # Set the log level.
        self.logger = None
        self._log_level = str(log).upper()
        self._init_logging(stream_handler)

        # Get data
        self.data = data
        self.theta = theta

        # Dimensions:
        num_channels = self.data.shape[0]
        num_projs = self.data.shape[1]
        num_slices = self.data.shape[2]
        num_pixels = self.data.shape[3]


        # Assign theta
        if not theta:
            self.theta = np.linspace(0, num_projs, num_projs) * 180 / (
                num_projs + 1)
            self.logger.warning("assumed 180-degree rotation [ok]")

        # Impose data types.
        if not isinstance(self.data, np.float32):
            self.data = np.array(self.data, dtype='float32', copy=False)
        if not isinstance(self.theta, np.float32):
            self.theta = np.array(self.theta, dtype='float32')

        # Update log.
        self.logger.debug('data shape: [{:d} {:d} {:d} {:d}]'.format(num_channels, num_projs, num_slices, num_pixels))

    def __repr__(self):
        return "<{:s}>".format(self.__class__)+" channels: {:d} projections: {:d} slices: {:d} pixels: {:d}".format(*self.data.shape)

    def _init_logging(self, stream_handler):
        """
        Setup and start command line logging.
        """
        # Top-level log setup.
        self.logger = logging.getLogger("tomopy")
        if self._log_level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif self._log_level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif self._log_level == 'WARN':
            self.logger.setLevel(logging.WARN)
        elif self._log_level == 'WARNING':
            self.logger.setLevel(logging.WARNING)
        elif self._log_level == 'ERROR':
            self.logger.setLevel(logging.ERROR)

        # Terminal stream log.
        ch = logging.StreamHandler()
        if self._log_level == 'DEBUG':
            ch.setLevel(logging.DEBUG)
        elif self._log_level == 'INFO':
            ch.setLevel(logging.INFO)
        elif self._log_level == 'WARN':
            ch.setLevel(logging.WARN)
        elif self._log_level == 'WARNING':
            ch.setLevel(logging.WARNING)
        elif self._log_level == 'ERROR':
            ch.setLevel(logging.ERROR)

        # Show date and time.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Update logger.
        if not len(self.logger.handlers):  # For fist time create handlers.
            if stream_handler:
                self.logger.addHandler(ch)
            else:
                self.logger.addHandler(logging.NullHandler())

