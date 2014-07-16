# -*- coding: utf-8 -*-
import numpy as np
import logging


class XTomoDataset:
    def __init__(self, log='INFO', color_log=True, stream_handler=True):
        """
        Constructor.
        
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
        if color_log: # enable colored logging
            from tomopy.tools import colorer

        # Set the log level.
        self.logger = None
        self._log_level = str(log).upper()
        self._init_logging(stream_handler)


    def dataset(self, data, data_white=None, 
                data_dark=None, theta=None):
        """
        Import X-ray absorption tomography data object.
        
        Parameters
        ----------
        data : ndarray
            3-D X-ray absorption tomography raw data. 
            Size of the dimensions should be: 
            [projections, slices, pixels].
    
        data_white, data_dark : ndarray,  optional
            3-D white-field/dark_field data. Multiple 
            projections are stacked together to obtain 
            a 3-D matrix. 2nd and 3rd dimensions should 
            be the same as data: [shots, slices, pixels].
            
        theta : ndarray, optional
            Data acquisition angles corresponding
            to each projection.
        """
 
        # Set the numpy Data-Exchange structure.
        self.data = data
        self.data_white = data_white
        self.data_dark = data_dark
        self.theta = np.squeeze(theta)
        
        # Dimensions:
        num_projs = self.data.shape[0]
        num_slices = self.data.shape[1]
        num_pixels = self.data.shape[2]
        
        # Assign data_white
        if data_white is None:
            self.data_white = np.zeros((1, num_slices, num_pixels))
            self.data_white += np.mean(self.data[:])
            self.logger.warning('auto-normalization [ok]')
            
        # Assign data_dark
        if data_dark is None:
            self.data_dark = np.zeros((1, num_slices, num_pixels))
            self.logger.warning('dark-field assumed as zeros [ok]')
                
        # Assign theta
        if theta is None:
            self.theta = np.linspace(0, num_projs, num_projs)*180/(num_projs+1)
            self.logger.warning("assumed 180-degree rotation [ok]")
            
        # Impose data types.
        if not isinstance(self.data, np.float32):
            self.data = np.array(self.data, dtype='float32', copy=False)
        if not isinstance(self.data_white, np.float32):
            self.data_white = np.array(self.data_white, dtype='float32')
        if not isinstance(self.data_dark, np.float32):
            self.data_dark = np.array(self.data_dark, dtype='float32')
        if not isinstance(self.theta, np.float32):
            self.theta = np.array(self.theta, dtype='float32')
            
        # Update log.
        self.logger.debug('data shape: [%i, %i, %i]', 
                           num_projs, num_slices, num_pixels)


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
        if not len(self.logger.handlers): # For fist time create handlers.
            if stream_handler:
                self.logger.addHandler(ch)
            else:
                self.logger.addHandler(logging.NullHandler())