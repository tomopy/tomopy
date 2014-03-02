# -*- coding: utf-8 -*-
"""
Module for creating dataset.
"""
import numpy as np
import h5py
import os
import time
import logging
logger = logging.getLogger("tomopy")


class Session():
    def __init__(tomo, log='INFO', clog=True):
        """
        Constructor for the data analysis.
        
        Attributes
        ----------
        tomo : tomopy data object
            This is the core object that all low-level 
            attributes and methods are bound to.
            
        log : str, optional
            Determines the logging level.
            Available arguments: {'DEBUG' 'INFO' 'WARN' 'WARNING' 'ERROR'}.
            
        clog : bool, optional
            If ``True`` command line logging is colored. 
            You may want to set it ``False`` if you will use 
            file logging only.
        """
        # Init all flags here. False unless checked.
        tomo.FLAG_DATA = False
        tomo.FLAG_WHITE = False
        tomo.FLAG_DARK = False
        tomo.FLAG_THETA = False
        tomo.FLAG_DATA_CHECK = False
        tomo.FLAG_DATA_RECON = False
        
        # Set the log level.
        tomo._log_level = str(log).upper()
        
        # Provenance initialization.
        tomo._init_provenance()
    
        # Logging init.
        if clog: # enable colored logging
            from tomopy.tools import colorer
        tomo._init_log()
        logger.debug("TomoPy session initialization [ok]")
        
    def dataset(tomo, data=None, data_white=None, 
                data_dark=None, theta=None):
        """
        Convert external dataset into TomoPy data object.
        
        Parameters
        ----------
        
        data : ndarray
            3-D tomography data. Dimensions should be
            [projections, slices, pixels].

        data_white : ndarray
            3-D white-field data. Multiple projections
            are stacked together to obtain 3-D matrix.
            2nd and 3rd dimensions should be the same as
            data [shots, slices, pixels].
            
        data_dark : ndarray
            3-D dark-field data. Multiple projections
            are stacked together to obtain 3-D matrix.
            2nd and 3rd dimensions should be the same as
            data [shots, slices, pixels].
            
        theta : ndarray
            Data acquisition angles corresponding
            to each projection.
        """
        # Control inputs.
        if data is None:
            logger.error("Dataset import [bypassed]")
            return
        
        # Set the numpy Data-Exchange structure.
        tomo.data = np.array(data, dtype='float32', copy=False) # do not squeeze
        tomo.data_white = np.array(data_white, dtype='float32', copy=False) # do not squeeze
        tomo.data_dark = np.array(data_dark, dtype='float32', copy=False) # do not squeeze
        tomo.theta = np.array(np.squeeze(theta), dtype='float32', copy=False)
            
        # Assign data_white
        if data_white is None:
            tomo.data_white = np.zeros((1, tomo.data.shape[1], tomo.data.shape[2]))
            tomo.data_white += np.mean(tomo.data[:])
            logger.warning("auto-normalization [ok]")
        tomo.FLAG_WHITE = True
            
        # Assign data_dark
        if data_dark is None:
            tomo.data_dark = np.zeros((1, tomo.data.shape[1], tomo.data.shape[2]))
            logger.warning("dark-field assumed as zeros [ok]")
        tomo.FLAG_DARK = True
                
        # Assign theta
        if theta is None:
            tomo.theta = np.linspace(0, tomo.data.shape[0], tomo.data.shape[0]) \
                * 180 / (tomo.data.shape[0] + 1)
            logger.warning("assign 180-degree rotation [ok]")
        tomo.FLAG_THETA = True

        # Check if data is as expected.
        tomo._check_input_data()


    def read(tomo, file_name,
             projections_start=None,
             projections_end=None,
             projections_step=None,
             slices_start=None,
             slices_end=None,
             slices_step=None,
             pixels_start=None,
             pixels_end=None,
             pixels_step=None,
             white_start=None,
             white_end=None,
             dark_start=None,
             dark_end=None):
        """
        Read Data Exchange HDF5 file.
        
        Parameters
        ----------
        file_name : str
            Input file.

        projections_start, projections_end, projections_step : scalar, optional
            Values of the start, end and step of the projections to
            be used for slicing for the whole data.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the slices to
            be used for slicing for the whole data.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the pixels to
            be used for slicing for the whole data.

        white_start, white_end : scalar, optional
            Values of the start and end of the
            slicing for the whole white field shots.

        dark_start, dark_end : scalar, optional
            Values of the start and end of the
            slicing for the whole dark field shots.
            
        Notes
        -----
        Unless specified in the file, a uniformly sampled
        180 degree rotation is assumed for ``theta``.
        
        If ``data_white`` is missing, then the ``data``
        is by default normalized with the average value of ''data''.
        
        If ``data_dark`` is missing, then ``data_dark``
        is by default set with zeros.
        """
        # Start working on checks and stuff.
        tomo.file_name = os.path.abspath(file_name)
        tomo.projections_start = projections_start
        tomo.projections_end = projections_end
        tomo.projections_step = projections_step
        tomo.slices_start = slices_start
        tomo.slices_end = slices_end
        tomo.slices_step = slices_step
        tomo.pixels_start = pixels_start
        tomo.pixels_end = pixels_end
        tomo.pixels_step = pixels_step
        tomo.white_start = white_start
        tomo.white_end = white_end
        tomo.dark_start = dark_start
        tomo.dark_end = dark_end
        
        # Prepare logging file specific to input filename and location.
        tomo._set_log_file()

        # Make some checks on input data. This method
        # sets the flags if all fields are as expected.
        tomo._check_input_file()
        tomo.provenance['file_name'] = tomo.file_name

        if tomo.FLAG_DATA:
            # All looks fine. Start reading data.
            f = h5py.File(tomo.file_name, "r")
            hdfdata = f["/exchange/data"]

            # Prepare slicing based on data shape.
            num_x, num_y, num_z = hdfdata.shape
            if projections_start is None:
                tomo.projections_start = 0
            if projections_end is None:
                tomo.projections_end = num_x
            if projections_step is None:
                tomo.projections_step = 1
            if slices_start is None:
                tomo.slices_start = 0
            if slices_end is None:
                tomo.slices_end = num_y
            if slices_step is None:
                tomo.slices_step = 1
            if pixels_start is None:
                tomo.pixels_start = 0
            if pixels_end is None:
                tomo.pixels_end = num_z
            if pixels_step is None:
                tomo.pixels_step = 1
        
            tomo.data = hdfdata[tomo.projections_start:
				      tomo.projections_end:
					  tomo.projections_step,
				  tomo.slices_start:
				      tomo.slices_end:
					  tomo.slices_step,
				  tomo.pixels_start:
				      tomo.pixels_end:
					  tomo.pixels_step]
            logger.info("read data from file [ok]")

            # Now read white fields.
            if tomo.FLAG_WHITE:
                hdfdata = f["/exchange/data_white"]

                # Prepare slicing based on data shape.
                if white_start is None:
                    tomo.white_start = 0
                if white_end is None:
                    tomo.white_end = hdfdata.shape[0]

                # Slice it now.
                tomo.data_white = hdfdata[tomo.white_start:
					         tomo.white_end,
					     tomo.slices_start:
						 tomo.slices_end:
						     tomo.slices_step,
					     tomo.pixels_start:
						 tomo.pixels_end:
						     tomo.pixels_step]
                logger.info("read data_white from file [ok]")
            else:
                tomo.data_white = np.zeros((1, tomo.data.shape[1], tomo.data.shape[2]))
                tomo.data_white += np.mean(tomo.data[:])
                tomo.FLAG_WHITE = True
                logger.warning("auto-normalization [ok]")
            
            # Now read dark fields.
            if tomo.FLAG_DARK:
                hdfdata = f["/exchange/data_dark"]

                # Prepare slicing based on data shape.
                if dark_start is None:
                    tomo.dark_start = 0
                if dark_end is None:
                    tomo.dark_end = hdfdata.shape[0]

                # Slice it now.
                tomo.data_dark = hdfdata[tomo.dark_start:
					         tomo.dark_end,
					     tomo.slices_start:
						 tomo.slices_end:
						     tomo.slices_step,
					     tomo.pixels_start:
						 tomo.pixels_end:
						     tomo.pixels_step]
                logger.info("read data_dark from file [ok]")
            else:
                tomo.data_dark = np.zeros((1, tomo.data.shape[1], tomo.data.shape[2]))
                tomo.FLAG_DARK = True
                logger.warning("dark-field assumed as zeros [ok]")

            # Read projection angles.
            if tomo.FLAG_THETA:
                hdfdata = f["/exchange/theta"]
                tomo.theta = hdfdata[tomo.projections_start:
					    tomo.projections_end:
						tomo.projections_step]
                logger.info("reading theta from file [ok]")
            else:
                tomo.theta = np.linspace(0, tomo.data.shape[0], tomo.data.shape[0]) \
                                * 180 / (tomo.data.shape[0] + 1)
                tomo.FLAG_THETA = True
                logger.warning("assign 180-degree rotation [ok]")

            # All done. Close file.
            f.close()
            
            # We want float32 inputs.
            if not isinstance(tomo.data, np.float32):
                tomo.data = tomo.data.astype(dtype=np.float32, copy=False)
            if not isinstance(tomo.data_white, np.float32):
                tomo.data_white = tomo.data_white.astype(dtype=np.float32, copy=False)
            if not isinstance(tomo.data_dark, np.float32):
                tomo.data_dark = tomo.data_dark.astype(dtype=np.float32, copy=False)
            if not isinstance(tomo.theta, np.float32):
                tomo.theta = tomo.theta.astype(dtype=np.float32, copy=False)
            
    def _init_provenance(tomo):
        # Start adding info.
        tomo.provenance = {}
        tomo.provenance['date'] = time.strftime('%Y-%m-%d')
        tomo.provenance['time'] = time.strftime('%H:%M:%S')

    def _init_log(tomo):
        """
        Setup and start command line logging.
        """
        # Top-level log setup.
        logger.setLevel(logging.DEBUG)
        
        # Terminal stram log.
        ch = logging.StreamHandler()
        if tomo._log_level == 'DEBUG':
            ch.setLevel(logging.DEBUG)
        elif tomo._log_level == 'INFO':
            ch.setLevel(logging.INFO) 
        elif tomo._log_level == 'WARN':
            ch.setLevel(logging.WARN)
        elif tomo._log_level == 'WARNING':
            ch.setLevel(logging.WARNING)
        elif tomo._log_level == 'ERROR':
            ch.setLevel(logging.ERROR)
        
        # Show date and time.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Update logger.
        logger.addHandler(ch)
        
    def _set_log_file(tomo):
        """
        Setup and start file logging. Log file is placed in
        the same directory of the data with the same data name.
        """
        log_name = os.path.splitext(tomo.file_name)[0] + ".log"
        
        # File log.
        try:
            fh = logging.FileHandler(log_name)
        except IOError:
            return
        if tomo._log_level == 'DEBUG':
            fh.setLevel(logging.DEBUG)
        elif tomo._log_level == 'INFO':
            fh.setLevel(logging.INFO)
        elif tomo._log_level == 'WARN':
            fh.setLevel(logging.WARN)
        elif tomo._log_level == 'WARNING':
            fh.setLevel(logging.WARNING)
        elif tomo._log_level == 'ERROR':
            fh.setLevel(logging.ERROR)
            
        # Show date and time.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Update logger.
        logger.addHandler(fh)
        logger.info("logger file [ok]")
        
    def _check_input_data(tomo):
        """
        Check if data is o.k.
        
        This method only modifies flags. Below is 
        the list of checks made and the corresponding 
        log messages.
        
        Check list (flags):
            - data dimensions (error)
            - data_white dimensions (warning)
            - data_dark dimensions (warning)
            - theta dimensions (warning)
            - consistency of data and data_white dimensions (warning)
            - consistency of data and data_dark dimensions (warning)
            - consistency of data and theta dimensions (warning)
        """
        # Check data dimensions.
        if len(tomo.data.shape) == 3:
            tomo.FLAG_DATA = True
            logger.debug("data dimensions [ok]")
        else:
            tomo.FLAG_DATA = False
            logger.error("data dimensions [failed]")
        if tomo.FLAG_WHITE:
            if len(tomo.data_white.shape) == 3:
                tomo.FLAG_WHITE = True
                logger.debug("data_white dimensions [ok]")
            else:
                tomo.FLAG_WHITE = False
                logger.warning("data_white dimensions [failed]")
        if tomo.FLAG_DARK:
            if len(tomo.data_dark.shape) == 3:
                tomo.FLAG_DARK = True
                logger.debug("data_dark dimensions [ok]")
            else:
                tomo.FLAG_DARK = False
                logger.warning("data_dark dimensions [failed]")
        if tomo.FLAG_THETA:
            if len(tomo.theta.shape) == 1 or len(tomo.theta.shape) == 0:
                tomo.FLAG_THETA = True
                logger.debug("theta dimensions [ok]")
            else:
                tomo.FLAG_THETA = False
                logger.warning("theta dimensions [failed]")
        
        # Check data consistencies.
        try:
            if tomo.FLAG_WHITE:
                if tomo.data_white.shape[1:2] == tomo.data.shape[1:2]:
                    tomo.FLAG_WHITE = True
                    logger.debug("data_white compatibility [ok]")
                else:
                    tomo.FLAG_WHITE = False
                    logger.warning("data_white compatibility [failed]")
            if tomo.FLAG_DARK:
                if tomo.data_dark.shape[1:2] == tomo.data.shape[1:2]:
                    tomo.FLAG_DARK = True
                    logger.debug("data_dark compatibility [ok]")
                else:
                    tomo.FLAG_DARK = False
                    logger.warning("data_dark compatibility [failed]")
            if tomo.FLAG_THETA:
                if tomo.theta.size == tomo.data.shape[0]:
                    tomo.FLAG_THETA = True
                    logger.debug("theta compatibility [ok]")
                else:
                    tomo.FLAG_THETA = False
                    logger.warning("theta compatibility [failed]")
        except IndexError: # if tomo.data is None
            pass
                
        # Good to go.
        tomo.FLAG_DATA_CHECK = True
        logger.debug("file check [ok]")
            

    def _check_input_file(tomo):
        """
        Check if HDF5 file is o.k.
        
        This method only modifies flags. Below is 
        the list of checks made and the corresponding 
        log messages.
        
        Check list (flags):
            - File existence (error)
            - File read/write permissions (error)
            - HDF5 exchange group existence (error)
            - HDF5 node existence (error)
            
                - exchange/data (error)
                - exchange/data_white (warning)
                - exchange/data_dark (warning)
                - exchange/theta (warning)
                
            - data dimensions (error)
            - data_white dimensions (warning)
            - data_dark dimensions (warning)
            - theta dimensions (warning)
            - consistency of data and data_white dimensions (warning)
            - consistency of data and data_dark dimensions (warning)
            - consistency of data and theta dimensions (warning)
        """
        # check if file exists.
        if os.path.isfile(tomo.file_name):
            tomo.FLAG_DATA = True
            logger.info("file check: %s [ok]", tomo.file_name)
        else:
            tomo.FLAG_DATA = False
            logger.error("file check: %s [failed]", tomo.file_name)

        # check read permissions.
        read_access = os.access(tomo.file_name, os.R_OK)
        write_access = os.access(tomo.file_name, os.W_OK)
        if read_access and write_access:
            tomo.FLAG_DATA = True
            logger.debug("file permissions [ok]")
        else:
            tomo.FLAG_DATA = False
            logger.error("file permissions [failed]")

        # check if file is hdf5.
        extension = os.path.splitext(tomo.file_name)[1]
        if extension == ".hdf" or extension == ".h5":
            if os.path.isfile(tomo.file_name):
                tomo.FLAG_DATA = True
            logger.debug("file extension: %s [ok]", extension)
        else:
            tomo.FLAG_DATA = False
            logger.error("file extension: %s [failed]", extension)

        # check exchange group.
        if tomo.FLAG_DATA:
            f = h5py.File(tomo.file_name, 'r')
            if "exchange" in f:
                tomo.FLAG_DATA = True
                logger.debug("/exchange group [ok]")
            else:
                tomo.FLAG_DATA = False
                logger.error("/exchange group [failed]")
            
            # Check exchange nodes.
            if "exchange/data" in f:
                tomo.FLAG_DATA = True
                logger.debug("/exchange/data [ok]")
            else:
                tomo.FLAG_DATA = False
                logger.error("/exchange/data [failed]")
            if "exchange/data_white" in f:
                tomo.FLAG_WHITE = True
                logger.debug("/exchange/data_white [ok]")
            else:
                tomo.FLAG_WHITE = False
                logger.warning("/exchange/data_white node [failed]")
            if "exchange/data_dark" in f:
                tomo.FLAG_DARK = True
                logger.debug("/exchange/data_dark [ok]")
            else:
                tomo.FLAG_DARK = False
                logger.warning("/exchange/data_dark node [failed]")
            if "exchange/theta" in f:
                tomo.FLAG_THETA = True
                logger.debug("/exchange/theta [ok]")
            else:
                tomo.FLAG_THETA = False
                logger.warning("/exchange/theta [failed]")
        
            # Check data dimensions.
            if len(f["/exchange/data"].shape) == 3:
                tomo.FLAG_DATA = True
                logger.debug("data dimensions [ok]")
            else:
                tomo.FLAG_DATA = False
                logger.error("data dimensions [failed]")
            if tomo.FLAG_WHITE:
                if len(f["/exchange/data_white"].shape) == 3:
                    tomo.FLAG_WHITE = True
                    logger.debug("data_white dimensions [ok]")
                else:
                    tomo.FLAG_WHITE = False
                    logger.warning("data_white dimensions [failed]")
            if tomo.FLAG_DARK:
                if len(f["/exchange/data_dark"].shape) == 3:
                    tomo.FLAG_DARK = True
                    logger.debug("data_dark dimensions [ok]")
                else:
                    tomo.FLAG_DARK = False
                    logger.warning("data_dark dimensions [failed]")
            if tomo.FLAG_THETA:
                if len(f["/exchange/theta"].shape) == 1 or len(f["/exchange/theta"].shape) == 0:
                    tomo.FLAG_THETA = True
                    logger.debug("theta dimensions [ok]")
                else:
                    tomo.FLAG_THETA = False
                    logger.warning("theta dimensions [failed]")
            
            # Check data consistencies.
            try:
                if tomo.FLAG_WHITE:
                    if f["/exchange/data_white"].shape[1:2] == f["/exchange/data"].shape[1:2]:
                        tomo.FLAG_WHITE = True
                        logger.debug("data_white compatibility [ok]")
                    else:
                        tomo.FLAG_WHITE = False
                        logger.warning("data_white compatibility [failed]")
                if tomo.FLAG_DARK:
                    if f["/exchange/data_dark"].shape[1:2] == f["/exchange/data"].shape[1:2]:
                        tomo.FLAG_DARK = True
                        logger.debug("data_dark compatibility [ok]")
                    else:
                        tomo.FLAG_DARK = False
                        logger.warning("data_dark compatibility [failed]")
                if tomo.FLAG_THETA:
                    if f["/exchange/theta"].size == f["/exchange/data"].shape[0]:
                        tomo.FLAG_THETA = True
                        logger.debug("theta compatibility [ok]")
                    else:
                        tomo.FLAG_THETA = False
                        logger.warning("theta compatibility [failed]")
            except IndexError: # if tomo.data is None
                pass
                    
            # Good to go.
            tomo.FLAG_DATA_CHECK = True
            logger.debug("file check [ok]")
        else:
            tomo.FLAG_DATA_CHECK = False
            logger.error("file check [failed]")
            
            
            