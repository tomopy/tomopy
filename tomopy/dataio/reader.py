# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
import logging
logger = logging.getLogger("tomopy")


class Dataset():
    def __init__(TomoObj, data=None, data_white=None, theta=None, log='DEBUG', clog=True):
        """
        Constructor for initial Data-Exchange data structure.
        
        Attributes
        ----------
        TomoObj : tomopy data object
            This is the core object that all low-level 
            attributes and methods are bound to.
            
        data : ndarray
            3-D tomography data. Dimensions should be
            [projections, slices, pixels].
            
        data_white : ndarray
            3-D white-field data. Multiple projections
            are stacked together to obtain 3-D matrix. 
            2nd and 3rd dimensions should be the same as
            data [shots, slices, pixels].
            
        theta : ndarray
            Data acquisition angles corresponding
            to each projection.
        
        """
        # Set the numpy Data-Exchange structure.
        TomoObj.data = np.array(data) # do not squeeze
        TomoObj.data_white = np.array(data_white) # do not squeeze
        TomoObj.theta = np.array(np.squeeze(theta))
        TomoObj._log_level = str(log).upper()
        
        # Init all flags here. False unless checked.
        TomoObj.FLAG_DATA = False
        TomoObj.FLAG_WHITE = False
        TomoObj.FLAG_THETA = False
        TomoObj.FLAG_FILE_CHECK = False
        TomoObj.FLAG_RECON = False
        
        # Logging init.
        if clog: # enable colored logging
            from tomopy.tools import colorer
        TomoObj._init_log()
        
        # Ignore inconsistent data.
        if not TomoObj.FLAG_DATA:
            TomoObj.data = None
        if not TomoObj.FLAG_WHITE:
            TomoObj.data_white = None
        if not TomoObj.FLAG_THETA:
            TomoObj.theta = None
        logger.debug("TomoObj initialization [ok]")
           
    def read(TomoObj, file_name,
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
             log='INFO'):
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
            Values of the start, end and step of the
            slicing for the whole white field shots.

        dtype : str, optional
            Desired output data type.
            
        Notes
        -----
        Unless specified in the file, a uniformly sampled
        180 degree rotation is assumed for ``theta``.
        
        If ``data_white`` is not available, then the ``data``
        is normalized with the average value ''data''.
        """
        # Start working on checks and stuff.
        TomoObj.file_name = os.path.abspath(file_name)
        TomoObj.projections_start = projections_start
        TomoObj.projections_end = projections_end
        TomoObj.projections_step = projections_step
        TomoObj.slices_start = slices_start
        TomoObj.slices_end = slices_end
        TomoObj.slices_step = slices_step
        TomoObj.pixels_start = pixels_start
        TomoObj.pixels_end = pixels_end
        TomoObj.pixels_step = pixels_step
        TomoObj._log_level = str(log).upper()
        
        # Prepare logging file.
        TomoObj._set_log_file()

        # Make checks.
        TomoObj._check_input_file()

        if TomoObj.FLAG_DATA:
            # All looks fine. Start reading data.
            f = h5py.File(TomoObj.file_name, "r")
            hdfdata = f["/exchange/data"]

            # Prepare slicing based on data shape.
            num_x, num_y, num_z = hdfdata.shape
            if projections_start is None:
                TomoObj.projections_start = 0
            if projections_end is None:
                TomoObj.projections_end = num_x
            if projections_step is None:
                TomoObj.projections_step = 1
            if slices_start is None:
                TomoObj.slices_start = 0
            if slices_end is None:
                TomoObj.slices_end = num_y
            if slices_step is None:
                TomoObj.slices_step = 1
            if pixels_start is None:
                TomoObj.pixels_start = 0
            if pixels_end is None:
                TomoObj.pixels_end = num_z
            if pixels_step is None:
                TomoObj.pixels_step = 1
        
            TomoObj.data = hdfdata[TomoObj.projections_start:
                                    TomoObj.projections_end:
                                        TomoObj.projections_step,
                                TomoObj.slices_start:
                                    TomoObj.slices_end:
                                        TomoObj.slices_step,
                                TomoObj.pixels_start:
                                    TomoObj.pixels_end:
                                        TomoObj.pixels_step]
            logger.info("read data from file [ok]")

            # Now read white fields.
            if TomoObj.FLAG_WHITE:
                hdfdata = f["/exchange/data_white"]

                # Prepare slicing based on data shape.
                if white_start is None:
                    TomoObj.white_start = 0
                if white_end is None:
                    TomoObj.white_end = hdfdata.shape[0]

                # Slice it now.
                TomoObj.data_white = hdfdata[TomoObj.white_start:
                                              TomoObj.white_end,
                                          TomoObj.slices_start:
                                              TomoObj.slices_end:
                                                  TomoObj.slices_step,
                                          TomoObj.pixels_start:
                                              TomoObj.pixels_end:
                                                  TomoObj.pixels_step]
                logger.info("read data_white from file [ok]")
            else:
                TomoObj.data_white = np.zeros((1, TomoObj.data.shape[1], TomoObj.data.shape[2]))
                TomoObj.data_white += np.mean(TomoObj.data[:])
                logger.warning("auto-normalization [ok]")

            # Read projection angles.
            if TomoObj.FLAG_THETA:
                hdfdata = f["/exchange/theta"]
                TomoObj.theta = hdfdata[TomoObj.projections_start:
                                        TomoObj.projections_end:
                                            TomoObj.projections_step,
                                    TomoObj.slices_start:
                                        TomoObj.slices_end:
                                            TomoObj.slices_step,
                                    TomoObj.pixels_start:
                                        TomoObj.pixels_end:
                                            TomoObj.pixels_step]
                logger.info("reading theta from file [ok]")
            else:
                TomoObj.theta = np.linspace(0, TomoObj.data.shape[0], TomoObj.data.shape[0]) \
                                * 180 / TomoObj.data.shape[0]
                logger.warning("assign 180-degree rotation [ok]")

            # All done. Close file.
            f.close()
        
        # We work with float32.
        if TomoObj.FLAG_DATA:
            TomoObj.data = TomoObj.data.astype('float32')
        if TomoObj.FLAG_WHITE:
            TomoObj.data_white = TomoObj.data_white.astype('float32')
        if TomoObj.FLAG_THETA:
            TomoObj.theta = TomoObj.theta.astype('float32')
        logger.debug("data conversion to float32 [ok]")

    def _init_log(TomoObj):
        # Top-level log setup.
        logger.setLevel(logging.DEBUG)
        
        # Terminal stram log.
        ch = logging.StreamHandler()
        if TomoObj._log_level == 'DEBUG':
            ch.setLevel(logging.DEBUG)
        elif TomoObj._log_level == 'INFO':
            ch.setLevel(logging.INFO)
        elif TomoObj._log_level == 'WARN':
            ch.setLevel(logging.WARN)
        elif TomoObj._log_level == 'WARNING':
            ch.setLevel(logging.WARNING)
        elif TomoObj._log_level == 'ERROR':
            ch.setLevel(logging.ERROR)
        
        # Show date and time.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Update logger.
        logger.addHandler(ch)
        
    def _set_log_file(TomoObj):
        log_name = os.path.splitext(TomoObj.file_name)[0] + ".log"
        
        # File log.
        fh = logging.FileHandler(log_name)
        if TomoObj._log_level == 'DEBUG':
            fh.setLevel(logging.DEBUG)
        elif TomoObj._log_level == 'INFO':
            fh.setLevel(logging.INFO)
        elif TomoObj._log_level == 'WARN':
            fh.setLevel(logging.WARN)
        elif TomoObj._log_level == 'WARNING':
            fh.setLevel(logging.WARNING)
        elif TomoObj._log_level == 'ERROR':
            fh.setLevel(logging.ERROR)
            
        # Show date and time.
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Update logger.
        logger.addHandler(fh)

        logger.info("logger file [ok]")

    def _check_input_file(TomoObj):
        """
        Check if HDF5 file is o.k.
        
        The function only modifies flags. 
        
        Check list (flags):
            - File existence (error)
            - File read/write permissions (error)
            - HDF5 exchange group existence (error)
            - HDF5 node existence (error)
            
                - exchange/data (error)
                - exchange/data_white (warning)
                - exchange/theta (warning)
                
            - data dimensions (error)
            - data_white dimensions (warning)
            - theta dimensions (warning)
            - consistency of data and data_white dimensions (warning)
            - consistency of data and theta dimensions (warning)
        """
        # check if file exists.
        if os.path.isfile(TomoObj.file_name):
            TomoObj.FLAG_DATA = True
            logger.info("file check: %s [passed]", TomoObj.file_name)
        else:
            TomoObj.FLAG_DATA = False
            logger.error("file check: %s [failed]", TomoObj.file_name)

        # check read permissions.
        read_access = os.access(TomoObj.file_name, os.R_OK)
        write_access = os.access(TomoObj.file_name, os.W_OK)
        if read_access and write_access:
            TomoObj.FLAG_DATA = True
            logger.debug("file permissions [ok]")
        else:
            TomoObj.FLAG_DATA = False
            logger.error("file permissions [failed]")

        # check if file is hdf5.
        extension = os.path.splitext(TomoObj.file_name)[1]
        if extension == ".hdf" or extension == ".h5":
            TomoObj.FLAG_DATA = True
            logger.debug("file extension: %s [ok]", extension)
        else:
            TomoObj.FLAG_DATA = False
            logger.error("file extension: %s [failed]", extension)

        # check exchange group.
        if TomoObj.FLAG_DATA:
            f = h5py.File(TomoObj.file_name, 'r')
            if "exchange" in f:
                TomoObj.FLAG_DATA = True
                logger.debug("/exchange group [ok]")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("/exchange group [failed]")
            
            # Check exchange nodes.
            if "exchange/data" in f:
                TomoObj.FLAG_DATA = True
                logger.debug("/exchange/data [ok]")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("/exchange/data [failed]")
            if "exchange/data_white" in f:
                TomoObj.FLAG_WHITE = True
                logger.debug("/exchange/data_white [ok]")
            else:
                TomoObj.FLAG_WHITE = False
                logger.warning("/exchange/data_white node [failed]")
            if "exchange/theta" in f:
                TomoObj.FLAG_THETA = True
                logger.debug("/exchange/theta [ok]")
            else:
                TomoObj.FLAG_THETA = False
                logger.warning("/exchange/theta [failed]")
        
            # Check data dimensions.
            if len(f["/exchange/data"].shape) == 3:
                TomoObj.FLAG_DATA = True
                logger.debug("data dimensions [ok]")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("data dimensions [failed]")
            if TomoObj.FLAG_WHITE:
                if len(f["/exchange/data_white"].shape) == 3:
                    TomoObj.FLAG_WHITE = True
                    logger.debug("data_white dimensions [ok]")
                else:
                    TomoObj.FLAG_WHITE = False
                    logger.warning("data_white dimensions [failed]")
            if TomoObj.FLAG_THETA:
                if len(f["/exchange/theta"].shape) == 1 or len(f["/exchange/theta"].shape) == 0:
                    TomoObj.FLAG_THETA = True
                    logger.debug("theta dimensions [ok]")
                else:
                    TomoObj.FLAG_THETA = False
                    logger.warning("theta dimensions [failed]")
            
            # Check data consistencies.
            try:
                if TomoObj.FLAG_WHITE:
                    if f["/exchange/data_white"].shape[1:2] == f["/exchange/data"].shape[1:2]:
                        TomoObj.FLAG_WHITE = True
                        logger.debug("data_white compatibility [ok]")
                    else:
                        TomoObj.FLAG_WHITE = False
                        logger.warning("data_white compatibility [failed]")

                if TomoObj.FLAG_THETA:
                    if f["/exchange/theta"].size == f["/exchange/data"].shape[0]:
                        TomoObj.FLAG_THETA = True
                        logger.debug("theta compatibility [ok]")
                    else:
                        TomoObj.FLAG_THETA = False
                        logger.warning("theta compatibility [failed]")
            except IndexError: # if TomoObj.data is None
                pass

        # Good to go.
        TomoObj.FLAG_FILE_CHECK = True
        logger.debug("file check [ok]")