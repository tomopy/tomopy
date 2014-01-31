# -*- coding: utf-8 -*-
import h5py
import os
import numpy as np
import logging
logger = logging.getLogger(__name__)


class Dataset():
    def __init__(TomoObj, data=None, data_white=None, theta=None, log_level='WARNING'):
        """Constructor for initial Data-Exchange data structure.
        """
        # Set the numpy Data-Exchange structure.
        TomoObj.data = np.array(data) # do not squeeze
        TomoObj.data_white = np.array(data_white) # do not squeeze
        TomoObj.theta = np.array(np.squeeze(theta))
        TomoObj._log_level = str(log_level).upper()
        
        # Init all flags here. False unless checked.
        TomoObj.FLAG_DATA = False
        TomoObj.FLAG_WHITE = False
        TomoObj.FLAG_THETA = False
        TomoObj.FLAG_FILE_CHECK = False
        
        # Logging init.
        TomoObj._init_log()
        
        # Ignore inconsistent data.
        if not TomoObj.FLAG_DATA:
            TomoObj.data = None
        if not TomoObj.FLAG_WHITE:
            TomoObj.data_white = None
        if not TomoObj.FLAG_THETA:
            TomoObj.theta = None
           
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
             log_level='INFO'):
        """Read Data Exchange HDF5 file.

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
        TomoObj._log_level = log_level
        
        # Prepare logging file.
        TomoObj._set_log_file()

        # Make checks.
        TomoObj._check_input_file()

        if TomoObj.FLAG_DATA:
            # All looks fine. Start reading data.
            logger.info("reading data from file")
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

            # Now read white fields.
            if TomoObj.FLAG_WHITE:
                logger.info("reading data_white from file")
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

            # Read projection angles.
            if TomoObj.FLAG_THETA:
                logger.info("reading theta from file")
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
            # All done. Close file.
            f.close()
        
        # We work with float32.
        if TomoObj.FLAG_DATA:
            TomoObj.data = TomoObj.data.astype('float32')
        if TomoObj.FLAG_WHITE:
            TomoObj.data_white = TomoObj.data_white.astype('float32')
        if TomoObj.FLAG_THETA:
            TomoObj.theta = TomoObj.theta.astype('float32')
            
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

    def _check_input_file(TomoObj):
        """Check if HDF5 file is o.k.
        
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
            logger.info("file found: %s", TomoObj.file_name)
        else:
            TomoObj.FLAG_DATA = False
            logger.error("no such file")

        # check read permissions.
        read_access = os.access(TomoObj.file_name, os.R_OK)
        write_access = os.access(TomoObj.file_name, os.W_OK)
        if read_access and write_access:
            TomoObj.FLAG_DATA = True
            logger.debug("file permissions are ok")
        else:
            TomoObj.FLAG_DATA = False
            logger.error("permission denied")

        # check if file is hdf5.
        extension = os.path.splitext(TomoObj.file_name)[1]
        if extension == ".hdf" or extension == ".h5":
            TomoObj.FLAG_DATA = True
            logger.debug("supported file: %s", extension)
        else:
            TomoObj.FLAG_DATA = False
            logger.error("unsupported file type")

        # check exchange group.
        if TomoObj.FLAG_DATA:
            f = h5py.File(TomoObj.file_name, 'r')
            if "exchange" in f:
                TomoObj.FLAG_DATA = True
                logger.debug("/exchange group found")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("no exchange group")
            
            # Check exchange nodes.
            if "exchange/data" in f:
                TomoObj.FLAG_DATA = True
                logger.debug("/exchange/data found")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("no /exchange/data node in exchange group")
            if "exchange/data_white" in f:
                TomoObj.FLAG_WHITE = True
                logger.debug("/exchange/data_white found")
            else:
                TomoObj.FLAG_WHITE = False
                logger.warning("no /exchange/data_white node in exchange group")
            if "exchange/theta" in f:
                TomoObj.FLAG_THETA = True
                logger.debug("/exchange/theta is found")
            else:
                TomoObj.FLAG_THETA = False
                logger.warning("no /exchange/theta node in exchange group")
        
            # Check data dimensions.
            if len(f["/exchange/data"].shape) == 3:
                TomoObj.FLAG_DATA = True
                logger.debug("data dimension is correct")
            else:
                TomoObj.FLAG_DATA = False
                logger.error("data dimension is incorrect")
            if TomoObj.FLAG_WHITE:
                if len(f["/exchange/data_white"].shape) == 3:
                    TomoObj.FLAG_WHITE = True
                    logger.debug("data_white dimension is correct")
                else:
                    TomoObj.FLAG_WHITE = False
                    logger.warning("data_white dimension is incorrect")
            if TomoObj.FLAG_THETA:
                if len(f["/exchange/theta"].shape) == 1 or len(f["/exchange/theta"].shape) == 0:
                    TomoObj.FLAG_THETA = True
                    logger.debug("theta dimension is correct")
                else:
                    TomoObj.FLAG_THETA = False
                    logger.warning("theta dimension is incorrect")
            
            # Check data consistencies.
            try:
                if TomoObj.FLAG_WHITE:
                    if f["/exchange/data_white"].shape[1:2] == f["/exchange/data"].shape[1:2]:
                        TomoObj.FLAG_WHITE = True
                        logger.debug("data_white dimension is compatible with data")
                    else:
                        TomoObj.FLAG_WHITE = False
                        logger.warning("data_white dimension is incompatible with data")

                if TomoObj.FLAG_THETA:
                    if f["/exchange/theta"].size == f["/exchange/data"].shape[0]:
                        TomoObj.FLAG_THETA = True
                        logger.debug("theta dimension is compatible with data")
                    else:
                        TomoObj.FLAG_THETA = False
                        logger.warning("theta dimension is incompatible with data")
            except IndexError: # if TomoObj.data is None
                pass

        # Good to go.
        TomoObj.FLAG_FILE_CHECK = True
        logger.info("file check completed")