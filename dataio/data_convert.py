# -*- coding: utf-8 -*-
# file_name: data_convert.py
import numpy as np
import os
import h5py
from dataio.file_types import Tiff, Hdf4, Hdf5, Txrm, Xrm
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry

class Convert():
    def __init__(self, data=None, white=None, dark=None,
                 center=None, angles=None):
        self.data = data
        self.white = white
        self.dark = dark
        self.center = center
        self.theta = angles
    
    def series_of_images(self, file_name,
                hdf5_file_name,
                projections_start=0,
                projections_end=0,
                projections_step=1,
                projections_angle_range=180,
                slices_start=None,
                slices_end=None,
                slices_step=None,
                pixels_start=None,
                pixels_end=None,
                pixels_step=None,
                white_file_name=None,
                white_start=0,
                white_end=0,
                white_step=1,
                dark_file_name=None,
                dark_start=0,
                dark_end=0,
                dark_step=1,
                digits=4,
                zeros=True,
                dtype='uint16',
                data_type='tiff',
                sample_name=None,
                verbose=True):
        """Read a stack of HDF-4 or TIFF files in a folder.

        Parameters
        ----------
        file_name : str
            Base name of the input HDF-4 or TIFF files.
            For example if the projections names are /local/data/test_XXXX.hdf
            file_name is /local/data/test_.hdf
            
        hdf5_file_name : str
            HDF5/data exchange file name

        projections_start, projections_end, projections_step : scalar, optional
            start and end index for the projection Tiff files to load. Use step define a stride.

        slices_start, slices_end, slices_step : scalar, optional
            start and end pixel of the projection image to load along the rotation axis. Use step define a stride.

        pixels_start, pixels_end, pixels_step : not used yet.

        white_file_name : str
            Base name of the white field input HDF-4 or TIFF files: string optional.
            For example if the white field names are /local/data/test_bg_XXXX.hdf
            file_name is /local/data/test_bg_.hdf
            if omitted white_file_name = file_name.

        white_start, white_end : scalar, optional
            start and end index for the white field Tiff files to load. Use step define a stride.

        dark_file_name : str
            Base name of the dark field input HDF-4 or TIFF files: string optinal.
            For example if the white field names are /local/data/test_dk_XXXX.hdf
            file_name is /local/data/test_dk_.hdf
            if omitted dark_file_name = file_name.

        dark_start, dark_end : scalar, optional
            start and end index for the dark field Tiff files to load. Use step define a stride.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.hdf

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the HDF-4 or TIFF file.

        data_type : str, optional
            if 'hdf4q m    ' will convert HDF-4 files (old 2-BM), deafult is 'tiff'

        Returns
        -------
        inputData : list of hdf files contating projections, white and dark images

        Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """

        # Initialize f to null.
        hdf5_file_extension = False

        # Get the file_name in lower case.
        lFn = hdf5_file_name.lower()

        # Split the string with the delimeter '.'
        end = lFn.split('.')
        if verbose: print end
        # If the string has an extension.
        if len(end) > 1:
            # Check.
            if end[len(end) - 1] == 'h5' or end[len(end) - 1] == 'hdf':
                hdf5_file_extension = True
                if verbose: print "HDF file extension is .h5 or .hdf"
            else:
                hdf5_file_extension = False
                if verbose: print "HDF file extension must be .h5 or .hdf"
                

        # If the extension is correct and the file does not exists then convert
        if (hdf5_file_extension and (os.path.isfile(hdf5_file_name) == False)):
            # Create new folder.
            dirPath = os.path.dirname(hdf5_file_name)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            # Prepare hdf file names to be read.
            if white_file_name == None:
                    white_file_name = file_name
                    if verbose: print "File Name White = ", white_file_name
            if dark_file_name == None:
                    dark_file_name = file_name
                    if verbose: print "File Name Dark = ", dark_file_name

            if verbose: print "File Name Projections = ", file_name
            if verbose: print "File Name White = ", white_file_name
            if verbose: print "File Name Dark = ", dark_file_name

            if (data_type is 'hdf4'):
                if file_name.endswith('h4') or \
                   file_name.endswith('hdf'):
                    dataFile = file_name.split('.')[-2]
                    dataExtension = file_name.split('.')[-1]
                if white_file_name.endswith('h4') or \
                   white_file_name.endswith('hdf'):
                    dataFileWhite = white_file_name.split('.')[-2]
                    dataExtensionWhite = white_file_name.split('.')[-1]
                if dark_file_name.endswith('h4') or \
                   dark_file_name.endswith('hdf'):
                    dataFileDark = dark_file_name.split('.')[-2]
                    dataExtensionDark = dark_file_name.split('.')[-1]
            else:
                if file_name.endswith('tif') or \
                   file_name.endswith('tiff'):
                    dataFile = file_name.split('.')[-2]
                    dataExtension = file_name.split('.')[-1]
                if white_file_name.endswith('tif') or \
                   white_file_name.endswith('tiff'):
                    dataFileWhite = white_file_name.split('.')[-2]
                    dataExtensionWhite = white_file_name.split('.')[-1]
                if dark_file_name.endswith('tif') or \
                   dark_file_name.endswith('tiff'):
                    dataFileDark = dark_file_name.split('.')[-2]
                    dataExtensionDark = dark_file_name.split('.')[-1]


            fileIndex = ["" for x in range(digits)]

            for m in range(digits):
                if zeros is True:
                   fileIndex[m] = '0' * (digits - m - 1)

                elif zeros is False:
                   fileIndex[m] = ''
                   
            # Reading projections.
            ind = range(projections_start, projections_end)
            if verbose: print 'Projections: Start =', projections_start, 'End =', projections_end, 'Step =', projections_step, 'ind =', ind, 'range(digits) =', range(digits),'len(ind) =', len(ind), 'range(lan(ind)) =', range(len(ind))
            for m in range(len(ind)):
                for n in range(digits):
                    if verbose: print 'n =', n, 'ind[m]', ind[m], '<', np.power(10, n + 1)
                    if ind[m] < np.power(10, n + 1):
                        fileName = dataFile + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                        if verbose: print 'Generating file names: ' + fileName
                        break

                if os.path.isfile(fileName):
                    if verbose: print 'Reading projection file: ' + os.path.realpath(fileName)
                    if verbose: print 'data type: ', data_type
                    if (data_type is 'hdf4'):
                        f = Hdf4()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            array_name = 'data'
                                         )
                    else:
                        f = Tiff()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            dtype=dtype
                                         )
                    if m == 0: # Get resolution once.
                        inputData = np.empty((projections_end-projections_start,
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype
                                    )
                    inputData[m, :, :] = tmpdata
            if len(ind) > 0:
                self.data = inputData

            # Reading white fields.
            ind = range(white_start, white_end, white_step)
            if verbose: print 'White: Start =', white_start, 'End =', white_end, 'Step =', white_step, 'ind =', ind, 'range(digits) =', range(digits),'len(ind) =', len(ind), 'range(lan(ind)) =', range(len(ind))
            for m in range(len(ind)):
                for n in range(digits):
                    if verbose: print 'n =', n, 'ind[m]', ind[m], '<', np.power(10, n + 1)
                    if ind[m] < np.power(10, n + 1):
                        fileName = dataFileWhite + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                        if verbose: print fileName
                        break

                if os.path.isfile(fileName):
                    if verbose: print 'Reading white file: ' + os.path.realpath(fileName)
                    if verbose: print 'data type: ', data_type
                    if (data_type is 'hdf4'):
                        f = Hdf4()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            array_name = 'data'
                                         )
                    else:
                        f = Tiff()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            dtype=dtype
                                         )
                    if m == 0: # Get resolution once.
                        if verbose: print 'Tempory Data Array: (white_end-white_start) =', (white_end-white_start)/white_step + 1, 'tmpdata shape = (', tmpdata.shape[0], ',', tmpdata.shape[1], ')'
                        inputData = np.empty(((white_end - white_start)/white_step + 1,
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype
                                        )
                    if verbose: print 'm', m
                    inputData[m, :, :] = tmpdata
            if len(ind) > 0:
                self.white = inputData
                
            # Reading dark fields.
            ind = range(dark_start, dark_end, dark_step)
            if verbose: print 'Dark: Start =', dark_start, 'End =', dark_end, 'Step =', dark_step, 'ind =', ind, 'range(digits) =', range(digits),'len(ind) =', len(ind), 'range(lan(ind)) =', range(len(ind))
            for m in range(len(ind)):
                for n in range(digits):
                    if ind[m] < np.power(10, n + 1):
                        fileName = dataFileDark + fileIndex[n] + str(ind[m]) + '.' + dataExtension
                        if verbose: print fileName
                        break

                if os.path.isfile(fileName):
                    if verbose: print 'Reading dark file: ' + os.path.realpath(fileName)
                    if verbose: print 'data type: ', data_type
                    if (data_type is 'hdf4'):
                        f = Hdf4()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            array_name = 'data'
                                         )
                    else:
                        f = Tiff()
                        tmpdata = f.read(fileName,
                                            x_start=slices_start,
                                            x_end=slices_end,
                                            x_step=slices_step,
                                            dtype=dtype
                                         )
                    if m == 0: # Get resolution once.
                        if verbose: print 'Tempory Data Array: (dark_end-dark_start) =', (dark_end-dark_start), 'tmpdata shape = (', tmpdata.shape[0], ',', tmpdata.shape[1], ')'
                        inputData = np.empty(((dark_end - dark_start),
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype
                                        )
                    inputData[m, :, :] = tmpdata
            if len(ind) > 0:
                self.dark = inputData
                
            # Fabricate theta values.
            z = np.arange(projections_end - projections_start);
            if verbose: print z, len(z)
                
            # Fabricate theta values
            self.theta = (z * float(projections_angle_range) / (len(z) - 1))
            if verbose: print self.theta

            # Write HDF5 file.
            # Open DataExchange file
            f = DataExchangeFile(hdf5_file_name, mode='w') 

            # Create core HDF5 dataset in exchange group for projections_theta_range
            # deep stack of x,y images /exchange/data
            f.add_entry( DataExchangeEntry.data(data={'value': self.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(theta={'value': self.theta, 'units':'degrees'}))
            f.add_entry( DataExchangeEntry.data(data_dark={'value': self.dark, 'units':'counts', 'axes':'theta_dark:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(data_white={'value': self.white, 'units':'counts', 'axes':'theta_white:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
            if verbose: print "Sample name = ", sample_name
            if (sample_name == None):
                sample_name = end[0]
                f.add_entry( DataExchangeEntry.sample( name={'value':sample_name}, description={'value':'Sample name was assigned by the HDF5 converter and based on the HDF5 fine name'}))
                if verbose: print "Assigned default file name", end[0]
            else:
                f.add_entry( DataExchangeEntry.sample( name={'value':sample_name}, description={'value':'Sample name was read from the user log file'}))
                if verbose: print "Assigned file name from user log"
                    
            
            f.close()
        else:
            if os.path.isfile(hdf5_file_name):
                print 'HDF5 already exists. Nothing to do ...'
            if (hdf5_file_extension == False):
                print "HDF file extension must be .h5 or .hdf"


    def x_radia(self, file_name,
                hdf5_file_name,
                projections_start=0,
                projections_end=0,
                projections_step=1,
                projections_angle_range=180,
                projections_data_type='txrm',
                slices_start=None,
                slices_end=None,
                slices_step=None,
                pixels_start=None,
                pixels_end=None,
                pixels_step=None,
                white_file_name='',
                white_start=0,
                white_end=0,
                white_step=1,
                white_data_type='xrm',
                dark_file_name='',
                dark_start=0,
                dark_end=0,
                dark_step=1,
                dark_data_type='xrm',
                sample_name=None,
                verbose=True):
        """Read a stack of HDF-4 or TIFF files in a folder.

        Parameters
        ----------
        file_name : str
            Base name of the input HDF-4 or TIFF files.
            For example if the projections names are /local/data/test_XXXX.hdf
            file_name is /local/data/test_.hdf
            
        hdf5_file_name : str
            HDF5/data exchange file name

        projections_start, projections_end, projections_step : scalar, optional
            start and end index for the projection Tiff files to load. Use step define a stride.

        slices_start, slices_end, slices_step : scalar, optional
            start and end pixel of the projection image to load along the rotation axis. Use step define a stride.

        pixels_start, pixels_end, pixels_step : not used yet.

        white_file_name : str
            Base name of the white field input HDF-4 or TIFF files: string optional.
            For example if the white field names are /local/data/test_bg_XXXX.hdf
            file_name is /local/data/test_bg_.hdf
            if omitted white_file_name = file_name.

        white_start, white_end : scalar, optional
            start and end index for the white field Tiff files to load. Use step define a stride.

        dark_file_name : str
            Base name of the dark field input HDF-4 or TIFF files: string optinal.
            For example if the white field names are /local/data/test_dk_XXXX.hdf
            file_name is /local/data/test_dk_.hdf
            if omitted dark_file_name = file_name.

        dark_start, dark_end : scalar, optional
            start and end index for the dark field Tiff files to load. Use step define a stride.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.hdf

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the HDF-4 or TIFF file.

        data_type : str, optional
            if 'hdf4q m    ' will convert HDF-4 files (old 2-BM), deafult is 'tiff'

        Returns
        -------
        inputData : list of hdf files contating projections, white and dark images

        Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """

        # Initialize f to null.
        hdf5_file_extension = False

        # Get the file_name in lower case.
        lFn = hdf5_file_name.lower()

        # Split the string with the delimeter '.'
        end = lFn.split('.')
        if verbose: print end
        # If the string has an extension.
        if len(end) > 1:
            # Check.
            if end[len(end) - 1] == 'h5' or end[len(end) - 1] == 'hdf':
                hdf5_file_extension = True
                if verbose: print "HDF file extension is .h5 or .hdf"
            else:
                hdf5_file_extension = False
                if verbose: print "HDF file extension must be .h5 or .hdf"
                

        # If the extension is correct and the file does not exists then convert
        if (hdf5_file_extension and (os.path.isfile(hdf5_file_name) == False)):
            # Create new folder.
            dirPath = os.path.dirname(hdf5_file_name)
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            if verbose: print "File Name Projections = ", file_name
            if verbose: print "File Name White = ", white_file_name
            if verbose: print "File Name Dark = ", dark_file_name

            if os.path.isfile(file_name):
                if verbose: print 'Reading projection file: ' + os.path.realpath(file_name)
                if verbose: print 'data type: ', projections_data_type
                if (projections_data_type is 'txrm'):
                    f = Txrm()
                    tmpdata = f.read(file_name)
                    self.data = tmpdata

            if os.path.isfile(white_file_name):
                if verbose: print 'Reading white file: ' + os.path.realpath(white_file_name)
                if verbose: print 'data type: ', white_data_type
                if (white_data_type is 'xrm'):
                    f = Xrm()
                    tmpdata = f.read(white_file_name)
                    #inputData[m, :, :] = tmpdata
                    self.white = tmpdata
            else:
                nx, ny, nz = np.shape(self.data)
                self.dark = np.ones((nx,ny,1))

            if os.path.isfile(dark_file_name):
                if verbose: print 'Reading dark file: ' + os.path.realpath(dark_file_name)
                if verbose: print 'data type: ', dark_data_type
                if (white_data_type is 'xrm'):
                    f = Xrm()
                    tmpdata = f.read(dark_file_name)
                    #inputData[m, :, :] = tmpdata
                    self.dark = tmpdata
            else:
                nx, ny, nz = np.shape(self.data)
                self.dark = np.zeros((nx,ny,1))


            # Write HDF5 file.
            # Open DataExchange file
            f = DataExchangeFile(hdf5_file_name, mode='w') 

            # Create core HDF5 dataset in exchange group for projections_theta_range
            # deep stack of x,y images /exchange/data
            f.add_entry( DataExchangeEntry.data(data={'value': self.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(theta={'value': self.theta, 'units':'degrees'}))
            f.add_entry( DataExchangeEntry.data(data_dark={'value': self.dark, 'units':'counts', 'axes':'theta_dark:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(data_white={'value': self.white, 'units':'counts', 'axes':'theta_white:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
            if verbose: print "Sample name = ", sample_name
            if (sample_name == None):
                sample_name = end[0]
                f.add_entry( DataExchangeEntry.sample( name={'value':sample_name}, description={'value':'Sample name was assigned by the HDF5 converter and based on the HDF5 fine name'}))
                if verbose: print "Assigned default file name", end[0]
            else:
                f.add_entry( DataExchangeEntry.sample( name={'value':sample_name}, description={'value':'Sample name was read from the user log file'}))
                if verbose: print "Assigned file name from user log"
                    
            
            f.close()
        else:
            if os.path.isfile(hdf5_file_name):
                print 'HDF5 already exists. Nothing to do ...'
            if (hdf5_file_extension == False):
                print "HDF file extension must be .h5 or .hdf"

