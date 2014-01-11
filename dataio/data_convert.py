# -*- coding: utf-8 -*-
# file_name: data_convert.py
import numpy as np
import os
import h5py
from dataio.file_types import Tiff, Hdf4, Hdf5
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry

class Convert():
    def __init__(self, data=None, white=None, dark=None,
                 center=None, angles=None):
        self.data = data
        self.white = white
        self.dark = dark
        self.center = center
        self.angles = angles
    
    def tiff(self, file_name,
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
                file_name_white=None,
                white_start=0,
                white_end=0,
                white_step=1,
                file_name_dark=None,
                dark_start=0,
                dark_end=0,
                dark_step=1,
                digits=4,
                zeros=True,
                dtype='uint16',
                sample_name=None,
                verbose=True):
        """Read a stack of TIFF files in a folder.

        Parameters
        ----------
        file_name : str
            Base name of the input TIFF files.
            For example if the projections names are /local/data/test_XXXX.tiff
            file_name is /local/data/test_.tiff
            
        hdf5_file_name : str
            HDF5/data exchange file name

        projections_start, projections_end, projections_step : scalar, optional
            start and end index for the projection Tiff files to load. Use step define a stride.

        slices_start, slices_end, slices_step : scalar, optional
            start and end pixel of the projection image to load along the rotation axis. Use step define a stride.

        pixels_start, pixels_end, pixels_step : not used yet ...

        file_name_white : str
            Base name of the white field input TIFF files: string optinal.
            For example if the white field names are /local/data/test_bg_XXXX.tiff
            file_name is /local/data/test_bg_.tiff
            if omitted file_name_white = file_name.

        white_start, white_end : scalar, optional
            start and end index for the white field Tiff files to load. Use step define a stride.

        file_name_dark : str
            Base name of the dark field input TIFF files: string optinal.
            For example if the white field names are /local/data/test_dk_XXXX.tiff
            file_name is /local/data/test_dk_.tiff
            if omitted file_name_dark = file_name.

        dark_start, dark_end : scalar, optional
            start and end index for the dark field Tiff files to load. Use step define a stride.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        Returns
        -------
        inputData : list of tiff files contating projections, white and dark images

        Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        # Check if hdf5_file_name already exists.
        if verbose: print "Check if the HDF5 was already created ..."
        #self.hdf5_file_name = file_name

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
                

        # If f == None the call converts the tiff files.
        if (hdf5_file_extension and (os.path.isfile(hdf5_file_name) == False)):
            # Prepare tiff file names to be read.
            if file_name_white == None:
                    file_name_white = file_name
                    if verbose: print "File Name White = ", file_name_white
            if file_name_dark == None:
                    file_name_dark = file_name
                    if verbose: print "File Name Dark = ", file_name_dark

            if verbose: print "File Name Projections = ", file_name
            if verbose: print "File Name White = ", file_name_white
            if verbose: print "File Name Dark = ", file_name_dark

            if file_name.endswith('tif') or \
               file_name.endswith('tiff'):
                dataFile = file_name.split('.')[-2]
                dataExtension = file_name.split('.')[-1]
            if file_name_white.endswith('tif') or \
               file_name_white.endswith('tiff'):
                dataFileWhite = file_name_white.split('.')[-2]
                dataExtensionWhite = file_name_white.split('.')[-1]
            if file_name_dark.endswith('tif') or \
               file_name_dark.endswith('tiff'):
                dataFileDark = file_name_dark.split('.')[-2]
                dataExtensionDark = file_name_dark.split('.')[-1]

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
                    f = Tiff()
                    tmpdata = f.read(fileName,
                                        x_start=slices_start,
                                        x_end=slices_end,
                                        x_step=slices_step,
                                        dtype=dtype)
                    if m == 0: # Get resolution once.
                        inputData = np.empty((projections_end-projections_start,
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype)
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
                    f = Tiff()
                    tmpdata = f.read(fileName,
                                        x_start=slices_start,
                                        x_end=slices_end,
                                        x_step=slices_step,
                                        dtype=dtype)
                    if m == 0: # Get resolution once.
                        if verbose: print 'Tempory Data Array: (white_end-white_start) =', (white_end-white_start)/white_step + 1, 'tmpdata shape = (', tmpdata.shape[0], ',', tmpdata.shape[1], ')'
                        inputData = np.empty(((white_end - white_start)/white_step + 1,
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype)
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
                    f = Tiff()
                    tmpdata = f.read(fileName,
                                        x_start=slices_start,
                                        x_end=slices_end,
                                        x_step=slices_step,
                                        dtype=dtype)
                    if m == 0: # Get resolution once.
                        if verbose: print 'Tempory Data Array: (dark_end-dark_start) =', (dark_end-dark_start), 'tmpdata shape = (', tmpdata.shape[0], ',', tmpdata.shape[1], ')'
                        inputData = np.empty(((dark_end - dark_start),
                                            tmpdata.shape[0],
                                            tmpdata.shape[1]),
                                            dtype=dtype)
                    inputData[m, :, :] = tmpdata
            if len(ind) > 0:
                self.dark = inputData
                
            # Fabricate theta values.
            z = np.arange(projections_end - projections_start);
            if verbose: print z, len(z)
                
            # Fabricate theta values
            self.angles = (z * float(projections_angle_range) / (len(z) - 1))
            if verbose: print self.angles

            # Write HDF5 file.
            # Open DataExchange file
            f = DataExchangeFile(hdf5_file_name, mode='w') 

            # Create core HDF5 dataset in exchange group for projections_theta_range
            # deep stack of x,y images /exchange/data
            f.add_entry( DataExchangeEntry.data(data={'value': self.data, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(theta={'value': self.angles, 'units':'degrees'}))
            f.add_entry( DataExchangeEntry.data(data_dark={'value': self.dark, 'units':'counts', 'axes':'theta_dark:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(data_white={'value': self.white, 'units':'counts', 'axes':'theta_white:y:x', 'dataset_opts':  {'compression': 'gzip', 'compression_opts': 4} }))
            f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))
            if verbose:
                print "Sample name = ", sample_name
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
                if verbose: print 'HDF5 already exists.'
            if (hdf5_file_extension == False):
                if verbose: print "HDF file extension must be .h5 or .hdf"


    def hdf4(input_file,
                     input_start,
                     input_end,
                     slices_start=None,
                     slices_end=None,
                     slices_step=None,
                     pixels_start=None,
                     pixels_end=None,
                     pixels_step=None,
                     digits=4,
                     zeros=True,
                     dtype='uint16',
                     array_name=None,
                     output_file='myfile.h5',
                     white_file=None,
                     white_start=None,
                     white_end=None,
                     dark_file=None,
                     dark_start=None,
                     dark_end=None):
        """ Converts a stack of projection 16-bit HDF files
        in a folder to a single HDF5 file. The dataset is
        constructed using the projection data, white field
        and dark field images.

        Parameters
        ----------
        input_file : str
            Name of the generic input file name
            for all the HDF4 files to be assembled.

        input_start, input_end : scalar
            Determines the portion of the TIFF images
            to be used for assembling the HDF file.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the TIFF file.

        hdftype : scalar, optional
            Type of HDF5 files to be read (4:HDF4, 5:HDF5)

        output_file : str
            Name of the output HDF5 file.

        white_file : str, optional
            Name of the generic input file name
            for all the white field
            HDF4 files to be assembled.

        white_start, white_end : scalar, optional
            Determines the portion of the white
            field HDF4 images to be used for
            assembling HDF5 file.

        dark_file : str, optional
            Name of the generic input file name
            for all the white field
            HDF4 files to be assembled.

        dark_start, dark_end : scalar, optional
            Determines the portion of the dark
            field HDF4 images to be used for
            assembling HDF5 file.
        """
        # Create new folders.
        dirPath = os.path.dirname(output_file)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # Prepare HDF5 file.
        print 'Assembling HDF5 file: ' + os.path.realpath(output_file)
        f = h5py.File(output_file, 'w')
        f.create_dataset('implements', data='exchange')
        exchangeGrp = f.create_group("exchange")

        # Update HDF5 file in chunks.
        chunkSize = 20
        ind1 = np.int(np.floor(np.float(input_start) / chunkSize))
        ind2 = np.int(np.floor(np.float(input_end) / chunkSize))
        for m in range(ind2-ind1):
            indStart = (m * chunkSize) + input_start
            indEnd = indStart + chunkSize
            if indEnd > input_end:
                indEnd = input_end

            # Read projection files in the given folder.
            input_data = read_stack(input_file,
                                   input_start=indStart,
                                   input_end=indEnd,
                                   slices_start=slices_start,
                                   slices_end=slices_end,
                                   slices_step=slices_step,
                                   pixels_start=pixels_start,
                                   pixels_end=pixels_end,
                                   pixels_step=pixels_step,
                                   digits=digits,
                                   zeros=zeros,
                                   array_name=array_name)

            # Update HDF5 file.
            if m == 0:
                dset = exchangeGrp.create_dataset('data',
                                                  (input_end-input_start,
                                                   input_data.shape[1],
                                                   input_data.shape[2]),
                                                  dtype=dtype)
            dset[(indStart-input_start):(indEnd-input_start), :, :] = input_data

        # Read white-field TIFF files in the given folder.
        if not white_file == None:
            whiteData = read_stack(white_file,
                                   white_start,
                                   white_end,
                                   digits=digits,
                                   zeros=zeros,
                                   array_name=array_name)
            exchangeGrp.create_dataset('data_white', data=whiteData, dtype=dtype)


        # Read dark-field TIFF files in the given folder.
        if not dark_file == None:
            darkData = read_stack(dark_file,
                                  dark_start,
                                  dark_end,
                                  digits=digits,
                                  zeros=zeros,
                                  array_name=array_name)
            exchangeGrp.create_dataset('data_dark', data=darkData, dtype=dtype)
        f.close()

    def read_stack(input_file,
                   input_start,
                   input_end,
                   slices_start=None,
                   slices_end=None,
                   slices_step=None,
                   pixels_start=None,
                   pixels_end=None,
                   pixels_step=None,
                   digits=4,
                   zeros=True,
                   dtype='uint16',
                   array_name=None):
        """Read a stack of files in a folder.

        Parameters
        ----------
        input_file : str
            Name of the input file.

        input_start, input_end : scalar
            Determines the portion of the images
            to be used for assembling the HDF file.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the
            slicing for the whole ndarray.

        digits : scalar, optional
            Number of digits used for file indexing.
            For example if 4: test_XXXX.tiff

        zeros : bool, optional
            If ``True`` assumes all indexing uses four digits
            (0001, 0002, ..., 9999). If ``False`` omits zeros in
            indexing (1, 2, ..., 9999)

        dtype : str, optional
            Corresponding Numpy data type of the file.

        Returns
        -------
        input_data : ndarray
            Output 2-D matrix as numpy array.

        .. See also:: http://docs.scipy.org/doc/numpy/user/basics.types.html
        """
        # Split the string with the delimeter '.'
        #data_file = input_file.split('.')[-3] + '.' + input_file.split('.')[-2]
        data_file = input_file.split('.')[-2]
        data_extension = input_file.split('.')[-1]

        file_index = ["" for x in range(digits)]
        for m in range(digits):
            if zeros is True:
                file_index[m] = '0' * (digits - m - 1)

            elif zeros is False:
                file_index[m] = ''

        ind = range(input_start, input_end)
        for m in range(len(ind)):
            for n in range(digits):
                if ind[m] < np.power(10, n + 1):
                    file_name = data_file + file_index[n] + str(ind[m]) + '.' + data_extension
                    break

            if os.path.isfile(file_name):
                print 'Reading file: ' + os.path.realpath(file_name)
                if data_extension == 'tiff' or data_extension == 'tif':
                    f = Tiff()
                    tmpdata = f.read(file_name, dtype=dtype,
                                    slices_start=slices_start,
                                    slices_end=slices_end,
                                    slices_step=slices_step,
                                    pixels_start=pixels_start,
                                    pixels_end=pixels_end,
                                    pixels_step=pixels_step)
                elif data_extension == 'hdf':
                    f = Hdf4()
                    tmpdata = f.read(file_name,
                                    slices_start=slices_start,
                                    slices_end=slices_end,
                                    slices_step=slices_step,
                                    pixels_start=pixels_start,
                                    pixels_end=pixels_end,
                                    pixels_step=pixels_step,
                                    array_name=array_name)
            if m == 0: # Get resolution once.
                input_data = np.empty((input_end-input_start,
                                      tmpdata.shape[0],
                                      tmpdata.shape[1]),
                                     dtype=dtype)
            input_data[m, :, :] = tmpdata
        return input_data
