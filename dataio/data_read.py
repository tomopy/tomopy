# -*- coding: utf-8 -*-
# file_name: data_read.py
from file_types import Hdf5

class Dataset():
    def __init__(self):
        pass

    def read_hdf5(self, file_name,
                  projections_start=None,
                  projections_end=None,
                  projections_step=None,
                  slices_start=None,
                  slices_end=None,
                  slices_step=None,
                  pixels_start=None,
                  pixels_end=None,
                  pixels_step=None,
                  whiteStart=None,
                  whiteEnd=None,
                  darkStart=None,
                  darkEnd=None):
        """ Read Data Exchange HDF5 file.

        Parameters
        ----------
        file_name : str
            Input file.

        projections_start, projections_end, projections_step : scalar, optional
            Values of the start, end and step of the projections to
            be used for slicing for the whole ndarray.

        slices_start, slices_end, slices_step : scalar, optional
            Values of the start, end and step of the slices to
            be used for slicing for the whole ndarray.

        pixels_start, pixels_end, pixels_step : scalar, optional
            Values of the start, end and step of the pixels to
            be used for slicing for the whole ndarray.

        whiteStart, whiteEnd : scalar, optional
            Values of the start, end and step of the
            slicing for the whole white field shots.

        darkStart, darkEnd : scalar, optional
            Values of the start, end and step of the
            slicing for the whole dark field shots.
        """
        print "Reading data..."
        self.file_name = file_name

        # Initialize f to null.
        f = None

        # Get the file_name in lower case.
        lFn = file_name.lower()

        # Split the string with the delimeter '.'
        end = lFn.split('.')

        # If the string has an extension.
        if len(end) > 1:
            # Check.
            if end[1] == 'h5' or end[1] == 'hdf':
                f = Hdf5()

        # If f != None the call read on it.
        if not f == None:
            # Read data from exchange group.
            self.data = f.read(file_name,
                                arrayName='exchange/data',
                                projections_start=projections_start,
                                projections_end=projections_end,
                                projections_step=projections_step,
                                slices_start=slices_start,
                                slices_end=slices_end,
                                slices_step=slices_step,
                                pixels_start=pixels_start,
                                pixels_end=pixels_end,
                                pixels_step=pixels_step)

            # Read white field data from exchange group.
            self.white = f.read(file_name,
                                arrayName='exchange/data_white',
                                projections_start=whiteStart,
                                projections_end=whiteEnd,
                                slices_start=slices_start,
                                slices_end=slices_end,
                                slices_step=slices_step,
                                pixels_start=pixels_start,
                                pixels_end=pixels_end,
                                pixels_step=pixels_step)

            # Read dark field data from exchange group.
            self.dark = f.read(file_name,
                                arrayName='exchange/data_dark',
                                projections_start=darkStart,
                                projections_end=darkEnd,
                                slices_start=slices_start,
                                slices_end=slices_end,
                                slices_step=slices_step,
                                pixels_start=pixels_start,
                                pixels_end=pixels_end,
                                pixels_step=pixels_step)

            # Assign the rotation center.
            self.center = self.data.shape[2] / 2

            # Assign angles.
            self.angles = None

        else:
            print 'Unsupported file.'
