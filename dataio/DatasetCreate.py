# -*- coding: utf-8 -*-
# Filename: DatasetCreate.py
from DatasetTypes import hdf5

class Dataset():
    def __init__(self):
        pass

    def readHdf5(self, fileName,
                 projectionsStart=None,
                 projectionsEnd=None,
                 projectionsStep=None,
                 slicesStart=None,
                 slicesEnd=None,
                 slicesStep=None,
                 pixelsStart=None,
                 pixelsEnd=None,
                 pixelsStep=None,
                 whiteStart=None,
                 whiteEnd=None,
                 darkStart=None,
                 darkEnd=None):
        """ Read Data Exchange HDF5 file.

        Parameters
        ----------
        fileName : str
            Input file.

        projectionsStart, projectionsEnd, projectionsStep : scalar, optional
            Values of the start, end and step of the projections to
            be used for slicing for the whole ndarray.

        slicesStart, slicesEnd, slicesStep : scalar, optional
            Values of the start, end and step of the slices to
            be used for slicing for the whole ndarray.

        pixelsStart, pixelsEnd, pixelsStep : scalar, optional
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
        self.fileName = fileName

        # Initialize f to null.
        f = None

        # Get the fileName in lower case.
        lFn = fileName.lower()

        # Split the string with the delimeter '.'
        end = lFn.split('.')

        # If the string has an extension.
        if len(end) > 1:
            # Check.
            if end[1] == 'h5' or end[1] == 'hdf':
                f = hdf5()

        # If f != None the call read on it.
        if not f == None:
            # Read data from exchange group.
            self.data = f.read(fileName,
                                arrayName='exchange/data',
                                projectionsStart=projectionsStart,
                                projectionsEnd=projectionsEnd,
                                projectionsStep=projectionsStep,
                                slicesStart=slicesStart,
                                slicesEnd=slicesEnd,
                                slicesStep=slicesStep,
                                pixelsStart=pixelsStart,
                                pixelsEnd=pixelsEnd,
                                pixelsStep=pixelsStep)

            # Read white field data from exchange group.
            self.white = f.read(fileName,
                                arrayName='exchange/data_white',
                                projectionsStart=whiteStart,
                                projectionsEnd=whiteEnd,
                                slicesStart=slicesStart,
                                slicesEnd=slicesEnd,
                                slicesStep=slicesStep,
                                pixelsStart=pixelsStart,
                                pixelsEnd=pixelsEnd,
                                pixelsStep=pixelsStep)

            # Read dark field data from exchange group.
            self.dark = f.read(fileName,
                                arrayName='exchange/data_dark',
                                projectionsStart=darkStart,
                                projectionsEnd=darkEnd,
                                slicesStart=slicesStart,
                                slicesEnd=slicesEnd,
                                slicesStep=slicesStep,
                                pixelsStart=pixelsStart,
                                pixelsEnd=pixelsEnd,
                                pixelsStep=pixelsStep)

            # Assign the rotation center.
            self.center = self.data.shape[2] / 2

            # Assign angles.
            self.angles = None

        else:
            print 'Unsupported file.'
