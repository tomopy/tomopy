# -*- coding: utf-8 -*-
# Filename: diagnose_center.py
import numpy as np
import shutil

def diagnose_center(data,
                    slice_no=None,
                    center_start=None,
                    center_end=None,
                    center_step=None):
    """ Diagnostic tools to find rotation center.
        
    Helps finding the rotation center manually by
    visual inspection of the selected reconstructions
    with different centers. The outputs for different
    centers are put into ``data/diagnose`` directory
    and the corresponding center positions are printed
    so that one can skim through the images and
    select the best.
        
    Parameters
    ----------
    data : ndarray
        Input data.
        
    slice_no : scalar, optional
        The index of the slice to be used for diagnostics.
        Default is the central slice.
        
    center_start, center_end, center_step : scalar, optional
        Values of the start, end and step of the center values to
        be used for diagnostics.
    """
    print "Diagnosing rotation center..."
    num_projections =  data.shape[0]
    num_slices =  data.shape[1]
    num_pixels =  data.shape[2]
    
    if slice_no is None:
        slice_no = numSlices / 2
    if center_start is None:
        center_start = (num_pixels / 2) - 20
    if center_end is None:
        center_end = (num_pixels / 2) + 20
    if center_step is None:
        center_step = 1
    
    sliceData = self.data[:, sliceNo, :]
    center = np.arange(center_start, center_end, center_step)
    num_center = center.size
    stacked_slices = np.zeros((num_projections, num_center, num_pixels),
                             dtype='float')
    for m in range(num_center):
        stacked_slices[:, m, :] = sliceData
    dataset = createDataObject(stacked_slices, center=center)
    recon = tomoRecon.tomoRecon(dataset)
    recon.run(dataset, printInfo=False)
    if os.path.isdir('data/diagnose'):
        shutil.rmtree('data/diagnose')
        tiff.write(recon.data, output_file='data/diagnose/center_.tiff',)
        for m in range(num_center):
            print 'Center for data/diagnose/xxx' + str(m) + '.tiff: ' + str(center[m])
