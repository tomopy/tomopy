# -*- coding: utf-8 -*-
"""
This is a template script  for simple image processing tasks. 
For more functionality please check the documentation.

Have fun!
"""
# Import the glorious package.
import tomopy

# Create a NULL dataset.
# All scripts starts with this line. 
# Use ``log='DEBUG'`` argument for dbugging.
d = tomopy.Dataset()

# Import data from HDF file.
# You can use slices_start, slices_end, etc. (see docs)
# arguments to read specific slices without loading
# whole data array into memory. 
d.read('demo/data.h5')

# Normalize data. 
d.normalize()

# Find center of rotation axis. 
# See also:``diagnose_center()'' function
d.optimize_center()

# Apply stripe (ring) removal to data.
# For large artifacts increase level argument.
# Or you can simply use ``d.stripe_removal()``
# to apply maximum available level value.
# By default uses the maximum number of processors,
# but you can change it using ``num_processors`` argument.
d.stripe_removal(level=2, sigma=2)

# Apply phase retireval. 
# Units are in [cgs] and energy is in [keV].
# You can change alpha value for desired output.
# By default uses the maximum number of processors,
# but you can change it using ``num_processors`` argument.
d.phase_retrieval(pixel_size=1e-4, dist=70, energy=30, alpha=1e-5)

# Image reconstruction using Gridrec.
# For parameters see docs. If center is undefined
# gridrec() uses optimize_center() first to find center of rotation axis.
d.gridrec()

# If the image is kind of binary, then you
# can try thresholding based segmentation.
# See also: ``region_segment(low=0.35, high=0.65)``
d.threshold_segment()

# Export data to file.
d.recon_to_tiff('demo/test_')
