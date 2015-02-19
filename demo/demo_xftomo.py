# -*- coding: utf-8 -*-
"""Demonstrate reocnstruction of XFTomo data.

"""
import tomopy
import ipdb
import scipy.ndimage.interpolation as spni
import phantom
import numpy as np
from skimage.transform import radon
import pylab as pyl

# SIMULATED TOMO DATA
#---------------------
p = 128
n_projections = 200
emission = True
iters = 1
simulate_misaligned_projections = False
theta = np.linspace(0,180,num=n_projections,endpoint=True)
msl = phantom.modified_shepp_logan((p,p,p))
data = np.zeros((1, n_projections, p, p))
for i in range(p):
    data[0,:,i,:] = np.rollaxis(radon(msl[:,i,:], theta=theta, circle=True),1,0)

if simulate_misaligned_projections:
    for i in range(n_projections):
        sx, sy = 3*(np.random.random()-0.5), 3*(np.random.random()-0.5)
        data[0,i,:,:]=spni.shift(data[0,i,:,:], (sx, sy))


d = tomopy.xftomo_dataset(data=data, theta=theta, channel_names=['Modified Shepp-Logan'], log='debug')
tomopy.xftomo_writer(d.data, channel=0, output_file='/tmp/projections/projection_{:}_{:}.tif')
if simulate_misaligned_projections:
    d.align_projections(output_gifs=True, output_filename='/tmp/projections.gif')
    d.align_projections(output_gifs=True, output_filename='/tmp/projections.gif')

d.diagnose_center()
d.optimize_center()

d.art(channel=0, emission=emission, iters=iters)
tomopy.xftomo_writer(d.data_recon, channel=0, output_file='/tmp/art/art_{:}_{:}.tif')

d.sirt(channel=0, emission=emission, iters=iters)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/sirt/sirt_{:}_{:}.tif')

d.theta = theta
d.gridrec(channel=0, fluorescence=1)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/gridrec/gridrec_{:}_{:}.tif')

d.mlem(channel=0, emission=emission, iters=iters)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/mlem/mlem_{:}_{:}.tif')

d.pml(channel=0, emission=emission, iters=iters)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/pml/pml_{:}_{:}.tif')
