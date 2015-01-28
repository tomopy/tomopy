"""Demonstrate reocnstruction of XFTomo data.

"""
import tomopy
import ipdb
import scipy.ndimage.interpolation as spni
import phantom
import numpy as np
from skimage.transform import radon
import pylab as pyl
#pyl.ion()

"""
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

ipdb.set_trace()
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
"""

# REAL TOMO DATA
#---------------

# Read HDF5 files.
data, theta, channel_names = tomopy.import_aps_2ide('/home/david/python/tomopy/demo/data/tomo/2xfm_{:04d}.h5',
                            f_start=100,
                            f_end=164,
                            f_exclude=[140, 141, 142, 143, 145]
                            )

channel=15
# xftomo object creation and pipeline of methods.
d = tomopy.xftomo_dataset(data=data, theta=theta, channel_names=channel_names, log='debug')
tomopy.xftomo_writer(d.data, output_file='/tmp/projections/projection_{:}_{:}.png')
ipdb.set_trace()

d.zinger_removal(zinger_level=10000, median_width=3)
d.align_projections(output_gifs=True, output_filename='/tmp/projections.gif')
d.diagnose_center()
d.optimize_center()
tomopy.xftomo_writer(d.data, channel=channel, output_file='/tmp/projections/projection_{:}_{:}.tif')
ipdb.set_trace()
d.art(channel=channel)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/art/art_{:}_{:}.tif')

d.sirt(channel=channel)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/sirt/sirt_{:}_{:}.tif')

d.theta=theta
d.gridrec(channel=channel, fluorescence=1)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/gridrec/gridrec_{:}_{:}.tif')

d.mlem(channel=channel)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/mlem/mlem_{:}_{:}.tif')

d.pml(channel=channel)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/pml/pml_{:}_{:}.tif')

