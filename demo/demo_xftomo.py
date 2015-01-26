"""Demonstrate reocnstruction of XFTomo data.

"""
import tomopy
import ipdb



"""
# Read HDF5 files.
data, theta, channel_names = tomopy.import_aps_2ide('/home/david/python/tomopy/demo/data/tomo/2xfm_{:04d}.h5',
                            f_start=100,
                            f_end=164,
                            f_exclude=[140, 141, 142, 143, 145]
                            )

# xftomo object creation and pipeline of methods.
d = tomopy.xftomo_dataset(data=data, theta=theta, channel_names=channel_names, log='debug')
d.zinger_removal(zinger_level=10000, median_width=3)
d.align_projections(output_gifs=True, output_filename='/tmp/projections.gif')
d.diagnose_center()
d.optimize_center()
d.art()
tomopy.xftomo_write(d.data_recon, output_file='/tmp/art_{:}_{:}.tif')
d.sirt()
tomopy.xftomo_write(d.data_recon, output_file='/tmp/sirt_{:}_{:}.tif')
d.gridrec()
tomopy.xftomo_write(d.data_recon, output_file='/tmp/gridrec_{:}_{:}.tif')
d.mlem()
tomopy.xftomo_write(d.data_recon, output_file='/tmp/mlem_{:}_{:}.tif')
d.pml()
tomopy.xftomo_write(d.data_recon, output_file='/tmp/pml_{:}_{:}.tif')
ipdb.set_trace()
"""
