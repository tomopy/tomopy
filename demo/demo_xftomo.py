"""Demonstrate reocnstruction of XFTomo data.

"""
import tomopy
import ipdb

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
ipdb.set_trace()
tomopy.xftomo_writer(d.data, channel=9, output_file='/tmp/projection_{:}_{:}.tif')
d.art(channel=9)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/art/art_{:}_{:}.tif')
d.sirt(channel=9)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/sirt/sirt_{:}_{:}.tif')
d.gridrec(channel=9)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/gridrec/gridrec_{:}_{:}.tif')
d.mlem(channel=9)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/mlem/mlem_{:}_{:}.tif')
d.pml(channel=9)
tomopy.xftomo_writer(d.data_recon, output_file='/tmp/pml/pml_{:}_{:}.tif')

