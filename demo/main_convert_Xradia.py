# -*- coding: utf-8 -*-
# Filename: main_convert_Xradia.py
""" Main program for convert xradia txrm and xrm data into tiff and HDF5 data exchange file.
"""
import dataio.xradia.xradia_xrm as xradia
import dataio.xradia.data_stack_sim as dstack
import dataio.xradia.data_struct as dstruct
from dataio.data_exchange import DataExchangeFile, DataExchangeEntry

import numpy as np
import os

import scipy

#def main():

#filename = '/local/data/databank/TXM_NSLS/XrmTxrmExampleFiles/20120627_006_Tomography_SOFC_E8332.txrm'
#bgfile = '/local/data/databank/TXM_NSLS/XrmTxrmExampleFiles/20120627_005_BackgroundforTomo_SOFC_E8332_Average.xrm'


##filename = '/local/data/databank/TXM_NSLS/2013_smallTomo/20130405_016_tomoTest_6565eV_256x256x181proj_noBKG.txrm'
##bgfile = '/local/data/databank/TXM_NSLS/2013_smallTomo/20130405_015_bkg_6565eV_256x256.xrm'
##
##save_dir = '/local/data/databank/TXM_NSLS/temp'
##
##filename = '/local/data/databank/TXM_NSLS/2012_largeTomo/20121015_002_SOFC_tomo_1kx1kx1441proj_noBKG.txrm'
##bgfile = '/local/data/databank/TXM_NSLS/2012_largeTomo/20121015_001_SOFC_bkg_1kx1k.xrm'
##
##filename = '/local/data/databank/TXM_NSLS/2011_largeTomo/20111101_007_SOFC_8400eV_tomo_1k1k1441proj_noBKG.txrm'
##bgfile = '/local/data/databank/TXM_NSLS/2011_largeTomo/20111101_006_SOFC_8400eV_bkg_1k1k.xrm'
##
##save_dir = '/local/data/databank/TXM_NSLS/temp2'

filename = '/local/data/databank/TXM_26ID/20130731_004_Stripe_Solder_Sample_Tip1_TomoScript_181imgs_p1s_b1.txrm'
bgfile = '/local/data/databank/TXM_26ID/20130731_001_Background_Reference_20imgs_p5s_b1.xrm'
save_dir = '/local/data/databank/TXM_26ID/temp'

HDF5 = '/local/data/databank/dataExchange/TXM/20130731_004_Stripe_Solder_Sample_Tip1.h5'

verbose = True

imgname='Image'

reader = xradia.xrm()
array = dstruct

if verbose: print "reading projections ... "

reader.read_txrm(filename,array)
nx, ny, nz_dt = np.shape(array.exchange.data)
if verbose: print "done reading ", nz_dt, " projections images of (", nx,"x", ny, ") pixels"

dt = np.zeros((nx,ny,nz_dt))
dt = array.exchange.data[:,:,:]

f = open(save_dir+'/configure.txt','w')
f.write('Data creation date: \n')
f.write(str(array.information.file_creation_datetime))
f.write('\n')
f.write('=======================================\n')
f.write('Sample name: \n')
f.write(str(array.information.sample.name))
f.write('\n')
f.write('=======================================\n')
f.write('Experimenter name: \n')
f.write(str(array.information.experimenter.name))
f.write('\n')
f.write('=======================================\n')
f.write('X-ray energy: \n')
f.write(str(array.exchange.energy))
f.write(str(array.exchange.energy_units))
f.write('\n')
f.write('=======================================\n')
f.write('nx, ny: \n')
f.write(str(dt.shape[0]))
f.write(', ')
f.write(str(dt.shape[1]))
f.write('\n')
f.write('=======================================\n')
f.write('Number of frames: \n')
f.write(str(dt.shape[2]))
f.write('\n')
f.write('=======================================\n')
f.write('Angles: \n')
f.write(str(array.exchange.angles))
f.write('\n')
f.write('=======================================\n')
f.write('Data type: \n')
f.write(str(dt.dtype))
f.write('\n')
f.write('=======================================\n')
f.write('Data axes: \n')
f.write(str(array.exchange.data_axes))
f.write('\n')
f.write('=======================================\n')
f.write('x distance: \n')
f.write(str(array.exchange.x))
f.write('\n')
f.write('=======================================\n')
f.write('x units: \n')
f.write(str(array.exchange.x_units))
f.write('\n')
f.write('=======================================\n')
f.write('y distance: \n')
f.write(str(array.exchange.y))
f.write('\n')
f.write('=======================================\n')
f.write('y units: \n')
f.write(str(array.exchange.y_units))
f.write('\n')
f.close()

n_angles = np.shape(array.exchange.angles)
if verbose: print "done reading ", n_angles, " angles"
angles = np.zeros(n_angles)
angles = array.exchange.angles[:]

if verbose: print "reading background ... "
reader.read_xrm(bgfile,array)
nx, ny, nz = np.shape(array.exchange.data)
bg = np.zeros((nx,ny,nz))
bg = array.exchange.data[:,:,:]
if verbose: print "done reading ", nz, " background image (s) of (", nx,"x", ny, ") pixels"

if verbose: print "reading dark fields ... "
## reader.read_xrm(dkfile,array)
nx, ny, nz = np.shape(array.exchange.data)
nz = 1
dk = np.zeros((nx,ny,nz))
## dk = array.exchange.data[:,:,:]
if verbose: print "done reading ", nz, " dark image (s) of (", nx,"x", ny, ") pixels"

#Write HDF5 file.

# Open DataExchange file
f = DataExchangeFile(HDF5, mode='w') 

# Create HDF5 subgroup
# /measurement/instrument
f.add_entry( DataExchangeEntry.instrument(name={'value': 'APS-CNM 26-ID'}) )

### Create HDF5 subgroup
### /measurement/instrument/source
f.add_entry( DataExchangeEntry.source(name={'value': "Advanced Photon Source"},
                                    date_time={'value': "2013-07-31T19:42:13+0100"},
                                    beamline={'value': "26-ID"},
                                    )
)

# Create HDF5 subgroup
# /measurement/instrument/monochromator
f.add_entry( DataExchangeEntry.monochromator(type={'value': 'Unknown'},
                                            energy={'value': float(array.exchange.energy[0]), 'units': 'keV', 'dataset_opts': {'dtype': 'd'}},
                                            mono_stripe={'value': 'Unknown'},
                                            )
    )

# Create HDF5 subgroup
# /measurement/experimenter
f.add_entry( DataExchangeEntry.experimenter(name={'value':"Robert Winarski"},
                                            role={'value':"Project PI"},
                )
    )

# Create HDF5 subgroup
# /measurement/sample
f.add_entry( DataExchangeEntry.sample( name={'value':'Stripe_Solder_Sample_Tip1'},
                                        description={'value':'data converted from txrm/xrm set'},
        )
    )

# Create core HDF5 dataset in exchange group for 180 deep stack
# of x,y images /exchange/data
f.add_entry( DataExchangeEntry.data(data={'value': dt, 'units':'counts', 'description': 'transmission', 'axes':'theta:y:x' }))
f.add_entry( DataExchangeEntry.data(theta={'value': angles, 'units':'degrees'}))
f.add_entry( DataExchangeEntry.data(data_dark={'value': dk, 'units':'counts', 'axes':'theta_dark:y:x' }))
f.add_entry( DataExchangeEntry.data(data_white={'value': bg, 'units':'counts', 'axes':'theta_white:y:x' }))
f.add_entry( DataExchangeEntry.data(title={'value': 'tomography_raw_projections'}))

f.close()
if verbose: print "Done converting ", filename

###if __name__ == "__main__":
###    main()

