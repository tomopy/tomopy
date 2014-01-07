# -*- coding: utf-8 -*-
# Filename: main_convert_xradia.py
""" Main program for convert xradia txrm and xrm data into tiff and binary stack.
"""
import dataio.xradia.xradia_xrm as xradia
import dataio.xradia.data_stack_sim as dstack
import dataio.xradia.data_struct as dstruct

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

imgname='Image'

reader = xradia.xrm()
array = dstruct

print "reading projections ... "

reader.read_txrm(filename,array)
nx, ny, nz_dt = np.shape(array.exchange.data)
print "done reading ", nz_dt, " projections images of (", nx,"x", ny, ") pixels"


dt = np.zeros((nx,ny,nz_dt))
dt = array.exchange.data[:,:,:]

#plt.figure(1)
#plt.imshow(dt[:,:,0])


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

print "reading background ... "
reader.read_xrm(bgfile,array)
nx, ny, nz = np.shape(array.exchange.data)
bg = np.zeros((nx,ny,nz))
bg = array.exchange.data[:,:,:]
print "done reading ", nz, " background image (s) of (", nx,"x", ny, ") pixels"

#plt.figure(2)
#plt.imshow(bg[:,:,0])

index = np.where(bg == 0.)
bg[index] = 1.
for i in range(0,nz_dt):
    # dt[:,:,i] = dt[:,:,i] / bg[:,:,0]
    scipy.misc.imsave(save_dir+'/'+imgname+'_raw_'+str(i)+'.tif', dt[:,:,i])


for i in range(0,nz):
        scipy.misc.imsave(save_dir+'/'+imgname+'_bg_'+str(i)+'.tif', bg[:,:,i])

#plt.figure(3)
#plt.imshow(dt[:,:,0])

print "saving binary file ..."
f = open(save_dir+'/binary_array_' + str(nx) + 'x' + str(ny) + '.dat','w')
print "done saving " + 'binary_array_' + str(nx) + 'x' + str(ny) + '.dat'
f.write(dt)
f.close()

###if __name__ == "__main__":
###    main()

