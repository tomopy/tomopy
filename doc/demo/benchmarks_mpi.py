# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:08:19 2016

@author: lbluque
"""
from __future__ import division

import multiprocessing as mp
import os
import tomopy
import time
import numpy as np
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# import astra
# import sirtfbp

# mpi sinogram chunking overlap
mpi_sino_overlap = 2

datasets = {'20150820_130628_Dleucopodia_10458_pieceA_10x_z30mm.h5': 1155.75}
            #'20150820_124324_Dleucopodia_10458_pieceA_10x_z80mm_moreangles.h5': 1223,
            #'20140829_132953_euc60mesh_1M-KI_10percLoad_time0_5x_33p5keV_10cmSampDet.h5': 1334.85}
algorithms = ('gridrec',) # 'fbp')
sinos = (100,) # 10, 1)
cores = (mp.cpu_count(),) #range(mp.cpu_count(), 0, -4)


if rank == 0:
    f = open('benchmark_results.txt', 'a')
    start_time = time.time()

sino = sinos[0]
dataset = datasets.keys()[0]
core = 1
algorithm = algorithms[0]

if rank == 0:
    _tomo, _flats, _darks, floc = tomopy.read_als_832h5(dataset,
                                                     sino=(0, sino, 1))
    chunk_size = int(math.ceil(_tomo.shape[1] / size))
    for r, _index in enumerate(range(0, _tomo.shape[1], chunk_size)):
        slc = np.s_[_index:_index+chunk_size]        
        if r == 0:
            index = _index
            tomo = _tomo[:,slc,:]
            flats = _flats[:,slc,:]
            darks = _darks[:,slc,:]
        else:
            comm.send(_index, dest=r)
            for data in [_tomo, _flats, _darks]:            
                data_slc = data[:,slc,:]  
                data_slc = np.require(data_slc, np.float32, 'C')
                comm.send(data_slc.shape, dest=r)
                comm.Send(data_slc, dest=r)
            comm.send(floc, dest=r)
else:
    index = comm.recv(source=0)
    tomo_shape = comm.recv(source=0)
    tomo = tomopy.util.dtype.empty_shared_array(tomo_shape, dtype=np.float32)
    comm.Recv(tomo, source=0)
    flats_shape = comm.recv(source=0)
    flats = np.empty(flats_shape, dtype=np.float32)
    comm.Recv(flats, source=0)
    darks_shape = comm.recv(source=0)
    darks = np.empty(darks_shape, dtype=np.float32)
    comm.Recv(darks, source=0)
    floc = comm.recv(source=0)

theta = tomopy.angles(tomo.shape[0])
tomo = tomopy.normalize(tomo, flats, darks, ncore=core)
tomo = tomopy.remove_stripe_fw(tomo, ncore=core)
tomo = tomopy.init_tomo(tomo, False, False)
rec = tomopy.recon(tomo, theta, center=datasets[dataset],
                   algorithm=algorithm, emission=True,
                   sinogram_order=True,
                   ncore=core)
rec = tomopy.circ_mask(rec, 0)
outname = os.path.join('.', '{0}_{1}_slices_{2}_cores_{3}'.format(dataset.split('.')[0], str(algorithm), str(sino), str(core)), dataset.split('.')[0])
tomopy.write_tiff_stack(rec, fname=outname, start=index)
    

if rank == 0:
    end_time = time.time() - start_time
    result = 'Function: {0}, Number of images: {1}, Runtime (s): {2}\n\n'.format('recon', rec.shape[0], end_time)
    print result
    f.write(result)
    f.flush()
    f.close()
    


#for dataset in datasets:
#    f.write('*****************************************************************************************************\n')
#    f.write(dataset + '\n\n')
#    for algorithm in algorithms:
#        for sino in sinos:
#            for core in cores:
#                start_time = time.time()
#                tomo, flats, darks, floc = tomopy.read_als_832h5(dataset,
#                                                                 sino=(0, sino, 1))
#                end_time = time.time() - start_time
#                f.write('Function: {0}, Number of sinos: {1}, Runtime (s): {2}\n'.format('read', sino, end_time))
#                theta = tomopy.angles(tomo.shape[0])
#                tomo = tomopy.normalize(tomo, flats, darks, ncore=core)
#                end_time = time.time() - start_time - end_time 
#                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Runtime (s): {3}\n'.format('normalize', sino, core, end_time))
#                tomo = tomopy.remove_stripe_fw(tomo, ncore=core)
#                end_time = time.time() - start_time - end_time 
#                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Runtime (s): {3}\n'.format('stripe_fw', sino, core, end_time))
#                tomo = tomopy.init_tomo(tomo, False, False)
#                end_time = time.time() - start_time - end_time 
#                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Runtime (s): {3}\n'.format('init_tomo', sino, core, end_time))
#                rec = tomopy.recon(tomo, theta, center=datasets[dataset],
#                                   algorithm=algorithm, emission=True,
#								   sinogram_order=True,
#                                   ncore=core)
#                end_time = time.time() - start_time - end_time
#                rec = tomopy.circ_mask(rec, 0)
#                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Algorithm: {3}, Runtime (s): {4}\n'.format('recon', sino, core, algorithm, end_time))
#                outname = os.path.join('.', '{0}_{1}_slices_{2}_cores_{3}'.format(dataset.split('.')[0], str(algorithm), str(sino), str(core)), dataset.split('.')[0])
#                tomopy.write_tiff_stack(rec, fname=outname)
#                end_time = time.time() - start_time - end_time  
#                f.write('Function: {0}, Number of images: {1}, Runtime (s): {2}\n\n'.format('write', rec.shape[0], end_time))
#                f.flush()
#f.close()
