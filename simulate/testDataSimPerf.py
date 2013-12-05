import numpy as np
import dataSimWrapper
import scipy.io
import time

# Define object.
#objVals = np.zeros((100, 100, 100), dtype='float32')
mat = scipy.io.loadmat('/local/dgursoy/data/shepplogan3d128.mat')
objVals = np.array(mat.values()[0], dtype='float32')
objPixelSize = np.array(10, dtype='float32')

srcx = np.array(1e4, dtype='float32')
srcy = np.array(10, dtype='float32')
srcz = np.array(0, dtype='float32')
detx = np.array(-1e4, dtype='float32')
dety = np.array(0, dtype='float32')
detz = np.array(0, dtype='float32')

d1 = dataSimWrapper.DataSim(objVals, objPixelSize)
t = time.time()
d1.calc(srcx, srcy, srcz, detx, dety, detz)
print time.time() - t
