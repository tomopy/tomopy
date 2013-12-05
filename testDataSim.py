import numpy as np
from simulate import dataSim_wrapper
from simulate import detector
import pylab
import time
import scipy.io

# Define object.
mat = scipy.io.loadmat('/local/dgursoy/data/shepplogan3d128.mat')
obj_vals = np.array(mat.values()[0][::-1], dtype='float32')
obj_pixel_size = np.array(1, dtype='float32')

pylab.ion()
num_proj = 18
d1 = dataSim_wrapper.DataSim(obj_vals, obj_pixel_size)
geo = detector.AreaDetector(resolution=(64, 64), pixel_size=1)
alpha, beta, gamma = geo.get_angles(phi=np.pi/3, num_proj=num_proj)
for m in range(num_proj):
    detx, dety, detz = geo.getPixelCoords(dist=1e4, alpha=alpha[m], beta=beta[m], gamma=gamma[m])
    srcx, srcy, srcz = geo.getPixelCoords(dist=-1e4, alpha=alpha[m], beta=beta[m], gamma=gamma[m])

    t = time.time()
    proj = d1.calc(srcx, srcy, srcz, detx, dety, detz)
    print time.time() - t
    pylab.imshow(proj, interpolation='none', cmap='gray')
    pylab.draw()




#from geometry import grid3d
#from geometry import transform3d
#grid = grid3d.Plane(num_pixels=(64, 64), limits=[-128, 128, -128, 128])
#src = transform3d.translate(grid.pixel_coords(), amount=(0, 100, 0))
#det = transform3d.translate(grid.pixel_coords(), amount=(0, -100, 0))
