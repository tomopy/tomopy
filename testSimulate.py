import numpy as np
from simulate import simulate_wrapper
from simulate import detector
from simulate import source
from simulate import phantom
import pylab
import scipy.io

# Load data.
mat = scipy.io.loadmat('/local/dgursoy/data/shepp_logan/shepp_logan_128.mat')
data = np.array(mat.values()[0][::-1], dtype='float32')
#data = np.ones((10, 10, 10), dtype='float32')

# Create phantom.
obj = phantom.Phantom()
obj.pixel_size(1)
obj.values(data)

# Create source.
src = source.Source()
src.pixel_size(1)
src.num_pixels(128, 128)
src.energy(21)
srcx, srcy, srcz = src.pixel_coords(gamma=0.3)

# Create detector.
det = detector.Detector()
det.pixel_size(1)
det.num_pixels(128, 128)
detx, dety, detz = det.pixel_coords(gamma=0.3)

# Solve forward problem.
d1 = simulate_wrapper.Simulate(src, det, obj)
proj1 = d1.calc3d(srcx, srcy, srcz, detx, dety, detz)
proj2 = d1.calc2d(srcx, srcy, detx, dety)

pylab.figure()
pylab.imshow(proj1, interpolation='none', cmap='gray')
pylab.show()

pylab.figure()
pylab.imshow(proj2, interpolation='none', cmap='gray')
pylab.show()
