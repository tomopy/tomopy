from simulate import detector
from simulate import source
import pylab
from mpl_toolkits.mplot3d import Axes3D

det = detector.Detector()
det.pixel_size(2)
det.num_pixels(10, 10)
detx, dety, detz = det.pixel_coords(dist=20, alpha=0, beta=0, gamma=0)

src = source.Source()
src.pixel_size(2)
src.num_pixels(10, 10)
srcx, srcy, srcz = src.pixel_coords(dist=-20, alpha=0, beta=0, gamma=0)

fig = pylab.figure()
ax = fig.gca(projection='3d', adjustable='box')
ax.scatter(detx, dety, detz)
ax.scatter(srcx, srcy, srcz, c='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pylab.grid()
ax.auto_scale_xyz([-20, 20], [-20, 20], [-20, 20])
pylab.show()
