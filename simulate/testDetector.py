import numpy as np
import dataSimWrapper
import pylab
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

numProj = 9
geo = dataSimWrapper.AreaDetector(resolution=(5, 5), pixelSize=1)
alpha, beta, gamma = geo.getAngles(phi=0.3, numProj=numProj)

alpha=0.055669240384
beta=0.396107227497
gamma=0.0698131700798

fig = pylab.figure()
ax = fig.gca(projection='3d', adjustable='box')
for m in range(numProj):
    detx, dety, detz = geo.getPixelCoords(dist=20, alpha=alpha, beta=beta, gamma=gamma)
    srcx, srcy, srcz = geo.getPixelCoords(dist=-20, alpha=alpha, beta=beta, gamma=gamma)
    ax.scatter(detx, dety, detz)
    ax.scatter(srcx, srcy, srcz, c='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
pylab.grid()
ax.auto_scale_xyz([-20, 20], [-20, 20], [-20, 20])
pylab.show()
