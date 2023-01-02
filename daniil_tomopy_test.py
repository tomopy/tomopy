import numpy as np
import matplotlib.pyplot as plt
import tomopy

proj3d_data =  np.load('/home/algol/Documents/DEV/larix/data/sample13076_3D.npy')
print(proj3d_data.dtype)

plt.figure()
plt.imshow(proj3d_data[0,:,:], cmap="gray", vmin= 0.0, vmax=0.5)
plt.title('projection')
plt.show()
print(np.mean(proj3d_data.flatten()))

from tomopy.misc.corr import median_filter3d
import importlib
importlib.reload(tomopy.misc.corr)
from tomopy.misc.corr import median_filter3d

filtered = median_filter3d(proj3d_data, kernel_half_size=1, ncore=1)

#plt.figure()
#plt.imshow(filtered[0,:,:], cmap="gray")
#plt.title('filtered')
#plt.show()
print(np.mean(filtered.flatten()))