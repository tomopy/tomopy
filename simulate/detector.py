import numpy as np

class AreaDetector():
    def __init__(self, resolution, pixel_size):
        """ Area detector constructor.
        """
        self.resolution = resolution
        self.pixel_size = float(pixel_size)

    def getPixelCoords(self, dist=1e12, alpha=0, beta=0, gamma=0):
        """ Positions the detector in space and
        returns the pixel coordinates.
        """
        lenx = self.resolution[0] * self.pixel_size
        leny = self.resolution[1] * self.pixel_size
        yi = np.arange(-(lenx - self.pixel_size)/2,
                        (lenx - self.pixel_size)/2 + self.pixel_size,
                        self.pixel_size)
        zi = np.arange(-(leny - self.pixel_size)/2,
                        (leny - self.pixel_size)/2 + self.pixel_size,
                        self.pixel_size)
        y0, z0 = np.meshgrid(yi, zi)
        x0 = dist * np.ones(self.resolution[::-1])

        s1 = np.sin(alpha)
        s2 = np.sin(beta)
        s3 = np.sin(gamma)
        c1 = np.cos(alpha)
        c2 = np.cos(beta)
        c3 = np.cos(gamma)

        x1 = x0
        y1 = y0 * np.cos(alpha) + z0 * np.sin(alpha)
        z1 = y0 * np.sin(alpha) - z0 * np.cos(alpha)

        x2 = -x1 * np.cos(beta) + z1 * np.sin(beta)
        y2 = y1
        z2 = x1 * np.sin(beta) + z1 * np.cos(beta)

        coordx = -x2 * np.cos(gamma) + y2 * np.sin(gamma)
        coordy = x2 * np.sin(gamma) +  y2 * np.cos(gamma)
        coordz = z2

        coordx = coordx.astype('float32')
        coordy = coordy.astype('float32')
        coordz = coordz.astype('float32')
        return coordx, coordy, coordz

    def get_angles(self, phi, num_proj):
        """ Calculates the detector-source pair
        alignments from object alignment parameters.
        """
        gamma = np.arange(0, np.pi, np.pi / num_proj)
        alpha = phi * np.sin(2 * gamma)
        beta = phi * np.cos(2 * gamma)
        return alpha, beta, gamma
