
    import tomopy
    import dxchange
    import matplotlib.pyplot as plt

if __name__ == '__main__':
    fname = '../../../tomopy/data/tooth.h5'

    start = 0
    end = 2

    proj, flat, dark = dxchange.read_aps_32id(fname, sino=(start, end))

    plt.imshow(proj[:, 0, :], cmap='Greys_r')
    plt.show()

    theta = tomopy.angles(proj.shape[0])

    proj = tomopy.normalize(proj, flat, dark)

    rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)

    tomopy.minus_log(proj)

    recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec')
    recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
    plt.imshow(recon[0, :,:], cmap='Greys_r')
    plt.show()

