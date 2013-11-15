# -*- coding: utf-8 -*-
# Filename: main.py
""" Main program for tomographic reconstruction.
"""
from tomoRecon import tomoRecon
from visualize import image
from dataio import data, tiff

def main():

    # Read input HDF file.
    inputFile = '/local/dgursoy/data/Harrison_Aus_2013/A01_.h5'
    dataset = data.read(inputFile, slicesStart=700, slicesEnd=703)

    # Normalize dataset.
    dataset.normalize()

    # Apply ring removal.
    dataset.removeRings()

    # Apply median filter.
    dataset.medianFilter()

    # Find rotation center.
    #dataset.optimizeCenter(tol=0.1)
    dataset.center = 657.125

    # Retrieve phase.
    dataset.retrievePhasePaganin(pixelSize=1e-4, dist=70, energy=30, deltaOverMu=1e-8)

    # Initialize reconstruction parameters.
    recon = tomoRecon.tomoRecon(dataset)

    # Perform tomographic reconstruction.
    recon.run(dataset)

    # Export data.
    tiff.write(recon.data, outputFile='/local/dgursoy/data/test.tiff')

    # Visualize a single slice.
    image.showSlice(recon.data)

if __name__ == "__main__":
    main()
