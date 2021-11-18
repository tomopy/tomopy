#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module with abstraction from core tomopy functions. Includes classes TomoData 
and Recon, which store useful information about the projections and
reconstructions. Can use these classes for plotting. Written for use in 
Jupyter notebook found in doc/demo (TODO: add Jupyter notebook here)

"""

from __future__ import print_function

import logging
import numexpr as ne
import dxchange
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tomopy
import smtplib
import time
import os

from matplotlib import animation, rc, colors
from matplotlib.widgets import Slider


# ----------------------------- Class TomoData -------------------------#


class TomoData:
    def __init__(
        self,
        prjImgs=None,
        numX=None,
        numY=None,
        numTheta=None,
        filename=None,
        raw="Yes",
        theta=None,
        angleStart=-90,
        angleEnd=90,
        cbarRange=[0, 1],
        verboseImport="No",
        rotate="Y",
        correctionOptions=dict(),
    ):
        self.prjImgs = prjImgs
        self.numX = numX
        self.numY = numY
        self.numTheta = numTheta
        self.theta = theta
        self.verboseImport = verboseImport
        self.filename = filename
        logging.getLogger("dxchange").setLevel(logging.WARNING)
        if filename is not None and numTheta is None and rotate == "Y":
            self = self.importAndRotateTiff(filename)
        elif rotate == "N" and numTheta == None:
            self = self.importTiff(filename)
        if filename is not None and numTheta is not None and rotate == "Y":
            self = self.importAndRotateTiffStack(filename, numTheta=numTheta)
        elif rotate == "N" and numTheta is not None:
            self = self.importTiffStack(filename, numTheta=numTheta)
        if raw is not "Yes":
            self.numTheta, self.numY, self.numX = self.prjImgs.shape
        if self.theta == None and self.numTheta is not None:
            self.theta = tomopy.angles(self.numTheta, angleStart, angleEnd)
        self.cbarRange = cbarRange

    # --------------------------Import Functions--------------------------#

    def importTiff(self, filename):
        """
        Import tiff and create TomoData object.

        Parameters
        ----------
        filename : str
            Relative or absolute

        Returns
        -------
        self : TomoData
        """
        print(filename)
        if self.verboseImport == "Y" or self.verboseImport == "Yes":
            logging.getLogger("dxchange").setLevel(logging.INFO)
        prjImgs = dxchange.reader.read_tiff(filename)
        self.prjImgs = prjImgs.astype(np.float32)
        if len(self.prjImgs.shape) == 2:
            print("this only has two dimensions")
            self.prjImgs = self.prjImgs[np.newaxis, :, :]
        if len(prjImgs.shape) == 3:
            self.numTheta, self.numY, self.numX = prjImgs.shape
        return self

    def importAndRotateTiff(self, filename):
        """
        Import tiff and create TomoData object,
        and rotate it 90 degrees clockwise.

        Parameters
        ----------
        filename : str
            Relative or absolute

        Returns
        -------
        self : TomoData
        """
        if self.verboseImport == "Y" or self.verboseImport == "Yes":
            logging.getLogger("dxchange").setLevel(logging.INFO)
        prjImgs = dxchange.reader.read_tiff(filename)
        prjImgs = np.swapaxes(prjImgs, 1, 2)
        prjImgs = np.flip(prjImgs, 2)
        self.prjImgs = prjImgs.astype(np.float32)
        if len(self.prjImgs.shape) == 2:
            self.prjImgs = self.prjImgs[np.newaxis, :, :]
        if len(prjImgs.shape) == 3:
            self.numTheta, self.numY, self.numX = prjImgs.shape
        else:
            self.numY, self.numX = prjImgs.shape
        return self

    def importTiffStack(self, filename, numTheta=None):
        """
        Import tiff stack (lots of files in one folder).
        TODO: remove requirement to take numTheta as an argument

        Parameters
        ----------
        filename : str of first file
            Typically 0000, relative or absolute
        numTheta: int, required
            Total number of projection images taken.
        Returns
        -------
        self : TomoData
        """
        if self.verboseImport == "Y" or self.verboseImport == "Yes":
            logging.getLogger("dxchange").setLevel(logging.INFO)
        prjImgs = dxchange.reader.read_tiff_stack(filename, list(range(self.numTheta)))
        prjImgs = prjImgs.astype(np.float32)
        if len(self.prjImgs.shape) == 2:
            self.prjImgs = self.prjImgs[np.newaxis, :, :]
        self.prjImgs = prjImgs
        self.numTheta, self.numY, self.numX = prjImgs.shape
        return self

    def importAndRotateTiffStack(self, filename, numTheta=None):
        """
        Import tiff stack (lots of files in one folder) and rotate it 90
        degrees clockwise.
        TODO: remove requirement to take numTheta as an argument

        Parameters
        ----------
        filename : str of first file
            Typically 0000, relative or absolute
        numTheta: int, required

        Returns
        -------
        self : TomoData
        """
        if self.verboseImport == "Y" or self.verboseImport == "Yes":
            logging.getLogger("dxchange").setLevel(logging.INFO)
        prjImgs = dxchange.reader.read_tiff_stack(filename, list(range(self.numTheta)))
        prjImgs = np.swapaxes(prjImgs, 1, 2)
        prjImgs = np.flip(prjImgs, 2)
        prjImgs = prjImgs.astype(np.float32)
        if len(self.prjImgs.shape) == 2:
            self.prjImgs = self.prjImgs[np.newaxis, :, :]
        self.prjImgs = prjImgs
        self.numTheta, self.numY, self.numX = prjImgs.shape
        return self

    # --------------------------Plotting Functions----------------------#

    def plotProjectionImage(
        self, projectionNo=0, figSize=(8, 4), cmap="viridis", cmapRange=None
    ):
        """
        Plot a specific projection image.
        This has controls so that you can plot the image and set the correct
        color map range. The colormap range set here can be used to plot a
        movie using plotProjectionMovie.

        Sliders idea taken from: https://stackoverflow.com/questions/65040676/
        matplotlib-sliders-rescale-colorbar-when-used-to-change-clim

        Parameters
        ----------
        projectionNo : int
            Must be from 0 to the total number of projection images you took.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str
            Colormap of choice. You can choose from the ones on
            matplotlib.colors
        cmapRange : list with 2 entries, [0,1]
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        if len(self.prjImgs.shape) == 3:
            imgData = self.prjImgs[projectionNo, :, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        if cmapRange == None:
            c_min = np.min(imgData)  # the min and max range of the sliders
            c_max = np.max(imgData)
        else:
            c_min = cmapRange[0]
            c_max = cmapRange[1]

        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # define sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)

        plt.show()

    def plotSinogram(
        self, sinogramNo=0, figSize=(8, 4), cmap="viridis", cmapRange=None
    ):
        """
        Plot a sinogram given the sinogram number. The slice for one particular
        value of y pixel. This has controls so that you can plot the image and
        set the correct color map range. The colormap range set here can be
        used to plot a movie using plotProjectionMovie.

        Parameters
        ----------
        sinogramNo : int,
            Must be from 0 to the total number of Y pixels number of projection
            images you took.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str,
            Colormap of choice. You can choose from the ones on
            matplotlib.colors.
        cmapRange : list with 2 entries, [0,1]
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        if len(self.prjImgs.shape) == 3:
            imgData = self.prjImgs[:, sinogramNo, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        if cmapRange == None:
            c_min = np.min(imgData)  # the min and max range of the sliders
            c_max = np.max(imgData)
        else:
            c_min = cmapRange[0]
            c_max = cmapRange[1]
        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # define sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)

        plt.show()

    def plotProjectionMovie(
        self, startNo=0, endNo=None, skipFrames=1, saveMovie=None, figSize=(8, 3)
    ):
        """
        Exports an animation. Run the plotProjectionImage function first to
        determine colormap range.

        Parameters
        ----------
        startNo : int
            must be from 0 to the total number of Y pixels.
        endNo : int
            must be from startNo to the total number of Y pixels.
        skipFrames : int
            number of frames you would like to skip. Increase
            this value for large datasets. 
        saveMovie : 'Y', optional
            Saves movie to 'movie.mp4' (do not know if functional).
        figSize : (int, int)
            choose the figure size you want to pop out.

        Returns
        -------
        ani : matplotlib animation, can be used to plot in jupyter notebook
        using HTML(ani.to_jshtml())
        """
        frames = []
        if endNo == None:
            endNo = self.numTheta
        animSliceNos = range(startNo, endNo, skipFrames)
        fig, ax = plt.subplots(figsize=figSize)
        for i in animSliceNos:
            frames.append(
                [
                    ax.imshow(
                        self.prjImgs[i, :, :],
                        cmap="viridis",
                        vmin=self.cbarRange[0],
                        vmax=self.cbarRange[1],
                    )
                ]
            )
            ani = animation.ArtistAnimation(
                fig, frames, interval=50, blit=True, repeat_delay=100
            )
        if saveMovie == "Y":
            ani.save("movie.mp4")
        plt.close()
        return ani

    def writeTiff(self, fname):
        """
        Writes prjImgs attribute to file.

        Parameters
        ----------
        filename : str, relative or absolute filepath.

        """
        dxchange.write_tiff(self.prjImgs, fname=fname)

    # --------------------------Correction Functions--------------------------#

    def removeStripes(self, options):
        """
        Remove stripes from sinograms so that you end up with less ring
        artifacts in reconstruction.

        eg

        .. highlight:: python
        .. code-block:: python

            tomoCorr = tomo.removeStripes(
                options={
                    "remove_all_stripe": {
                        "snr": 3,
                        "la_size": 61,
                        "sm_size": 21,
                        "dim": 1,
                        "ncore": None,
                        "nchunk": None,
                    },
                    "remove_large_stripe": {
                        "snr": 3,
                        "size": 51,
                        "drop_ratio": 0.1,
                        "norm": True,
                        "ncore": None,
                        "nchunk": None,
                    },
                }
            )

        Parameters
        ----------
        options : nested dict

            The formatting here is important - the keys in the 0th level of
            the dictionary (i.e. 'remove_all_stripe') will call
            the tomopy.prep.stripe function with the same name. Its
            corresponding value are the options input into that function.
            The order of operations will proceed with the first dictionary
            key given, then the second, and so on...

        Returns
        -------
        self : tomoData
        """
        for key in options:
            if key == "remove_all_stripe":
                print("Performing ALL stripe removal.")
                self.prjImgs = tomopy.prep.stripe.remove_all_stripe(
                    self.prjImgs, **options[key]
                )
            if key == "remove_large_stripe":
                print("Performing LARGE stripe removal.")
                self.prjImgs = tomopy.prep.stripe.remove_large_stripe(
                    self.prjImgs, **options[key]
                )
        self.correctionOptions = options
        return self


############################# TomoDataCombined #############################


class TomoDataCombined:
    """
    Combines tomodata to normalize it. TODO: make this a function under 
    TomoData.
    """
    def __init__(self, tomo, flat, dark):
        self.tomo = tomo
        self.flat = flat
        self.dark = dark

    def normalize(self, rmZerosAndNans=True):
        """
        Normalizes the data with typical options for normalization. TODO: Needs
        more options. 
        TODO: add option to delete the previous objects from memory.

        Parameters
        ----------
        rmZerosAndNans : bool
            Remove the zeros and nans from normalized data.

        Returns
        -------
        tomoNorm : TomoData
            Normalized data in TomoData object
        tomoNormMLog : TomoData
            Normalized + -log() data in TomoData object
        """
        tomoNormPrjImgs = tomopy.normalize(
            self.tomo.prjImgs, self.flat.prjImgs, self.dark.prjImgs
        )
        tomoNorm = TomoData(prjImgs=tomoNormPrjImgs, raw="No")
        tomoNormMLogPrjImgs = tomopy.minus_log(tomoNormPrjImgs)
        tomoNormMLog = TomoData(prjImgs=tomoNormMLogPrjImgs, raw="No")
        if rmZerosAndNans == True:
            tomoNormMLog.prjImgs = tomopy.misc.corr.remove_nan(
                tomoNormMLog.prjImgs, val=0.0
            )
            tomoNormMLog.prjImgs[tomoNormMLog.prjImgs == np.inf] = 0
        return tomoNorm, tomoNormMLog


################################### Recon ###################################

class Recon:
    """
    Class for performing reconstructions.

    Parameters
    ----------
    tomo : TomoData object.
        Normalize the raw tomography data with the TomoData class. Then,
        initialize this class with a TomoData object.
    """

    def __init__(
        self,
        tomo,
        center=None,
        recon=None,
        numSlices=None,
        numX=None,
        numY=None,
        options=dict(),
        reconTime=dict(),
        prjRange=None,
        cbarRange=[0, 1],
    ):
        self.tomo = tomo  # tomodata object
        self.recon = recon
        self.center = tomo.numX / 2  # defaults to center of X array
        self.options = options
        self.numX = numX
        self.numY = numY
        self.numSlices = numSlices
        self.reconTime = reconTime
        if prjRange == None:
            self.prjRange = [i for i in range(self.tomo.numTheta)]
        else:
            self.prjRange = [i for i in range(prjRange[0], prjRange[1])]
        self.cbarRange = cbarRange

    # --------------------------Astra Reconstruction----------------------#

    def reconstructAstra(
        self,
        options={
            "proj_type": "cuda",
            "method": "SIRT_CUDA",
            "num_iter": 1,
            "ncore": 1,
            "extra_options": {"MinConstraint": 0},
        },
    ):
        """
        Reconstructs projection images after creating a Recon object.
        Uses the Astra toolbox to do the reconstruction.
        Puts the reconstructed dataset into self.recon.

        Parameters
        ----------
        options : dict
            Dictionary format typically used to specify options in tomopy, see
            LINK TO DESCRIPTION OF OPTIONS.

        """
        import astra

        os.environ["TOMOPY_PYTHON_THREADS"] = "1"
        if "center" in options:
            self.center = options["center"]
        else:
            options["center"] = self.center
        # Perform the reconstruction
        print(
            "Astra reconstruction beginning on projection images",
            self.prjRange[0],
            "to",
            self.prjRange[-1],
            ".",
        )
        print(
            "Running",
            str(options["method"]),
            "for",
            str(options["num_iter"]),
            "iterations.",
        )
        if "extra_options" in options:
            print("Extra options: " + str(options["extra_options"]))
        tic = time.perf_counter()
        self.recon = tomopy.recon(
            self.tomo.prjImgs[:, self.prjRange[0] : self.prjRange[-1] : 1, :],
            self.tomo.theta,
            algorithm=tomopy.astra,
            options=options,
        )
        toc = time.perf_counter()
        self.reconTime = {
            "seconds": tic - toc,
            "minutes": (tic - toc) / 60,
            "hours": (tic - toc) / 3600,
        }
        print(f"Finished reconstruction after {toc - tic:0.3f} seconds.")
        self.recon = tomopy.circ_mask(self.recon, 0)
        self.numSlices, self.numY, self.numX = self.recon.shape
        self.options = options

    # --------------------------Tomopy Reconstruction----------------------#

    def reconstructTomopy(
        self,
        options={
            "algorithm": "gridrec",
            "filter_name": "butterworth",
            "filter_par": [0.25, 2],
        },
    ):
        """
        Reconstructs projection images after creating a Recon object.
        Uses tomopy toolbox to do the reconstruction.

        Parameters
        ----------
        options : dict,
            Dictionary format typically used to specify options in tomopy, see
            LINK TO DESCRIPTION OF OPTIONS.

        """
        import os

        if "center" in options:
            self.center = options["center"]
        else:
            options["center"] = self.center
        # Beginning reconstruction on range specified.
        print(
            "Tomopy reconstruction beginning on projection images",
            self.prjRange[0],
            "to",
            self.prjRange[-1],
            ".",
        )
        print("Running", str(options["algorithm"]))
        print("Options:", str(options))
        os.environ["TOMOPY_PYTHON_THREADS"] = "40"
        tic = time.perf_counter()
        self.recon = tomopy.recon(
            self.tomo.prjImgs[:, self.prjRange[0] : self.prjRange[-1] : 1, :],
            self.tomo.theta,
            ncore=40,
            **options,
        )
        toc = time.perf_counter()
        self.reconTime = {
            "seconds": tic - toc,
            "minutes": (tic - toc) / 60,
            "hours": (tic - toc) / 3600,
        }
        print(f"Finished reconstruction after {toc - tic:0.3f} seconds.")
        self.recon = tomopy.circ_mask(self.recon, 0)
        self.numSlices, self.numY, self.numX = self.recon.shape
        self.options = options

    # --------------------------Plotting Section----------------------#

    def plotReconSlice(self, sliceNo=0, figSize=(8, 4), cmap="viridis"):
        """
        Plot a slice of the reconstructed data. TODO: take arguments to slice
        through X, Y, or Z.

        This has controls so that you can plot the image and set the correct
        color map range. The colormap range set here can be used to plot a
        movie using plotProjectionMovie.

        Sliders idea taken from: https://stackoverflow.com/questions/65040676/
        matplotlib-sliders-rescale-colorbar-when-used-to-change-clim

        Parameters
        ----------
        sliceNo : int
            Must be from 0 to Z dimension of reconstructed object.
        figSize : (int, int)
            Choose the figure size you want to pop out.
        cmap : str
            Colormap of choice. You can choose from the ones on
            matplotlib.colors
        cmapRange : list with 2 entries, [0,1]
            TODO: FEATURE NOT ON HERE.
            Changes the maximum and minimum values for the color range.
        """
        fig, ax = plt.subplots(figsize=figSize)
        plt.subplots_adjust(left=0.25, bottom=0.3)
        imgData = self.recon[sliceNo, :, :]
        plotImage = ax.imshow(imgData, cmap=cmap)
        cbar = plt.colorbar(plotImage)
        c_min = np.min(imgData)  # the min and max range of the sliders
        c_max = np.max(imgData)
        # positions sliders beneath plot
        ax_cmin = plt.axes([0.25, 0.1, 0.65, 0.03])
        ax_cmax = plt.axes([0.25, 0.15, 0.65, 0.03])
        # defines sliders
        s_cmin = Slider(ax_cmin, "min", c_min, c_max, valinit=c_min)
        s_cmax = Slider(ax_cmax, "max", c_min, c_max, valinit=c_max)

        def update(val):
            _cmin = s_cmin.val
            self.cbarRange[0] = _cmin
            _cmax = s_cmax.val
            self.cbarRange[1] = _cmax
            plotImage.set_clim([_cmin, _cmax])

        s_cmin.on_changed(update)
        s_cmax.on_changed(update)
        plt.show()

    def plotReconMovie(
        self, startNo=0, endNo=None, skipFrames=1, saveMovie=None, figSize=(4, 4)
    ):
        """
        Exports an animation. Run the plotReconSlice function first to
        determine colormap range.

        Parameters
        ----------
        startNo : int
            must be from 0 to the total number of Y pixels.
        endNo : int
            must be from startNo to the total number of Y pixels.
        skipFrames : int, optional
            number of frames you would like to skip. Increase
            this value for large datasets. 
        saveMovie : 'Y', optional
            Saves movie to 'movie.mp4' TODO: make naming file possible.
        figSize : (int, int)
            Choose the figure size you want to pop out.

        Returns
        -------
        ani : matplotlib animation,
            can be used to plot in jupyter notebook, using
            HTML(ani.to_jshtml())
            TODO: MAKE THIS ONE TASK SO THAT JUPYTER NOTEBOOK IS CLEANER
        """
        frames = []
        if endNo == None:
            endNo = self.numSlices
        animSliceNos = range(startNo, endNo, skipFrames)
        fig, ax = plt.subplots(figsize=figSize)
        for i in animSliceNos:
            frames.append(
                [
                    ax.imshow(
                        self.recon[i, :, :],
                        cmap="viridis",
                        vmin=self.cbarRange[0],
                        vmax=self.cbarRange[1],
                    )
                ]
            )
            ani = animation.ArtistAnimation(
                fig, frames, interval=50, blit=True, repeat_delay=100
            )
        if saveMovie == "Y":
            ani.save("movie.mp4")
        plt.close()
        return ani


################################### Misc. Functions ###########################


def textme(phoneNumber, carrierEmail, gmail_user, gmail_password):
    """
    From https://stackabuse.com/how-to-send-emails-with-gmail-using-python/.

    Sends a text message to you when called.

    Parameters
    ----------
    phoneNumber : str
    carrierEmail : this is your carrier email. TODO, find list of these, and
        allow input of just the carrier. Idea from TXMWizard software.
    gmail_user : str, gmail username
    gmail_password : str, gmail password
    """

    toaddr = str(phoneNumber + "@" + carrierEmail)
    fromaddr = gmail_user
    message_subject = "Job done."
    message_text = "Finished the job."
    message = (
        "From: %s\r\n" % fromaddr
        + "To: %s\r\n" % toaddr
        + "Subject: %s\r\n" % message_subject
        + "\r\n"
        + message_text
    )
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(fromaddr, toaddr, message)
        server.close()
    except:
        print("Something went wrong...")
