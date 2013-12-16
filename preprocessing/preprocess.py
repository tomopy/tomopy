# -*- coding: utf-8 -*-
# Filename: preprocess.py
from dataio.data_read import Dataset
from preprocessing import correct_alignment
from preprocessing import correct_view
from preprocessing import filters
from preprocessing import normalize
from preprocessing import phase_retrieval
from preprocessing import ring_removal
from preprocessing import zoom


class Preprocess(Dataset):
    def correct_view(self, num_overlap_pixels=None, overwrite=True):
        print "Correcting field of view..."
        if overwrite is True:
            self.data = correct_view.correct_view(self.data, num_overlap_pixels=num_overlap_pixels)
        elif overwrite is False:
            return correct_view.correct_view(self.data, num_overlap_pixels=num_overlap_pixels)

    def diagnose_center(self, slice_no=None, center_start=None, center_end=None, center_step=None, overwrite=True):
        print "Diagnosing rotation center..."
        correct_alignment.diagnose_center(self.data, slice_no=slice_no, center_start=center_start, center_end=center_end, center_step=center_step)

    def median_filter(self, axis=1, size=(1, 3), overwrite=True):
        print "Applying median filter to data..."
        if overwrite is True:
            self.data = filters.median_filter(self.data, axis=axis, size=size)
        elif overwrite is False:
            return filters.median_filter(self.data, axis=axis, size=size)

    def normalize(self, cutoff=None, overwrite=True):
        print "Normalizing data..."
        if overwrite is True:
            self.data = normalize.normalize(self.data, self.white, cutoff=cutoff)
        elif overwrite is False:
            return normalize.normalize(self.data, self.white, cutoff=cutoff)

    def optimize_center(self, slice_no=None, center_init=None, hist_min=None, hist_max=None, tol=0.5, sigma=2, overwrite=True):
        print "Opimizing rotation center using Nelder-Mead method..."
        if overwrite is True:
            self.center = correct_alignment.optimize_center(self.data, slice_no=slice_no, center_init=center_init, hist_min=hist_min, hist_max=hist_max, tol=tol, sigma=sigma)
        elif overwrite is False:
            return correct_alignment.optimize_center(self.data, slice_no=slice_no, center_init=center_init, hist_min=hist_min, hist_max=hist_max, tol=tol, sigma=sigma)

    def register_to(self, data, axis=0, num=0):
        print "Registering..."
        return correct_alignment.register_translation(self.data, data.data, axis=axis, num=num)

    def remove_rings(self, level=6, wname='db10', sigma=2, overwrite=True):
        print "Removing rings..."
        if overwrite is True:
            self.data = ring_removal.dwtfft(self.data, level=level, wname=wname, sigma=sigma)
        elif overwrite is False:
            return ring_removal.dwtfft(self.data, level=level, wname=wname, sigma=sigma)

    def retrieve_phase(self, pixel_size, dist, energy, delta_over_mu=1e-8, overwrite=True):
        print "Retrieving phase..."
        if overwrite is True:
            self.data = phase_retrieval.single_material(self.data, pixel_size=pixel_size, dist=dist, energy=energy, delta_over_mu=delta_over_mu)
        elif overwrite is False:
            return phase_retrieval.single_material(self.data, pixel_size=pixel_size, dist=dist, energy=energy, delta_over_mu=delta_over_mu)

    def zinger_filter(self, cutoff=2, overwrite=True):
        print "Removing zingers..."
        if overwrite is True:
            self.data = filters.zinger_filter(self.data, cutoff=cutoff)
        elif overwrite is False:
            return filters.zinger_filter(self.data, cutoff=cutoff)

    def zoom(self, scale, axis=0, kind='bilinear', padding=True, overwrite=True):
        print "Zooming..."
        if overwrite is True:
            self.data = zoom.zoom(self.data, scale, axis=axis, kind=kind)
        elif overwrite is False:
            return zoom.zoom(self.data, scale, axis=axis, kind=kind)
