# -*- coding: utf-8 -*-
# Filename: preprocess.py
from dataio.data_read import Dataset
from preprocessing import normalize
from preprocessing import median_filter
from preprocessing import optimize_center
from preprocessing import retrieve_phase
from preprocessing import remove_rings
from preprocessing import correct_view

class Preprocess(Dataset):
    def normalize(self, overwrite=True):
        if overwrite is True:
            self.data = normalize.normalize(self.data, self.white)
        elif overwrite is False:
            return normalize.normalize(self.data, self.white)

    def median_filter(self, overwrite=True):
        if overwrite is True:
            self.data = median_filter.median_filter(self.data)
        elif overwrite is False:
            return median_filter.median_filter(self.data)

    def optimize_center(self, overwrite=True):
        if overwrite is True:
            self.data = optimize_center.optimize_center(self.data)
        elif overwrite is False:
            return optimize_center.optimize_center(self.data)

    def diagnose_center(self, overwrite=True):
        if overwrite is True:
            self.data = optimize_center.diagnose_center(self.data)
        elif overwrite is False:
            return optimize_center.diagnose_center(self.data)

    def remove_rings(self, overwrite=True):
        if overwrite is True:
            self.data = remove_rings.remove_rings(self.data)
        elif overwrite is False:
            return remove_rings.remove_rings(self.data)

    def correct_view(self, overwrite=True):
        if overwrite is True:
            self.data = correct_view.correct_view(self.data)
        elif overwrite is False:
            return correct_view.correct_view(self.data)

    def retrieve_phase(self, overwrite=True):
        if overwrite is True:
            self.data = retrieve_phase.retrieve_phase(self.data)
        elif overwrite is False:
            return retrieve_phase.retrieve_phase(self.data)
