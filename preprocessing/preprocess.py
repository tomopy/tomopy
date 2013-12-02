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
    def normalize(self):
        self.data = normalize.normalize(self.data)

    def median_filter(self):
        self.data = median_filter.median_filter(self.data)

    def optimize_center(self):
        self.data = optimize_center.optimize_center(self.data)

    def diagnose_center(self):
        self.data = optimize_center.diagnose_center(self.data)

    def remove_rings(self):
        self.data = remove_rings.remove_rings(self.data)

    def correct_view(self):
        self.data = correct_view.correct_view(self.data)

    def retrieve_phase(self):
        self.data = retrieve_phase.retrieve_phase(self.data)
