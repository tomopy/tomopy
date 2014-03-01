# -*- coding: utf-8 -*-
"""
Module in construction!
"""
import h5py
import re
from tomopy.tools import constants
import numpy as np
from scipy import interpolate


class Element():
    def __init__(self, element):
        self.element = element.lower()
        self._f = h5py.File('/local/dgursoy/Projects/tomopy/tomopy/material/elements.h5', 'r')
        
    def atomic_number(self):
        _atomic_number = np.squeeze(self._f[self.element+'/atomic_number'][:])
        return _atomic_number

    def atomic_weight(self):
        _atomic_weight = np.squeeze(self._f[self.element+'/atomic_weight'][:])
        return _atomic_weight
        
    def total_attenuation(self, energy):
        _attenuation = self._f[self.element+'/attenuation'][:]
        kev = _attenuation[:, 0]
        val = _attenuation[:, 1]
        f = interpolate.interp1d(kev, val)
        return f(energy)
    
    def form_factor_imag(self, energy):
        _form_factor_imag = self._f[self.element+'/form_factor_imag'][:]
        kev = _form_factor_imag[:, 0]
        val = _form_factor_imag[:, 1]
        f = interpolate.interp1d(kev, val)
        return f(energy)
    
    def form_factor_real(self, energy):
        _form_factor_real = self._f[self.element+'/form_factor_real'][:]
        kev = _form_factor_real[:, 0]
        val = _form_factor_real[:, 1]
        f = interpolate.interp1d(kev, val)
        return f(energy)
    
    def nominal_density(self):
        _nominal_density = np.squeeze(self._f[self.element+'/nominal_density'][:])
        return _nominal_density
    
    def relativistic_corr(self):
        _relativistic_corr = np.squeeze(self._f[self.element+'/relativistic_corr'][:])
        return _relativistic_corr
    
    def thompson_corr(self):
        _thompson_corr = np.squeeze(self._f[self.element+'/thompson_corr'][:])
        return _thompson_corr