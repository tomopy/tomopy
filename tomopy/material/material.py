# -*- coding: utf-8 -*-
import h5py
import re
from tomopy.tools import constants
import numpy as np
from scipy import interpolate


class Element():
    def __init__(self, element):
        self.element = element.lower()
        self._f = h5py.File('elements.h5', 'r')
        
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


class Compound():
    def __init__(self, compound, density):
        self.compound = compound
        self.density = density
    
    def mass_ratio(self):
        elem_name = {'Al':'aluminium',
                     'Be':'beryllium',
                     'Cd':'cadmium',
                     'Ca':'calcium',
                     'C':'carbon',
                     'Cs':'cesium',
                     'Cl':'chlorine',
                     'F':'fluorine',
                     'Au':'gold',
                     'H':'hydrogen',
                     'I':'iodine',
                     'Fe':'iron',
                     'Pb':'lead',
                     'Mg':'magnesium',
                     'N':'nitrogen',
                     'O':'oxygen',
                     'P':'phosphorus',
                     'K':'potassium',
                     'Si':'silicon',
                     'Na':'sodium',
                     'S':'sulphur',
                     'Zn':'zinc'}
        
        all_matches = re.findall('[A-Z][a-z]?|[0-9]+', self.compound)
        elems = re.findall('[A-Z][a-z]?', self.compound)
        quant = re.findall('\d+', self.compound)
        num_elems = len(elems)
        count_elems = np.ones(num_elems)
        
        count = 0
        num_counts = 0
        for each in all_matches:
            if re.match('\d+', each):
                count_elems[count-1] = quant[num_counts]
                num_counts += 1
                count -= 1
            count += 1
        
        _f = h5py.File('elements.h5', 'r')
        _atomic_weight = np.zeros(num_elems)
        for m in range(num_elems):
            _atomic_weight[m] = np.squeeze(_f[elem_name[elems[m]]+'/atomic_weight'][:])
        
        mass_ratio = np.zeros(num_elems)
        for m in range(num_elems):
            mass_ratio[m] = _atomic_weight[m] * count_elems[m] / np.dot(_atomic_weight, count_elems)
        return mass_ratio
    
    def atom_concentration(self, energy):
        pass
    
    def total_attenuation(self, energy):
        pass
        
    def photo_absorption(self, energy):
        pass
    
    def compton_scattering(self, energy):
        pass
    
    def refractive_index(self, energy):
        pass
    
    def electron_density(self, energy):
        pass










