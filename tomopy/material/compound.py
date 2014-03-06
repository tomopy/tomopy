# -*- coding: utf-8 -*-
"""
Module in construction!
"""
import re
import numpy as np
import h5py

# --------------------------------------------------------------------

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