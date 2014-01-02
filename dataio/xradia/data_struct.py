# 
#   This file is part of Mantis, a Multivariate ANalysis Tool for Spectromicroscopy.
# 
#   Copyright (C) 2011 Mirna Lerotic, 2nd Look
#   http://2ndlook.co/products.html
#   License: GNU GPL v3
#
#   Mantis is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   any later version.
#
#   Mantis is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details <http://www.gnu.org/licenses/>.


#    This module is used to pass data between routines and is based on 
#    https://confluence.aps.anl.gov/display/NX/Data+Exchange+Basics


class Descr(object):

    def __get__(self, instance, owner):
        # Check if the value has been set
        if (not hasattr(self, "_value")):
            return None
        #print "Getting value: %s" % self._value
        return self._value

    def __set__(self, instance, value):
        #print "Setting to %s" % value
        self._value = value

    def __delete__(self, instance):
        del(self._value)


### Information HDF5 Group
#----------------------------------------------------------------------
class ids(object):

    proposal = Descr()   
    activity = Descr()
    esaf = Descr() 


#----------------------------------------------------------------------
class experimenter(object):

    name = Descr()   
    role = Descr()
    affiliation = Descr() 
    address = Descr()  
    phone = Descr()
    email = Descr()
    facility_user_id = Descr() 
    

#----------------------------------------------------------------------
class sample(object):
    
    name = Descr()
    description = Descr()
#   preparation_date [string - ISO 8601 format]        
    preparation_datetime = Descr()
#   chemical_formula [string - abbreviated CIF format]
    chemical_formula = Descr()    
    environment = Descr()
    temperature = Descr()
    temperature_units = Descr()
    pressure = Descr()
    pressure_units = Descr()
        
        
#----------------------------------------------------------------------
class objective(object):

    manufacturer = Descr()   
    model = Descr()
    comment = Descr() 
    magnification = Descr()
    

#----------------------------------------------------------------------
class scintillator(object):

    name = Descr()   
    type = Descr()
    comment = Descr() 
    scintillating_thickness = Descr()   
    scintillating_thickness_units = Descr()
    substrate_thickness = Descr()
    substrate_thickness_units = Descr()
    
    
#----------------------------------------------------------------------
class facility(object):

    name = Descr()   
    beamline = Descr()
    
    
#----------------------------------------------------------------------
class accelerator(object):
        
    ring_current = Descr() 
    ring_current_units = Descr()
    primary_beam_energy = Descr()   
    primary_beam_energy_units = Descr()
    monostripe = Descr()
    
    
#----------------------------------------------------------------------
class pixel_size(object):
           
    horizontal = Descr()     
    horizontal_units = Descr()      
    vertical = Descr()    
    vertical_units = Descr()      
           
#----------------------------------------------------------------------
class dimensions(object):
           
    horizontal = Descr()
    vertical = Descr()
           
#----------------------------------------------------------------------
class binning(object):
           
    horizontal = Descr()
    vertical = Descr()
           
#----------------------------------------------------------------------
class axis_directions(object):
           
    horizontal = Descr()     
    vertical = Descr()    
     
#----------------------------------------------------------------------
class roi(object):
           
    x1 = Descr()
    y1 = Descr()
    x2 = Descr()
    y2 = Descr() 
    
    
#----------------------------------------------------------------------
class detector(object):  
    
    manufacturer = Descr()     
    model = Descr()    
    serial_number = Descr()      
    bit_depth  = Descr()    
    operating_temperature  = Descr()      
    operating_temperature_units = Descr()     
    exposure_time  = Descr()     
    exposure_time_units = Descr()    
    frame_rate = Descr()
    
    pixel_size = pixel_size()
    dimensions = dimensions()
    binning = binning()
    axis_directions = axis_directions()
    roi = roi()

        
#----------------------------------------------------------------------        
class information(object):

    title = Descr()
    comment = Descr()
    file_creation_datetime = Descr()
    
    ids = ids()
    experimenter = experimenter() 
    sample = sample()
    objective = objective()
    scintillator = scintillator()
    facility = facility()
    accelerator = accelerator()
    detector = detector()
    
         

#Exchange HDF5 group  
 #----------------------------------------------------------------------
class exchange(object):

    title = Descr()
    comment = Descr()
    data_collection_datetime = Descr()
    
    #n-dimensional dataset 
    data = Descr()
    
    data_signal = Descr()
    data_description = Descr()
    data_units = Descr()
    data_axes = Descr()
    data_detector = Descr()
    
    #These are described in data attribute axes 'x:y:z' but can be arbitrary 
    x = Descr()
    x_units = Descr()
    y = Descr()
    y_units = Descr()
    z = Descr()
    z_units = Descr()
    
    energy = Descr()  
    energy_units = Descr()
    
    white_data = Descr()
    white_data_units = Descr()
    dark_data = Descr()
    dark_data_units = Descr()
    rotation = Descr()
    
    angles = Descr()

# Spectromicroscopy HDF5 Group        
#----------------------------------------------------------------------
class normalization(object):

    white_spectrum = Descr()
    white_spectrum_units = Descr()
    white_spectrum_energy = Descr()
    white_spectrum_energy_units = Descr()
               
               
#----------------------------------------------------------------------
class spectromicroscopy(object):

    positions = Descr()
    positions_units = Descr()
    positions_names = Descr()
        
    normalization = normalization()
        
    optical_density = Descr()
    
    data_dwell = Descr()
    i0_dwell = Descr()
        

# HDF5 Root Group
#----------------------------------------------------------------------
class h5(object):
    
    #implements [string] comma separated string that tells the user which entries file contains
    implements = Descr()
    version = Descr() 

    information = information()
    exchange = exchange()
    spectromicroscopy = spectromicroscopy()
        
