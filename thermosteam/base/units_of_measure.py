# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
__all__ = ('chemical_units_of_measure', 
           'stream_units_of_measure',
           'ureg', 'get_dimensionality',
           'Units', 'convert')

# %% Import unit registry

from pint import UnitRegistry
from pint.quantity import to_units_container
import os

# Set pint Unit Registry
ureg = UnitRegistry()
ureg.default_format = '~P'
ureg.load_definitions(os.path.dirname(os.path.realpath(__file__)) + '/units_of_measure.txt')
convert = ureg.convert
del os, UnitRegistry

# %% Manage conversion factors

class Units:
    __slots__ = ('_units', '_units_container', '_dimensionality', '_factor_cache')
    _cache = {}
    def __new__(cls, units=""):
        cache = cls._cache
        if units in cache:
            return cache[units]
        else:
            self = super().__new__(cls)
            self._units = units
            self._units_container = to_units_container(units, ureg)
            self._dimensionality = get_dimensionality(self._units_container)
            self._factor_cache = {}
            cache[units] = self
            return self
    
    @property
    def units(self):
        return self._units
    
    @property
    def dimensionality(self):
        return self._dimensionality
    
    def conversion_factor(self, to_units):
        cache = self._factor_cache
        if to_units in cache:
            factor = cache[to_units]
        else:
            cache[to_units] = factor = ureg.convert(1., self._units_container, to_units)
        return factor
    
    def __bool__(self):
        return bool(self._units)
    
    def __str__(self):
        return self._units
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self._units)})"


def get_dimensionality(units, cache={}):
    if units in cache:
        dim = cache[units]
    else:
        cache[units] = dim = ureg._get_dimensionality(to_units_container(units, ureg))
    return dim
    

# %% Units of measure

chemical_units_of_measure = {'MW': Units('g/mol'),
                             'T': Units('K'),
                             'Tm': Units('K'),
                             'Tb': Units('K'),
                             'Tt': Units('K'),
                             'Tc': Units('K'),
                             'P': Units('Pa'),
                             'Pc': Units('Pa'),
                             'Psat': Units('Pa'),
                             'Pt': Units('K'),
                             'V': Units('m^3/mol'),
                             'Vc': Units('m^3/mol'),
                             'Cp': Units('J/mol/K'),
                             'rho': Units('kg/m^3'), 
                             'rhoc': Units('kg/m^3'),
                             'nu': Units('m^2/s'),
                             'mu': Units('Pa*s'),
                             'sigma': Units('N/m'),
                             'kappa': Units('W/m/K'),
                             'alpha': Units('m^2/s'), 
                             'Hvap': Units('J/mol'),
                             'H': Units('J/mol'),  
                             'Hf': Units('J/mol'), 
                             'Hc': Units('J/mol'), 
                             'Hfus': Units('J/mol'), 
                             'Hsub': Units('J/mol'),
                             'S': Units('J/mol'), 
                             'G': Units('J/mol'), 
                             'U': Units('J/mol'), 
                             'A': Units('J/mol'),
                             'H_excess': Units('J/mol'), 
                             'S_excess': Units('J/mol'),
                             'R': Units('J/mol/K'),
                             'delta': Units('Pa^0.5'),
                             'epsilon': Units(''),
}
stream_units_of_measure = dict(molar_flow=Units('kmol/hr'),
                               mass_flow=Units('kg/hr'),
                               volumetric_flow=Units('m^3/hr'),
                               cost=Units('USD/hr'),
                               mass=Units('kg/hr'),
                               mol=Units('kmol/hr'),
                               vol=Units('m^3/hr'),
                               massnet=Units('kg/hr'),
                               molnet=Units('kmol/hr'),
                               volnet=Units('m^3/hr'),
                               Hvap=Units('kJ/hr'),
                               Hf=Units('kJ/hr'), 
                               Hc=Units('kJ/hr'), 
                               H=Units('kJ/hr'),
                               S=Units('kJ/hr'),
                               G=Units('kJ/hr'),
                               U=Units('kJ/hr'),
                               A=Units('kJ/hr'),
                               C=Units('kJ/hr/K'),
)
for i in ('T', 'P', 'mu', 'V', 'rho', 'sigma', 'kappa', 'nu', 'epsilon', 'delta', 'Psat', 'Cp'):
    stream_units_of_measure[i] = chemical_units_of_measure[i]

definitions = {'MW': 'Molecular weight',
               'T': 'Temperature',
               'Tm': 'Melting point temperature',
               'Tb': 'Boiling point temperature',
               'Tt': 'Triple point temperature',
               'Tc': 'Critical point temperature',
               'P': 'Pressure',
               'Pc': 'Critical point pressure',
               'Psat': 'Saturated vapor pressure',
               'Pt': 'Triple point pressure',
               'V': 'Molar volume',
               'Vc': 'Critical point volume',
               'Cp': 'Molar heat capacity',
               'rho': 'Density',
               'rhoc': 'Critical point density',
               'nu': 'Kinematic viscosity',
               'mu': 'Hydrolic viscosity',
               'sigma': 'Surface tension',
               'kappa': 'Thermal conductivity',
               'alpha': 'Thermal diffusivity',
               'Hvap': 'Heat of vaporization',
               'H': 'Enthalpy',
               'Hf': 'Heat of formation',
               'Hc': 'Heat of combustion', 
               'Hfus': 'Heat of fusion',
               'Hsub': 'Heat of sublimation',
               'S': 'Entropy',
               'G': 'Gibbs free energy',
               'U': 'Internal energy',
               'A': 'Helmholtz energy',
               'H_excess': 'Excess enthalpy',
               'S_excess': 'Excess entropy',
               'R': 'Universal gas constant',
               'Zc': 'Critical compressibility',
               'omega': 'Acentric factor',
               'delta': 'Solubility parameter',
               'epsilon': 'Relative permittivity',
}

types = {}
# Synonyms
for i, j in [('ω', 'omega')]:
    definitions[i] = definitions[j]
    if j in chemical_units_of_measure: chemical_units_of_measure[i] = chemical_units_of_measure[j]

# Phase properties
for var in ('Cp', 'H', 'S', 'V', 'kappa', 'H_excess', 'S_excess'):
    units = chemical_units_of_measure[var]
    definition = definitions[var].lower()
    for tag, phase in zip(('s', 'l', 'g'), ('Solid ', 'Liquid ', 'Gas ')):
        phasevar = var + '.' +tag
        definitions[phasevar] = phase + definition
