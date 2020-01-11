# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:02:53 2019

@author: yoelr
"""
__all__ = ('chemical_units_of_measure', 
           'stream_units_of_measure',
           'ureg', 'get_dimensionality',
           'DisplayUnits', 'Units', 'convert')

from ..exceptions import DimensionError

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


# %% Manage display units

class DisplayUnits:
    """Create a DisplayUnits object where default units for representation are stored."""
    def __init__(self, **display_units):
        dct = self.__dict__
        dct.update(display_units)
        dct['dims'] = {}
        list_keys = []
        for k, v in display_units.items():
            try: # Assume units is one string
                dims = getattr(ureg, v).dimensionality
            except:
                try: # Assume units are a list of possible units
                    dims = [getattr(ureg, i).dimensionality for i in v]
                    list_keys.append(k)
                except: # Assume the user uses value as an option, and ignores units
                    dims = v
            self.dims[k] = dims
        for k in list_keys:
            dct[k] = dct[k][0] # Default units is first in list
    
    def __setattr__(self, name, unit):
        if name not in self.__dict__:
            raise AttributeError(f"can't set display units for '{name}'")
        if isinstance(unit, str):
            name_dim = self.dims[name]
            unit_dim = getattr(ureg, unit).dimensionality
            if isinstance(name_dim, list):
                if unit_dim not in name_dim:
                    name_dim = [f"({i})" for i in name_dim]
                    raise DimensionError(f"dimensions for '{name}' must be either {', '.join(name_dim[:-1])} or {name_dim[-1]}; not ({unit_dim})")    
            else:
                if name_dim != unit_dim:
                    raise DimensionError(f"dimensions for '{name}' must be in ({name_dim}), not ({unit_dim})")
        object.__setattr__(self, name, unit)
            
    def __repr__(self):
        sig = ', '.join((f"{i}='{j}'" if isinstance(j, str) else f'{i}={j}') for i,j in self.__dict__.items() if i != 'dims')
        return f'{type(self).__name__}({sig})'


# %% Units of measure

chemical_units_of_measure = {'MW': Units('g/mol'),
                             'T': Units('K'),
                             'Tr': Units('K'),
                             'Tm': Units('K'),
                             'Tb': Units('K'),
                             'Tbr': Units('Pa'),
                             'Tt': Units('K'),
                             'Tc': Units('K'),
                             'P': Units('Pa'),
                             'Pr': Units('Pa'),
                             'Pc': Units('Pa'),
                             'Psat': Units('Pa'),
                             'Pt': Units('K'),
                             'V': Units('m^3/mol'),
                             'Vc': Units('m^3/mol'),
                             'Cp': Units('J/g/K'),
                             'Cn': Units('J/mol/K'),
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
stream_units_of_measure = {'mol': Units('kmol/hr'),
                           'mass': Units('kg/hr'),
                           'vol': Units('m^3/hr'),
                           'F_mass': Units('kg/hr'),
                           'F_mol': Units('kmol/hr'),
                           'F_vol': Units('m^3/hr'),
                           'cost': Units('USD/hr'),
                           'Hvap': Units('kJ/hr'),
                           'Hf': Units('kJ/hr'), 
                           'Hc': Units('kJ/hr'), 
                           'H': Units('kJ/hr'),
                           'S': Units('kJ/hr'),
                           'G': Units('kJ/hr'),
                           'U': Units('kJ/hr'),
                           'A': Units('kJ/hr'),
                           'C': Units('kJ/hr/K'),
}
for i in ('T', 'P', 'mu', 'V', 'rho', 'sigma', 'kappa', 'nu', 'epsilon', 'delta', 'Psat', 'Cp', 'Cn'):
    stream_units_of_measure[i] = chemical_units_of_measure[i]

definitions = {'MW': 'Molecular weight',
               'T': 'Temperature',
               'Tr': 'Reduced temperature',
               'Tm': 'Melting point temperature',
               'Tb': 'Boiling point temperature',
               'Tbr': 'Reduced boiling point temperature',
               'Tt': 'Triple point temperature',
               'Tc': 'Critical point temperature',
               'P': 'Pressure',
               'Pr': 'Reduced pressure',
               'Pc': 'Critical point pressure',
               'Psat': 'Saturated vapor pressure',
               'Pt': 'Triple point pressure',
               'V': 'Molar volume',
               'Vc': 'Critical point volume',
               'Cp': 'Specific heat capacity',
               'Cn': 'Molar heat capacity',
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
               'dZ': 'Change in compressibility factor',
               'omega': 'Acentric factor',
               'delta': 'Solubility parameter',
               'epsilon': 'Relative permittivity',
               'similarity_variable': 'Heat capacity similarity variable',
               'iscyclic_aliphatic': 'Whether a chemical is cyclic aliphatic',
}

types = {}
types['iscyclic_aliphatic'] = 'bool'

# Synonyms
for i, j in [('ω', 'omega')]:
    definitions[i] = definitions[j]
    if j in chemical_units_of_measure: chemical_units_of_measure[i] = chemical_units_of_measure[j]

# Phase properties
for var in ('Cn', 'H', 'S', 'V', 'kappa', 'H_excess', 'S_excess'):
    units = chemical_units_of_measure[var]
    definition = definitions[var].lower()
    for tag, phase in zip(('s', 'l', 'g'), ('Solid ', 'Liquid ', 'Gas ')):
        phasevar = var + '.' +tag
        definitions[phasevar] = phase + definition
