# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('chemical_units_of_measure', 
           'stream_units_of_measure',
           'ureg', 'get_dimensionality',
           'DisplayUnits', 'AbsoluteUnitsOfMeasure', 'convert',
           'Quantity', 'format_plot_units')

from .exceptions import DimensionError

# %% Import unit registry

from pint import UnitRegistry
from pint.quantity import to_units_container
import os

# Set pint Unit Registry
ureg = UnitRegistry()
ureg.default_format = '~P'
ureg.load_definitions(os.path.dirname(os.path.realpath(__file__)) + '/units_of_measure.txt')
convert = ureg.convert
Quantity = ureg.Quantity
del os, UnitRegistry

# %% Functions

def format_degrees(units):
    if units.startswith('deg'):
        units = '^\circ ' + units[3:]
    return units

def format_units(units, isnumerator=True):
    if '^' in units:
        units, power = units.split('^')
        units = format_degrees(units)
        units = '\mathrm{' + units + '}'
        units += '^{' + (power if isnumerator else -power) + '}'
    else:
        units = format_degrees(units)
        units = '\mathrm{' + units + '}'
    return units

def format_plot_units(units):
    units = str(units)
    all_numerators = []
    all_denominators = []
    term_is_numerator = True
    for unprocessed_denominators in units.split("/"):
        term, *numerators = unprocessed_denominators.split("*")
        if term_is_numerator:
            numerators.append(term)
        else:
            all_denominators.append(term)
        term_is_numerator = not term_is_numerator
        all_numerators.extend(numerators)
    all_numerators = [format_units(i) for i in all_numerators]
    all_denominators = [format_units(i, False) for i in all_denominators]
    return '$' + ' \cdot '.join(all_numerators + all_denominators) + '$'

def get_dimensionality(units, cache={}):
    if units in cache:
        dim = cache[units]
    else:
        cache[units] = dim = ureg._get_dimensionality(to_units_container(units, ureg))
    return dim

# %% Manage conversion factors

class UnitsOfMeasure:
    __slots__ = ('_units', '_units_container', '_dimensionality')
    
    @property
    def units(self):
        return self._units
    
    @property
    def dimensionality(self):
        return self._dimensionality
    
    def __bool__(self):
        return bool(self._units)
    
    def __str__(self):
        return self._units
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self._units)})"

class AbsoluteUnitsOfMeasure(UnitsOfMeasure):
    __slots__ = ('_factor_cache',)
    _cache = {}
    
    def __new__(cls, units):
        if isinstance(units, cls):
            return units
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
    
    def conversion_factor(self, to_units):
        cache = self._factor_cache
        if to_units in cache:
            factor = cache[to_units]
        else:
            cache[to_units] = factor = ureg.convert(1., self._units_container, to_units)
        return factor
    
    def convert(self, value, to_units):
        return value * self.conversion_factor(to_units)
    
    def unconvert(self, value, to_units):
        return value / self.conversion_factor(to_units)


class RelativeUnitsOfMeasure(UnitsOfMeasure):
    __slots__ = ()
    _cache = {}
    
    def __new__(cls, units):
        if isinstance(units, cls):
            return units
        cache = cls._cache
        if units in cache:
            return cache[units]
        else:
            self = super().__new__(cls)
            self._units = units
            self._units_container = to_units_container(units, ureg)
            self._dimensionality = get_dimensionality(self._units_container)
            cache[units] = self
            return self
    
    def conversion_factor(self, to_units):
        return ureg.convert(1., self._units_container, to_units)
    
    def convert(self, value, to_units):
        return ureg.convert(value, self._units_container, to_units)
    
    def unconvert(self, value, to_units):
        return ureg.convert(value, to_units, self._units_container)


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

chemical_units_of_measure = {'MW': AbsoluteUnitsOfMeasure('g/mol'),
                             'T': RelativeUnitsOfMeasure('K'),
                             'Tr': RelativeUnitsOfMeasure('K'),
                             'Tm': RelativeUnitsOfMeasure('K'),
                             'Tb': RelativeUnitsOfMeasure('K'),
                             'Tbr': RelativeUnitsOfMeasure('K'),
                             'Tt': RelativeUnitsOfMeasure('K'),
                             'Tc': RelativeUnitsOfMeasure('K'),
                             'P': AbsoluteUnitsOfMeasure('Pa'),
                             'Pr': AbsoluteUnitsOfMeasure('Pa'),
                             'Pc': AbsoluteUnitsOfMeasure('Pa'),
                             'Psat': AbsoluteUnitsOfMeasure('Pa'),
                             'Pt': AbsoluteUnitsOfMeasure('Pa'),
                             'V': AbsoluteUnitsOfMeasure('m^3/mol'),
                             'Vc': AbsoluteUnitsOfMeasure('m^3/mol'),
                             'Cp': AbsoluteUnitsOfMeasure('J/g/K'),
                             'Cn': AbsoluteUnitsOfMeasure('J/mol/K'),
                             'R': AbsoluteUnitsOfMeasure('J/mol/K'),
                             'rho': AbsoluteUnitsOfMeasure('kg/m^3'),
                             'rhoc': AbsoluteUnitsOfMeasure('kg/m^3'),
                             'nu': AbsoluteUnitsOfMeasure('m^2/s'),
                             'alpha': AbsoluteUnitsOfMeasure('m^2/s'),
                             'mu': AbsoluteUnitsOfMeasure('Pa*s'),
                             'sigma': AbsoluteUnitsOfMeasure('N/m'),
                             'kappa': AbsoluteUnitsOfMeasure('W/m/K'),
                             'Hvap': AbsoluteUnitsOfMeasure('J/mol'),
                             'H': AbsoluteUnitsOfMeasure('J/mol'),  
                             'Hf': AbsoluteUnitsOfMeasure('J/mol'), 
                             'Hc': AbsoluteUnitsOfMeasure('J/mol'), 
                             'Hfus': AbsoluteUnitsOfMeasure('J/mol'), 
                             'Hsub': AbsoluteUnitsOfMeasure('J/mol'),
                             'HHV': AbsoluteUnitsOfMeasure('J/mol'),
                             'LHV': AbsoluteUnitsOfMeasure('J/mol'),
                             'S': AbsoluteUnitsOfMeasure('J/mol'), 
                             'G': AbsoluteUnitsOfMeasure('J/mol'), 
                             'U': AbsoluteUnitsOfMeasure('J/mol'), 
                             'A': AbsoluteUnitsOfMeasure('J/mol'),
                             'H_excess': AbsoluteUnitsOfMeasure('J/mol'), 
                             'S_excess': AbsoluteUnitsOfMeasure('J/mol'),
                             'dipole': AbsoluteUnitsOfMeasure('Debye'),
                             'delta': AbsoluteUnitsOfMeasure('Pa^0.5'),
                             'epsilon': AbsoluteUnitsOfMeasure(''),
}
stream_units_of_measure = {'mol': AbsoluteUnitsOfMeasure('kmol/hr'),
                           'mass': AbsoluteUnitsOfMeasure('kg/hr'),
                           'vol': AbsoluteUnitsOfMeasure('m^3/hr'),
                           'F_mass': AbsoluteUnitsOfMeasure('kg/hr'),
                           'F_mol': AbsoluteUnitsOfMeasure('kmol/hr'),
                           'F_vol': AbsoluteUnitsOfMeasure('m^3/hr'),
                           'cost': AbsoluteUnitsOfMeasure('USD/hr'),
                           'Hvap': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'Hf': AbsoluteUnitsOfMeasure('kJ/hr'), 
                           'Hc': AbsoluteUnitsOfMeasure('kJ/hr'), 
                           'H': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'S': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'G': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'U': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'A': AbsoluteUnitsOfMeasure('kJ/hr'),
                           'C': AbsoluteUnitsOfMeasure('kJ/hr/K'),
}
for i in ('T', 'P', 'mu', 'V', 'rho', 'sigma',
          'kappa', 'nu', 'epsilon', 'delta',
          'Psat', 'Cp', 'Cn', 'alpha'):
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
               'mu': 'hydraulic viscosity',
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
               'dipole': 'Dipole momment',
               'delta': 'Solubility parameter',
               'epsilon': 'Relative permittivity',
               'similarity_variable': 'Heat capacity similarity variable',
               'iscyclic_aliphatic': 'Whether a chemical is cyclic aliphatic',
               'has_hydroxyl': 'Whether a polar chemical has hydroxyl groups',
               'atoms': 'Atom-count pairs'
}

types = {'atoms': 'Dict[str, int]'}
types['iscyclic_aliphatic'] = types['has_hydroxy'] = 'bool'

# Synonyms
definitions['ω'] = definitions['omega']

# Phase properties
for var in ('mu', 'Cn', 'H', 'S', 'V', 'kappa', 'H_excess', 'S_excess'):
    definition = definitions[var].lower()
    for tag, phase in zip(('s', 'l', 'g'), ('Solid ', 'Liquid ', 'Gas ')):
        phase_var = var + '.' + tag
        phase_var2 = var + '_' + tag
        definitions[phase_var] = definitions[phase_var2] = phase + definition
