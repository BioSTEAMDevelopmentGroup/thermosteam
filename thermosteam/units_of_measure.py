# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
__all__ = ('chemical_units_of_measure', 
           'stream_units_of_measure',
           'ureg', 'get_dimensionality',
           'DisplayUnits', 
           'DisplayNotation',
           'AbsoluteUnitsOfMeasure', 
           'convert',
           'Quantity', 
           'parse_units_notation',
           'format_units', 
           'format_plot_units',
           'reformat_units')

from .exceptions import DimensionError

# %% Import unit registry

import pint
import os

# Set pint Unit Registry
appreg = pint.get_application_registry()
ureg = appreg.get()
ureg.default_format = '~P'
ureg._on_redefinition = 'warn' # Avoid breaking packages dependent on thermosteam (or biosteam)
if not getattr(pint, 'BioSTEAM_units_loaded', False): # Avoid reloading units of measure in pint if thermosteam module is reloaded
    ureg.load_definitions(os.path.dirname(os.path.realpath(__file__)) + '/units_of_measure.txt')
    
convert = ureg.convert
Quantity = ureg.Quantity
del os

# %% Functions

def parse_units_notation(value):
    if value is None:
        units = None
        notation = None
    elif ':' in value:
        units, notation = value.split(':')
        if not units: units = None
    else:
        units = value
        notation = None
    return units, notation

def format_degrees(units):
    r"""
    Format units of measure to have a latex degree symbol, if needed.

    Examples
    --------
    >>> format_degrees('degC')
    '^\\circ C'
    
    """
    if units.startswith('deg'):
        units = r'^\circ ' + units[3:]
    return units

def format_units_power(units, isnumerator=True, mathrm=True):
    r"""
    Format units of measure power sign to have a latex friendly format.

    Examples
    --------
    >>> format_units_power('m^3')
    '\\mathrm{m}^{3}'
    >>> format_units_power('m^3', isnumerator=False)
    '\\mathrm{m}^{-3}'
    
    """
    if '^' in units:
        units, power = units.split('^')
        units = format_degrees(units)
        if mathrm: units = '\mathrm{' + units + '}'
        power, *other = power.split(' ', 1)
        units += '^{' + (power if isnumerator else '-' + power) + '}'
        if other: units += other[0]
    else:
        if units != '%':
            units = format_degrees(units)      
            if mathrm: units = '\mathrm{' + units + '}'
        if not isnumerator:
            units = units + '^{-1}'
    return units

def format_units(units, ends='$', mathrm=True):
    r"""
    Format units of measure to have a latex friendly format.

    Examples
    --------
    >>> format_units('USD/m^3')
    '$\\mathrm{USD} \\cdot \\mathrm{m}^{-3}$'
    
    >>> format_units('USD/MT')
    '$\\mathrm{USD} \\cdot \\mathrm{MT}^{-1}$'
    
    """
    units = str(units)
    all_numerators = []
    all_denominators = []
    unprocessed_numerators, *unprocessed_denominators = units.split("/")
    all_numerators = unprocessed_numerators.split("*")
    for unprocessed_denominator in unprocessed_denominators:
        denominator, *unprocessed_numerators = unprocessed_denominator.split("*")
        all_numerators.extend(unprocessed_numerators)
        all_denominators.append(denominator)
    all_numerators = [format_units_power(i, True, mathrm) for i in all_numerators if i != '1']
    all_denominators = [format_units_power(i, False, mathrm) for i in all_denominators if i != '1']
    return ends + ' \cdot '.join(all_numerators + all_denominators).replace('$', '\$').replace('%', '\%') + ends

def reformat_units(name):
    left, right = name.split('[')
    units, right = right.split(']')
    return f"{left} [{format_units(units)}]{right}"

format_plot_units = format_units
get_dimensionality = ureg.get_dimensionality

# %% Manage conversion factors

class UnitsOfMeasure:
    __slots__ = ('units', 'units_container', 'dimensionality')
    
    def __bool__(self):
        return bool(self.units)
    
    def __str__(self):
        return self.units
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.units)})"


class AbsoluteUnitsOfMeasure(UnitsOfMeasure):
    __slots__ = ('factor_cache',)
    _cache = {}
    
    def __new__(cls, units):
        if isinstance(units, cls):
            return units
        cache = cls._cache
        if units in cache:
            return cache[units]
        else:
            self = super().__new__(cls)
            self.units = units
            self.units_container = ureg[units]
            self.dimensionality = self.units_container.dimensionality
            self.factor_cache = {}
            cache[units] = self
            return self
    
    def conversion_factor(self, to_units):
        cache = self.factor_cache
        if to_units in cache:
            factor = cache[to_units]
        else:
            cache[to_units] = factor = ureg.convert(1., self.units_container, to_units)
        return factor
    
    def convert(self, value, to_units):
        return value * self.conversion_factor(to_units)
    
    def unconvert(self, value, from_units):
        return value / self.conversion_factor(from_units)


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
            self.units = units
            self.units_container = ureg.Unit(units)
            self.dimensionality = self.units_container.dimensionality
            cache[units] = self
            return self
    
    def conversion_factor(self, to_units):
        return ureg.convert(1., self.units_container, to_units)
    
    def convert(self, value, to_units):
        return ureg.convert(value, self.units_container, to_units)
    
    def unconvert(self, value, from_units):
        return ureg.convert(value, from_units, self.units_container)


# %% Manage display units

class DisplayNotation:
    """
    Create a DisplayNotation object where default units for representation are stored.
    
    Examples
    --------
    Its possible to change the default units of measure for the Stream show method:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water'], cache=True)
    >>> tmo.Stream.display_notation.flow = '.2g'
    >>> stream = tmo.Stream('stream', Water=1.324, units='kg/hr')
    >>> stream.show()
    Stream: stream
    phase: 'l', T: 298.15 K, P: 101325 Pa
    flow (kmol/hr): Water  0.073
    
    >>> # Change back to kmol/hr
    >>> tmo.Stream.display_notation.flow = '.3g'
    
    """
    __slots__ = ('T', 'P', 'flow')
    
    def __init__(self, T, P, flow):
        self.T = T
        self.P = P
        self.flow = flow

    def __repr__(self):
        sig = ', '.join([f"{i}={getattr(self, i)!r}'" for i in self.__slots__])
        return f'{type(self).__name__}({sig})'


class DisplayUnits:
    """
    Create a DisplayUnits object where default units for representation are stored.
    
    Examples
    --------
    Its possible to change the default units of measure for the Stream show method:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water'], cache=True)
    >>> tmo.Stream.display_units.flow = 'kg/hr'
    >>> stream = tmo.Stream('stream', Water=1, units='kg/hr')
    >>> stream.show()
    Stream: stream
    phase: 'l', T: 298.15 K, P: 101325 Pa
    flow (kg/hr): Water  1
    
    >>> # Change back to kmol/hr
    >>> tmo.Stream.display_units.flow = 'kmol/hr'
    
    """
    def __init__(self, **display_units):
        dct = self.__dict__
        dct.update(display_units)
        dct['dims'] = {}
        list_keys = []
        for k, v in display_units.items():
            try: # Assume units is one string
                dims = ureg[v].dimensionality
            except:
                try: # Assume units are a list of possible units
                    dims = [ureg[i].dimensionality for i in v]
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
        sig = ', '.join([f"{i}={j!r}" for i,j in self.__dict__.items() if i != 'dims'])
        return f'{type(self).__name__}({sig})'


# %% Units of measure

chemical_units_of_measure = {
    'MW': AbsoluteUnitsOfMeasure('g/mol'),
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
    'S': AbsoluteUnitsOfMeasure('J/K/mol'),
    'S0': AbsoluteUnitsOfMeasure('J/K/mol'),
    'G': AbsoluteUnitsOfMeasure('J/mol'), 
    'U': AbsoluteUnitsOfMeasure('J/mol'),
    'H_excess': AbsoluteUnitsOfMeasure('J/mol'), 
    'S_excess': AbsoluteUnitsOfMeasure('J/mol'),
    'dipole': AbsoluteUnitsOfMeasure('Debye'),
    'delta': AbsoluteUnitsOfMeasure('Pa^0.5'),
    'epsilon': AbsoluteUnitsOfMeasure(''),
}
stream_units_of_measure = {
    'mol': AbsoluteUnitsOfMeasure('kmol/hr'),
    'mass': AbsoluteUnitsOfMeasure('kg/hr'),
    'vol': AbsoluteUnitsOfMeasure('m^3/hr'),
    'F_mass': AbsoluteUnitsOfMeasure('kg/hr'),
    'F_mol': AbsoluteUnitsOfMeasure('kmol/hr'),
    'F_vol': AbsoluteUnitsOfMeasure('m^3/hr'),
    'cost': AbsoluteUnitsOfMeasure('USD/hr'),
    'HHV': AbsoluteUnitsOfMeasure('kJ/hr'),
    'LHV': AbsoluteUnitsOfMeasure('kJ/hr'),
    'Hvap': AbsoluteUnitsOfMeasure('kJ/hr'),
    'Hf': AbsoluteUnitsOfMeasure('kJ/hr'), 
    'S0': AbsoluteUnitsOfMeasure('kJ/K/hr'), 
    'Hc': AbsoluteUnitsOfMeasure('kJ/hr'), 
    'H': AbsoluteUnitsOfMeasure('kJ/hr'),
    'S': AbsoluteUnitsOfMeasure('kJ/K/hr'),
    'G': AbsoluteUnitsOfMeasure('kJ/hr'),
    'U': AbsoluteUnitsOfMeasure('kJ/hr'),
    'C': AbsoluteUnitsOfMeasure('kJ/hr/K'),
}
for i in ('T', 'P', 'mu', 'V', 'rho', 'sigma',
          'kappa', 'nu', 'epsilon', 'delta',
          'Psat', 'Cp', 'Cn', 'alpha'):
    stream_units_of_measure[i] = chemical_units_of_measure[i]

power_utility_units_of_measure = {
    'cost': AbsoluteUnitsOfMeasure('USD/hr'),
    'rate': AbsoluteUnitsOfMeasure('kW'),
    'consumption': AbsoluteUnitsOfMeasure('kW'),
    'production': AbsoluteUnitsOfMeasure('kW'),
}

heat_utility_units_of_measure = {
    'cost': AbsoluteUnitsOfMeasure('USD/hr'),
    'flow': AbsoluteUnitsOfMeasure('kmol/hr'),
    'duty': AbsoluteUnitsOfMeasure('kJ/hr'),
}

definitions = {
    'MW': 'Molecular weight',
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
    'S0': 'Absolute entropy of formation',
    'G': 'Gibbs free energy',
    'U': 'Internal energy',
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

pint.BioSTEAM_units_loaded = True