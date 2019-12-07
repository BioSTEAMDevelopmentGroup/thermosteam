# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
from .display_units import DisplayUnits
from .base.units_of_measure import units_of_measure, get_dimensionality
from .exceptions import DimensionError
from .settings import settings
from .material_array import MolarFlow
from .thermal_condition import ThermalCondition
from .utils import Cache

# %%

class Stream:
    """Abstract class from material data."""
    __slots__ = ('_molar_flow', '_mass_flow_cache', '_volumetric_flow_cache',
                 '_thermal_condition', '_mixture', 'price')
    
    # Information regarding properties
    _prop_info = (
        # ID         # Description               # Dependency # Units      # Type
        ('T',        'temperature',              '',          'K',         'float'),
        ('H',        'enthalpy',                 'T',         'kJ/hr',     'float'),
        ('S',        'entropy',                  'TP',        'kJ/hr',     'float'),
        ('G',        'Gibbs free energy',        'TP',        'kJ/hr',     'float'),
        ('U',        'interal energy',           'TP',        'kJ/hr',     'float'),
        ('A',        'Helmholtz free energy',    'TP',        'kJ/hr',     'float'),
        ('Hf',       'enthalpy of formation',    '',          'kJ/hr',     'float'),
        ('P',        'pressure',                 '',          'Pa',        'float'),
        ('Cp',       'molar heat capacity',      'T',         'J/kg/K',    'float'),
        ('Vm',       'molar volume',             'TP',        'm^3/mol',   'float'),
        ('rho',      'density',                  'TP',        'kg/m^3',    'float'),
        ('nu',       'kinematic viscosity',      'TP',        'm^2/s',     'float'),
        ('mu',       'hydraulic viscosity',      'TP',        'Pa*s',      'float'),
        ('sigma',    'surface tension',          'T',         'N/m',       'float'),
        ('k',        'thermal conductivity',     'TP',        'W/m/K',     'float'),
        ('alpha',    'thermal diffusivity',      'TP',        'm^2/s',     'float'),
        ('Pr',       'Prantl number',            'TP',        "''",        'float'),
        ('mass',     'mass flow rates',          '',          'kg/hr',     'ndarray'),
        ('mol',      'molar flow rates',         '',          'kmol/hr',   'ndarray'),
        ('vol',      'volumetric flow rates',    'TP',        'm^3/hr',    'ndarray'),
        ('massnet',  'net mass flow rate',       '',          'kg/hr',     'float'),
        ('molnet',   'net molar flow rate',      '',          'kmol/hr',   'float'),
        ('volnet',   'net volumetric flow rate', 'TP',        'm^3/hr',    'float'),
        ('massfrac', 'mass fractions',           '',          'kg/kg',     'ndarray'),
        ('molfrac',  'molar fractions',          '',          'kmol/kmol', 'ndarray'),
        ('volfrac',  'volumetric fractions',     'TP',        'm^3/m^3',   'ndarray'))
    
    units = units_of_measure

    #: [DisplayUnits] Units of measure for IPython display
    display_units = DisplayUnits(T='K', P='Pa',
                                 flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                 fraction=False,
                                 N=5)

    def __init__(self, flow=(), phase='l', T=298.15, P=101325., units=None,
                 price=0., thermo=None, **chemical_flows):
        self._thermal_condition = ThermalCondition(T, P)
        self._thermo = thermo or settings.get_thermo(thermo)
        self._load_flow(flow, phase, thermo._chemicals, chemical_flows)
        if units:
            dimensionality = get_dimensionality(units)
            if dimensionality == self.molar_flow.units.dimensionality:
                self.molar_flow.set_data(self.molar_flow, units)
            elif dimensionality == self.mass_flow.units.dimensionality:
                self.mass_flow.set_data(self.molar_flow, units)
            elif dimensionality == self.volumetric_flow.units.dimensionality:
                self.volumetric_flow.set_data(self.molar_flow, units)
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dimensionality}'")
    
    def _load_flow(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if flow:
            assert not chemical_flows, ("may specify either 'flow' or "
                                        "'chemical_flows', but not both")
            molar_flow = MolarFlow.from_data(flow, phase, chemicals)
            
        elif chemical_flows:
            molar_flow = MolarFlow(chemicals, phase, **chemical_flows)
        else:
            molar_flow = MolarFlow.blank(phase, chemicals)
        self._molar_flow = molar_flow
        self._mass_flow = Cache(molar_flow.as_mass_flow)
        self._volumetric_flow = Cache(molar_flow.as_volumetric_flow)

    @property
    def mass_flow(self):
        return self._mass_flow_cache()
    @property
    def volumetric_flow(self):
        return self._volumetric_flow_cache()
    
    
