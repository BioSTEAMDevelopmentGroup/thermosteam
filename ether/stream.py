# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
import numpy as np
from array_collections import property_array
from free_properties import PropertyFactory
from .chemicals import Chemicals
from .mixture import IdealMixture
from .display_units import DisplayUnits
from .exceptions import DimensionError
from .base.units_of_measure import units_of_measure, dimensionality, _Q
from .equilibrium import ActivityCoefficients, IdealActivityCoefficients, \
                         DortmundActivityCoefficients, FugacityCoefficients
from .thermo import Thermo

# %% Flow properties

@PropertyFactory
def MassFlow(self):
    """Mass flow (kg/hr)."""
    mol, index, MW = self.data
    return mol[index] * MW
    
@MassFlow.setter
def MassFlow(self, value):
    mol, index, MW = self.data
    mol[index] = value/MW

@PropertyFactory    
def VolumetricFlow(self):
    """Volumetric flow (m^3/hr)."""
    stream, mol, index, V = self.data
    molar_flow = mol[index]
    if molar_flow:
        return V(stream._phase, stream.T, stream.P) * molar_flow * 1000.
    else:
        return 0.

@VolumetricFlow.setter
def VolumetricFlow(self, value):
    stream, mol, index, V = self.data
    if value:
        mol[index] = value/(V(stream._phase, stream.T, stream.P) * 1000)
    else:
        mol[index] = 0.

def flow(fget):
    def fset(self, value):
        flow = fget(self)
        if flow is not value:
            flow[:] = value
    return property(fget, fset)


# %%

class Stream:
    """Abstract class from material data."""
    __slots__ = ('phase', 'T', 'P', 'price',
                 '_mol', '_mass', '_vol',  
                 '_chemicals', '_mixture')
    
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
    _mol_dim = dimensionality('kmol/hr')
    _mass_dim = dimensionality('kg/hr')
    _vol_dim = dimensionality('m^3/hr')

    #: [DisplayUnits] Units of measure for IPython display
    display_units = DisplayUnits(T='K', P='Pa',
                                 flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                 fraction=False,
                                 N=5)

    def __init__(self, flow=(), phase='l', T=298.15, P=101325., units=None,
                 price=0., chemicals=None, mixture=None, gamma=None, phi=None,
                 **chemical_flows):
        self._load_chemicals(chemicals)
        chemicals = self._chemicals
        self.phase = phase
        self.T = T
        self.P = P
        self._mixture = mixture or IdealMixture(chemicals)
        self._load_flow(flow, chemicals, chemical_flows)
        if units:
            q = _Q(self._mol)
            dim = units.dimensionality
            if dim == self._mol_dim:
                self._mol[:] = q.to('kmol/hr').magnitude
            elif dim == self._mass_dim:
                self._mass[:] = q.to('kg/hr').magnitude
            elif dim == self._vol_dim:
                self._vol[:] = q.to('m3/hr').magnitude
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dim}'")
        self._gamma = gamma or self._default_thermo.gamma
        self._phi = phi or self._default_thermo.phi
    
    def _load_flow(self, flow, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        size = len(flow)
        if size:
            if chemical_flows:
                raise ValueError("may specify either 'flow' or "
                                 "'chemical_flows', but not both")
            elif size == chemicals.size:
                self._mol = np.array(flow, float)
            else:
                raise ValueError('size of flow rates must be equal to '
                                 'size of stream chemicals')
        else:
            self._mol = np.zeros(self._chemicals.size, float)
            if chemical_flows:
                IDs, flows = zip(*chemical_flows.items())
                self._mol[IDs] = flows
    
    def _load_chemicals(self, chemicals):
        if not chemicals: 
            assert self._default_chemicals, 'must define Stream.chemicals first'
            self._chemicals = chemicals = self._default_chemicals
        elif isinstance(chemicals, Chemicals):
            self._chemicals = chemicals
        else:
            raise ValueError("chemicals must be a 'Chemicals' object, "
                            f"not '{type(chemicals).__name__}'")
    
    def set_flow(self, units=None, **chemical_flows):
        """Set flow rates at given units"""
        IDs, flow = zip(*chemical_flows.items())
        index = self.indices(IDs)
        if units:
            q = _Q(flow, units)
            dim = q.dimensionality
            if dim == self._mol_dim:
                self._mol[index] = q.to('kmol/hr').magnitude
            elif dim == self._mass_dim:
                self._mass[index] = q.to('kg/hr').magnitude
            elif dim == self._vol_dim:
                self._vol[index] = q.to('m3/hr').magnitude
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dim}'")
    
    def get_flow(self, *chemicals, units=None):
        """Get flow rates of chemicals in given units."""
        index = self.indices(chemicals)
        if units:
            q = _Q(1, units)
            dim = q.dimensionality
            if dim == self._mol_dim:
                return self._mol[index]*q.to('kmol/hr').magnitude
            elif dim == self._mass_dim:
                return self._mass[index]*q.to('kg/hr').magnitude
            elif dim == self._vol_dim:
                return self._vol[index]*q.to('m3/hr').magnitude
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dim}'")
        else:
            return self._mol[index]
    
    def mix(self, other):
        assert self._chemicals is other._chemicals, "chemicals must be the same to mix material data"
        if isinstance(other, Stream):
            self._material_data += other._material_data
        else:
            self._material_data += other._material_data.sum(0)


# class MixedPhaseMaterialData(MaterialData):
#     __slots__ = ('_chemicals', '_material_data', '_phases', '_phase_data')
    
#     def __init__(self, chemicals, phases, data):
#         self._chemicals = chemicals
#         self._material_data = data
#         self._phases = phases
#         self._phase_data = dict(zip(phases, data))
        
#     def mix(self, other):
#         assert self._chemicals is other._chemicals, "chemicals must be the same to mix material data"
#         phases = self._phases
#         data = self._material_data
#         if isinstance(other, SinglePhaseMaterialData):
#             other_phase = other._phase
#             if other_phase in phases:
#                 self._phase_data[other_phase] += other._material_data
#             else:
#                 phases += other_phase
#                 self._phases = phases
#                 self._material_data = data = np.vstack((data, other._material_data))     
#                 self._phase_data = dict(zip(phases, data))
#         else:
#             other_phases = other._phases
#             other_data = other._material_data
#             if other_phases == phases:
#                 data += other_data
#             else:
#                 new_phases = ''
#                 phase_data = self._phase_data
#                 other_phase_data = other._phase_data
#                 for phase in other_phases:
#                     if phase in phases:
#                         phase_data[phase] += other_phase_data[phase]
#                     else:
#                         new_phases += phase
#                 if new_phases:
#                     new_data = [other_phase_data[phase] for phase in new_phases]
#                     phases += new_phases
#                     self._phases = phases              
#                     self._material_data = data = np.vstack((data, new_data))     
#                     self._phase_data = dict(zip(phases, data))
    
    # def __repr__(self):
    #     nonzero = np.any(self.data, 0)
    #     IDs = self.chemicals.IDs
    #     IDs = [i for i,j in zip(IDs, nonzero) if j]
    #     return f"<{type(self).__name__}:>"