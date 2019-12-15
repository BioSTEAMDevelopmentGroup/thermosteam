# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
from .base.units_of_measure import get_dimensionality, units_of_measure, flow_units_of_measure, convert
from .base.display_units import DisplayUnits
from .exceptions import DimensionError
from .settings import settings
from .material_index import ChemicalMolarFlowIndex, ChemicalMassFlowIndex, ChemicalVolumetricFlowIndex
from .thermal_condition import ThermalCondition
from .phase_container import new_phase_container
from .equilibrium import BubblePoint, DewPoint
from .registry import registered
from .utils import Cache
from .equilibrium import VLE
import numpy as np

__all__ = ('Stream', )


# %% Utilities

molar_flow_dimensionality = ChemicalMolarFlowIndex.units.dimensionality
mass_flow_dimensionality = ChemicalMassFlowIndex.units.dimensionality
volumetric_flow_dimensionality = ChemicalVolumetricFlowIndex.units.dimensionality

def assert_same_chemicals(stream, others):
    chemicals = stream.chemicals
    assert all([chemicals is i.chemicals for i in others]), "chemicals must match to mix streams"


# %%
@registered(ticket_name='s')
class Stream:
    __slots__ = ('_ID', '_molar_index', '_TP', '_thermo', '_streams', '_vle', 'price')
    
    #: [DisplayUnits] Units of measure for IPython display (class attribute)
    display_units = DisplayUnits(T='K', P='Pa',
                                 flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                 N=5)

    def __init__(self, ID='', flow=(), phase='l', T=298.15, P=101325., units=None,
                 price=0., thermo=None, **chemical_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_index(flow, phase, thermo.chemicals, chemical_flows)
        self.price = price
        if units:
            self._select_indexer(units).set_data(self.molar_flow, units)
        self._register(ID)
    
    def _load_index(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if flow is ():
            if chemical_flows:
                molar_index = ChemicalMolarFlowIndex(phase, chemicals=chemicals, **chemical_flows)
            else:
                molar_index = ChemicalMolarFlowIndex.blank(phase, chemicals)
        else:
            assert not chemical_flows, ("may specify either 'flow' or "
                                        "'chemical_flows', but not both")
            if isinstance(flow, ChemicalMolarFlowIndex):
                molar_index = flow 
                molar_index.phase = phase
            else:
                molar_index = ChemicalMolarFlowIndex.from_data(flow, phase, chemicals)
        self._molar_index = molar_index

    def _select_indexer(self, units):
        dimensionality = get_dimensionality(units)
        if dimensionality == molar_flow_dimensionality:
            return self.molar_index
        elif dimensionality == mass_flow_dimensionality:
            return self.mass_index
        elif dimensionality == volumetric_flow_dimensionality:
            return self.volumetric_index
        else:
            raise DimensionError(f"dimensions for flow units must be in molar, "
                                 f"mass or volumetric flow rates, not '{dimensionality}'")

    ### Property getters ###

    def get_flow(self, units, IDs=...):
        indexer = self._select_indexer(units)
        flow = indexer[IDs]
        return flow * indexer.units.conversion_factor(units)
    
    def set_flow(self, data, units, IDs=...):
        indexer = self._select_indexer(units)
        indexer[IDs] = np.asarray(data, dtype=float) / indexer.units.conversion_factor(units)    
    
    def get_property(self, name, units):
        if name in flow_units_of_measure:
            original_units = flow_units_of_measure[name]
        elif name in units_of_measure:
            original_units = units_of_measure[name]
        else:
            raise ValueError(f"no property with name '{name}'")
        value = getattr(self, name)
        factor = original_units.conversion_factor(units)
        return value * factor
    
    ### Stream data ###
    
    @property
    def thermo(self):
        return self._thermo
    @property
    def chemicals(self):
        return self._thermo.chemicals
    @property
    def mixture(self):
        return self._thermo.mixture

    @property
    def thermal_condition(self):
        return self._TP

    @property
    def T(self):
        return self._TP.T
    @T.setter
    def T(self, T):
        self._TP.T = T
    
    @property
    def P(self):
        return self._TP.P
    @P.setter
    def P(self, P):
        self._TP.P = P
    
    @property
    def phase(self):
        return self._molar_index.phase
    @phase.setter
    def phase(self, phase):
        self._molar_index.phase = phase
    
    @property
    def molar_flow(self):
        return self._molar_index._data
    @molar_flow.setter
    def molar_flow(self, value):
        if self.molar_flow is not value:
            raise AttributeError("cannot replace attribute with another object")
    
    @property
    def mass_flow(self):
        return self.mass_index._data
    @mass_flow.setter
    def mass_flow(self, value):
        if self.mass is not value:
            raise AttributeError("cannot replace attribute with another object")
    
    @property
    def volumetric_flow(self):
        return self.volumetric_index._data
    @volumetric_flow.setter
    def volumetric_flow(self, value):
        if self.vol is not value:
            raise AttributeError("cannot replace attribute with another object")
        
    @property
    def molar_index(self):
        return self._molar_index
    @molar_index.setter
    def molar_index(self, value):
        if self._molar_index is not value:
            raise AttributeError("cannot replace attribute with another object")
    
    @property
    def mass_index(self):
        return self._molar_index.by_mass()
    @mass_index.setter
    def mass_index(self, value):
        if self.mass_index is not value:
            raise AttributeError("cannot replace attribute with another object")
    
    @property
    def volumetric_index(self):
        return self._molar_index.by_volume(self._TP)
    @volumetric_index.setter
    def volumetric_index(self, value):
        if self.volumetric_index is not value:
            raise AttributeError("cannot replace attribute with another object")
    
    ### Net flow properties ###
    
    @property
    def cost(self):
        return self.price * self.net_mass_flow
    
    @property
    def net_molar_flow(self):
        return self.mol.sum()
    @property
    def net_mass_flow(self):
        return (self.chemicals.MW * self.molar_flow).sum()
    @property
    def net_volumetric_flow(self):
        return self.mixture.V_at_TP(self.phase, self.molar_flow, self._TP)
    
    @property
    def H(self):
        return self.mixture.H_at_TP(self.phase, self.molar_flow, self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.solve_T(self.phase, self.molar_flow, H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.S_at_TP(self.phase, self.molar_flow, self._TP)
    
    @property
    def Hf(self):
        return (self.chemicals.Hf * self.molar_flow).sum()
    @property
    def Hc(self):
        return (self.chemicals.Hc * self.molar_flow).sum()    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self.molar_flow, self._TP)
    
    @property
    def C(self):
        return self.mixture.Cp_at_TP(self.molar_flow, self._TP)
    
    ### Composition properties ###
    
    @property
    def molar_composition(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return mol / molnet if molnet else mol.copy()
    @property
    def mass_composition(self):
        mass = self.chemicals.MW * self.molar_flow
        massnet = mass.sum()
        return mass / massnet if massnet else mass
    @property
    def volumetric_composition(self):
        vol = self.volumetric_flow
        volnet = vol.sum()
        return vol / volnet if volnet else vol.value
    
    @property
    def V(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.V_at_TP(self.phase, mol / molnet, self._TP) if molnet else 0
    @property
    def kappa(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.kappa_at_TP(self.phase, mol / molnet, self._TP) if molnet else 0
    @property
    def Cp(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.Cp_at_TP(self.phase, mol / molnet, self._TP) if molnet else 0
    @property
    def mu(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.mu_at_TP(self.phase, mol / molnet, self._TP) if molnet else 0
    @property
    def sigma(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.sigma_at_TP(mol / molnet, self._TP) if molnet else 0
    @property
    def epsilon(self):
        mol = self.molar_flow
        molnet = mol.sum()
        return self.mixture.epsilon_at_TP(mol / molnet, self._TP) if molnet else 0
    
    ### Stream methods ###
    
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        isa = isinstance
        self.mol[:] = sum([i.mol if isa(i, Stream) else i.mol.sum(0) for i in others])
        self.H = sum([i.H for i in others])
    
    def split_to(self, s1, s2, split):
        mol = self.molar_flow
        s1.mol[:] = dummy = mol * split
        s2.mol[:] = mol - dummy
        
    def link(self, other, TP=True, flow=True, phase=True):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        other._molar_index._data_cache.clear()
        if TP:
            self._TP = other._TP
        if flow:
            self._molar_index._data = other._molar_index._data
        if phase:
            self._molar_index._phase = other._molar_index._phase
            
    def unlink(self):
        self._molar_index._data_cache.clear()
        self._TP = self._TP.copy()
        self._molar_index._data = self._molar_index._data.copy()
        self._molar_index._phase = new_phase_container(self._molar_index._phase)
    
    def copy_like(self, other):
        self._molar_index.copy_like(other._molar_index)
        self._TP.copy_like(other._TP)
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new._ID = None
        new._thermo = self._thermo
        new._molar_index = self._molar_index.copy()
        new._TP = self._TP.copy()
        return new
    __copy__ = copy
    
    def empty(self):
        self._molar_index._data[:] = 0
    
    ### Equilibrium ###

    @property
    def vle(self):
        self.phases = 'gl'
        return self.vle

    @property
    def z_chemicals(self):
        mol = self.molar_flow
        chemicals = self.chemicals
        indices = chemicals.equilibrium_indices(mol != 0)
        flow = mol[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        z = flow / netflow  
        chemicals_tuple = chemicals.tuple
        return z, [chemicals_tuple[i] for i in indices]
    
    @property
    def equilibrim_chemicals(self):
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.equilibrium_indices(self.molar_flow != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def z(self):
        mol = self.molar_flow
        indices = self.chemicals.equilibrium_indices(mol != 0)
        flow = mol[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        return flow / netflow  
    
    @property
    def bubble_point(self):
        return BubblePoint(self.equilibrim_chemicals, self._thermo)
    
    @property
    def dew_point(self):
        return DewPoint(self.equilibrim_chemicals, self._thermo)
    
    @property
    def T_bubble(self):
        z, chemicals = self.z_chemicals
        bp = BubblePoint(chemicals, self._thermo)
        return bp.solve_Ty(z, self.P)[0]
    
    @property
    def T_dew(self):
        z, chemicals = self.z_chemicals
        dp = DewPoint(chemicals, self._thermo)
        return dp.solve_Tx(z, self.P)[0]
    
    @property
    def P_bubble(self):
        z, chemicals = self.z_chemicals
        bp = BubblePoint(chemicals, self._thermo)
        return bp.solve_Py(z, self.T)[0]
    
    @property
    def P_dew(self):
        z, chemicals = self.z_chemicals
        dp = DewPoint(chemicals, self._thermo)
        return dp.solve_Px(z, self.T)[0]
    
    ### Casting ###
    
    @property
    def phases(self):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'phases'")
    @phases.setter
    def phases(self, phases):
        self.__class__ = multi_stream.MultiStream
        self._molar_index = self._molar_index.to_material_array(phases)
        self._vle = Cache(VLE, self._molar_index, self._TP, thermo=self._thermo)
    
    ### Representation ###
    
    def _basic_info(self):
        return type(self).__name__ + ': ' + (self.ID or '') + '\n'
    
    def _info_phaseTP(self, phase, T_units, P_units):
        T = convert(self.T, 'K', T_units)
        P = convert(self.P, 'Pa', P_units)
        s = '' if isinstance(phase, str) else 's'
        return f" phase{s}: {repr(phase)}, T: {T:.5g} {T_units}, P: {P:.6g} {P_units}\n"
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .material_index import nonzeros
        basic_info = self._basic_info()
        IDs = self.chemicals.IDs
        data = self.molar_index.data
        IDs, data = nonzeros(IDs, data)
        IDs = tuple(IDs)
        T_units, P_units, flow_units, N = self.display_units.get_units(T=T, P=P, flow=flow, N=N)
        basic_info += self._info_phaseTP(self.phase, T_units, P_units)
        len_ = len(IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 
        
        # Start of third line (flow rates)
        index = self._select_indexer(flow_units)
        if index is self._molar_index:
            flow = 'molar_flow'
        elif index is self._mass_index:
            flow = 'mass_flow'
        else:
            flow = 'volumetric_flow'
        beginning = f' {flow} ({flow_units}): '
            
        # Remaining lines (all flow rates)
        new_line_spaces = len(beginning) * ' '
        flow_array = index.get_data(flow_units, IDs)        
        flowrates = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 1
        _N = N - 1
        for i in range(len_-1):
            spaces = ' ' * (maxlen - lengths[i])
            if i == _N:
                flowrates += '...\n' + new_line_spaces
                break
            flowrates += IDs[i] + spaces + f' {flow_array[i]:.3g}\n' + new_line_spaces
        spaces = ' ' * (maxlen - lengths[len_-1])
        flowrates += IDs[len_-1] + spaces + f' {flow_array[len_-1]:.3g}'
        return (basic_info 
              + beginning
              + flowrates)

    def show(self, T=None, P=None, flow=None, N=None):
        """Print all specifications.
        
        Parameters
        ----------
        T: str, optional
            Temperature units.
        P: str, optional
            Pressure units.
        flow: str, optional
            Flow rate units.
        N: int, optional
            Number of compounds to display.
        
        Notes
        -----
        Default values are stored in `Stream.display_units`.
        
        """
        print(self._info(T, P, flow, N))
    _ipython_display_ = show
    
    def print(self):
        from .utils import repr_IDs_data, repr_kwarg
        chemical_flows = repr_IDs_data(self.chemicals.IDs, self.molar_flow)
        price = repr_kwarg('price', self.price)
        print(f"{type(self).__name__}(ID={repr(self.ID)}, phase={repr(self.phase)}, T={self.T:.2f}, "
              f"P={self.P:.6g}{price}{chemical_flows})")
        
from . import multi_stream