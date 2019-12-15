# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from .stream import Stream, assert_same_chemicals
from .base.units_of_measure import get_dimensionality
from .thermal_condition import ThermalCondition
from .material_index import MolarFlowIndex
from .exceptions import DimensionError
from .settings import settings
from .equilibrium import VLE
from .utils import Cache
import numpy as np

__all__ = ('MultiStream', )

class MultiStream(Stream):
    __slots__ = ()
    def __init__(self, ID="", flow=(), T=298.15, P=101325., phases='gl', units=None,
                 thermo=None, price=None, **phase_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_index(flow, phases, thermo.chemicals, phase_flows)
        self._streams = {}
        self._vle = Cache(VLE, self._molar_index, self._TP, thermo=self._thermo)
        self.price = price
        if units:
            self._select_indexer(units).set_data(self.molar_flow, units)
        self._register(ID)
         
    def _load_index(self, flow, phases, chemicals, phase_flows):
        if flow is ():
            if phase_flows:
                molar_index = MolarFlowIndex(phases, chemicals=chemicals, **phase_flows)
            else:
                molar_index = MolarFlowIndex.blank(phases, chemicals)
        else:
            assert not phase_flows, ("may specify either 'flow' or "
                                    "'phase_flows', but not both")
            if isinstance(flow, MolarFlowIndex):
                molar_index = flow
            else:
                molar_index = MolarFlowIndex.from_data(flow, phases, chemicals)
        self._molar_index = molar_index
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._molar_index = self._molar_index.get_phase(phase)
            stream._ID = None
            stream._TP = self._TP
            stream._thermo = self._thermo
            streams[phase] = stream
        return stream
    
    ### Property getters ###
    
    def get_flow(self, units, phase, IDs=...):
        indexer = self._select_indexer(units)
        flow = indexer[phase, IDs]
        return flow * indexer.units.conversion_factor(units)
    
    def set_flow(self, data, units, phase, IDs=...):
        indexer = self._select_indexer(units)
        indexer[phase, IDs] = np.asarray(data, dtype=float) / indexer.units.conversion_factor(units)    
    
    ### Stream data ###
    
    @property
    def phases(self):
        return self._molar_index._phases
    @phases.setter
    def phases(self, phases):
        phases = sorted(phases)
        if phases != self.phases:
            self._molar_index = self._molar_index.to_material_array(phases)
            self._vle = Cache(VLE, self._molar_index, self._TP, thermo=self._thermo)
    
    ### Net flow properties ###
    
    @property
    def H(self):
        return self.mixture.xH_at_TP(self._molar_index.iter_phase_data(), self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self._molar_index.iter_phase_data(), H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self._molar_index.iter_phase_data(), self._TP)
    @property
    def C(self):
        return self.mixture.xCp_at_TP(self._molar_index.iter_phase_data(), self._TP)
    @property
    def net_volumetric_flow(self):
        return self.mixture.xV_at_TP(self._molar_index.iter_phase_data(), self._TP).sum()
    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self._molar_index['l'], self._TP)
    
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
        return mass / massnet if massnet else np.zeros(mass.shape)
    @property
    def volumetric_composition(self):
        vol = self.volumetric_flow.values()
        volnet = vol.sum()
        return vol / volnet if volnet else np.zeros(vol.shape)
    
    @property
    def V(self):
        return self.mixture.xV_at_TP(self._molar_index.iter_phase_composition(), self._TP)
    @property
    def kappa(self):
        return self.mixture.xkappa_at_TP(self._molar_index.iter_phase_composition(), self._TP)        
    @property
    def Cp(self):
        return self.mixture.xCp_at_TP(self._molar_index.iter_phase_composition(), self._TP)
    @property
    def mu(self):
        return self.mixture.xmu_at_TP(self._molar_index.iter_phase_composition(), self._TP)

    @property
    def sigma(self):
        mol = self._molar_index['l']
        molnet = mol.sum()
        return self.mixture.xsigma_at_TP(mol / molnet, self._TP) if molnet else 0
    @property
    def epsilon(self):
        mol = self._molar_index['l']
        molnet = mol.sum()
        return self.mixture.epsilon_at_TP(mol / molnet, self._TP) if molnet else 0
        
    ### Methods ###
        
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        multi = []; single = []; isa = isinstance
        for i in others:
            (multi if isa(i, MultiStream) else single).append(i)
        self.empty()
        for i in single:
            self.molar_index[i.phase] += i.mol    
        self.molar_flow[:] += sum([i.mol for i in multi])
        self.H = sum([i.H for i in others])
        
    def split_to(self, s1, s2, split):
        mol = self.mol.sum(0)
        s1.molar_flow[:] = dummy = mol * split
        s2.molar_flow[:] = mol - dummy
        
    def link(self, other):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        self._TP = other._TP
        self._molar_index._data = other._molar_index._data
        self._streams = other._streams
        self._vle = other._vle
            
    def unlink(self):
        molar_index = self._molar_index
        molar_index._data_cache.clear()
        self._TP = TP = self._TP.copy()
        molar_index._data = self._molar_index._data.copy()
        self._vle = Cache(VLE, molar_index, TP, thermo=self._thermo)
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new._ID = None
        new._thermo = thermo = self._thermo
        new._molar_index = molar_index = self._molar_index.copy()
        new._TP = TP = self._TP.copy()
        new._vle = self._vle = Cache(VLE, molar_index, TP, thermo=thermo)
        return new
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        return self._vle()
    
    @property
    def z_chemicals(self):
        mol = self.molar_flow.sum(0)
        chemicals = self.chemicals
        indices = chemicals.equilibrium_indices(mol != 0)
        flow = mol[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        z = flow / netflow  
        chemicals_tuple = chemicals.tuple
        return z, [chemicals_tuple[i] for i in indices]
    
    @property
    def equilibrium_chemicals(self):
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        mol = self.molar_flow.sum(0)
        indices = chemicals.equilibrium_indices(mol != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def z(self):
        mol = self.molar_flow.sum(0)
        indices = self.chemicals.equilibrium_indices(mol != 0)
        flow = mol[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        return flow / netflow  
    
    ### Casting ###
    
    @property
    def phase(self):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'phases'")
    @phase.setter
    def phase(self, phase):
        assert len(phase) == 1, f'invalid phase {repr(phase)}'
        self.__class__ = Stream
        self._molar_index = self._molar_index.to_chemical_array(phase)
        
    ### Representation ###
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .material_index import nonzeros
        IDs = self.chemicals.IDs
        basic_info = self._basic_info()
        all_IDs, _ = nonzeros(self.chemicals.IDs, self.molar_flow.sum(0))
        all_IDs = tuple(all_IDs)
        T_units, P_units, flow_units, N = self.display_units.get_units(T=T, P=P, flow=flow, N=N)
        basic_info += Stream._info_phaseTP(self, self.phases, T_units, P_units)
        len_ = len(all_IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs]
        maxlen = max(all_lengths + [8]) 

        index = self._select_indexer(flow_units)
        if index is self._molar_index:
            flow = 'molar_flow'
        elif index is self._mass_index:
            flow = 'mass_flow'
        else:
            flow = 'volumetric_flow'
        first_line = ' ' + flow + f' ({flow_units}):'
        first_line_spaces = len(first_line)*" "

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = index.get_data(flow_units, phase, all_IDs)
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
        
            # Get basic structure for phase data
            
            beginning = (first_line or first_line_spaces) + f' ({phase}) '
            first_line = False
            new_line_spaces = len(beginning) * ' '

            # Set chemical data
            flowrates = ''
            l = len(data)
            lengths = [len(i) for i in IDs]
            _N = N - 1
            for i in range(l-1):
                spaces = ' ' * (maxlen - lengths[i])
                if i == _N:
                    flowrates += '...\n' + new_line_spaces
                    break
                flowrates += f'{IDs[i]} ' + spaces + \
                    f' {data[i]:.4g}\n' + new_line_spaces
            spaces = ' ' * (maxlen - lengths[l-1])
            flowrates += (f'{IDs[l-1]} ' + spaces
                          + f' {data[l-1]:.4g}')

            # Put it together
            phases_flowrates_info += beginning + flowrates + '\n'
            
        return basic_info + phases_flowrates_info[:-1]
    
    def print(self):
        from .utils import repr_kwarg, repr_couples
        IDs = self.chemicals.IDs
        phase_data = []
        for phase, data in self.molar_flow.iter_phase_data():
            IDdata = repr_couples(", ", IDs, data)
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
        dlim = ", "
        phase_data = dlim.join(phase_data)
        phases = f'phases={self.phases}'
        if phase_data:
            phase_data = dlim + phase_data
        price = repr_kwarg('price', self.price)
        print(f"{type(self).__name__}(ID={repr(self.ID)}, {phases}, T={self.T:.2f}, "
              f"P={self.P:.6g}{price}{phase_data})")
    