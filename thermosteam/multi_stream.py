# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from .stream import Stream, assert_same_chemicals
from .thermal_condition import ThermalCondition
from .material_indexer import MolarFlowIndexer
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
        self._load_indexer(flow, phases, thermo.chemicals, phase_flows)
        self._streams = {}
        self._vle = Cache(VLE, self._imol, self._TP, thermo=self._thermo)
        self.price = price
        if units:
            indexer, factor = self._get_indexer_and_factor(units)
            indexer[...] = self.mol * factor
        self._register(ID)
         
    def _load_indexer(self, flow, phases, chemicals, phase_flows):
        if flow is ():
            if phase_flows:
                imol = MolarFlowIndexer(phases, chemicals=chemicals, **phase_flows)
            else:
                imol = MolarFlowIndexer.blank(phases, chemicals)
        else:
            assert not phase_flows, ("may specify either 'flow' or "
                                    "'phase_flows', but not both")
            if isinstance(flow, MolarFlowIndexer):
                imol = flow
            else:
                imol = MolarFlowIndexer.from_data(flow, phases, chemicals)
        self._imol = imol
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._imol = self._imol.get_phase(phase)
            stream._ID = None
            stream._TP = self._TP
            stream._thermo = self._thermo
            streams[phase] = stream
        return stream
    
    ### Property getters ###
    
    def get_flow(self, units, phase, IDs=...):
        indexer, factor = self._get_indexer_and_factor(units)
        return factor * indexer[phase, IDs]
    
    def set_flow(self, data, units, phase, IDs=...):
        indexer, factor = self._get_indexer_and_factor(units)
        indexer[phase, IDs] = np.asarray(data, dtype=float) / factor
    
    ### Stream data ###
    
    @property
    def phases(self):
        return self._imol._phases
    @phases.setter
    def phases(self, phases):
        phases = sorted(phases)
        if phases != self.phases:
            self._imol = self._imol.to_material_array(phases)
            self._vle = Cache(VLE, self._imol, self._TP, thermo=self._thermo)
    
    ### Net flow properties ###
    
    @property
    def H(self):
        return self.mixture.xH_at_TP(self._imol.iter_phase_data(), self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self._imol.iter_phase_data(), H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self._imol.iter_phase_data(), self._TP)
    @property
    def C(self):
        return self.mixture.xCp_at_TP(self._imol.iter_phase_data(), self._TP)
    @property
    def F_vol(self):
        return self.mixture.xV_at_TP(self._imol.iter_phase_data(), self._TP)
    @F_vol.setter
    def F_vol(self, value):
        self.vol[:] *= value/self.F_vol
    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self._imol['l'], self._TP)
    
    ### Composition properties ###
    
    @property
    def z_mol(self):
        mol = self.mol.sum(0)
        F_mol = mol.sum()
        return mol / F_mol if F_mol else mol.copy()
    @property
    def z_mass(self):
        mass = (self.chemicals.MW * self.mol).sum(0)
        F_mass = mass.sum()
        return mass / F_mass if F_mass else np.zeros(mass.shape)
    @property
    def z_vol(self):
        vol = self.vol.value.sum(0)
        F_vol = vol.sum()
        return vol / F_vol if F_vol else np.zeros(vol.shape)
    
    @property
    def V(self):
        return self.mixture.xV_at_TP(self._imol.iter_phase_composition(), self._TP)
    @property
    def kappa(self):
        return self.mixture.xkappa_at_TP(self._imol.iter_phase_composition(), self._TP)        
    @property
    def Cp(self):
        return self.mixture.xCp_at_TP(self._imol.iter_phase_composition(), self._TP)
    @property
    def mu(self):
        return self.mixture.xmu_at_TP(self._imol.iter_phase_composition(), self._TP)

    @property
    def sigma(self):
        mol = self._imol['l']
        F_mol = mol.sum()
        return self.mixture.xsigma_at_TP(mol / F_mol, self._TP) if F_mol else 0
    @property
    def epsilon(self):
        mol = self._imol['l']
        F_mol = mol.sum()
        return self.mixture.epsilon_at_TP(mol / F_mol, self._TP) if F_mol else 0
        
    ### Methods ###
        
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        multi = []; single = []; isa = isinstance
        for i in others:
            (multi if isa(i, MultiStream) else single).append(i)
        self.empty()
        for i in single:
            self.imol[i.phase] += i.mol    
        self.mol[:] += sum([i.mol for i in multi])
        self.H = sum([i.H for i in others])
        
    def split_to(self, s1, s2, split):
        mol = self.mol.sum(0)
        s1.mol[:] = dummy = mol * split
        s2.mol[:] = mol - dummy
        
    def link(self, other):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        self._TP = other._TP
        self._imol._data = other._imol._data
        self._streams = other._streams
        self._vle = other._vle
            
    def unlink(self):
        imol = self._imol
        imol._data_cache.clear()
        self._TP = TP = self._TP.copy()
        imol._data = self._imol._data.copy()
        self._vle = Cache(VLE, imol, TP, thermo=self._thermo)
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new._ID = None
        new._thermo = thermo = self._thermo
        new._imol = imol = self._imol.copy()
        new._TP = TP = self._TP.copy()
        new._vle = self._vle = Cache(VLE, imol, TP, thermo=thermo)
        return new
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        return self._vle()
    
    @property
    def z_equilibrium_chemicals(self):
        mol = self.mol.sum(0)
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
        mol = self.mol.sum(0)
        indices = chemicals.equilibrium_indices(mol != 0)
        return [chemicals_tuple[i] for i in indices]
    
    ### Casting ###
    
    @property
    def phase(self):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute 'phases'")
    @phase.setter
    def phase(self, phase):
        assert len(phase) == 1, f'invalid phase {repr(phase)}'
        self.__class__ = Stream
        self._imol = self._imol.to_chemical_array(phase)
        
    ### Representation ###
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .material_indexer import nonzeros
        IDs = self.chemicals.IDs
        basic_info = self._basic_info()
        all_IDs, _ = nonzeros(self.chemicals.IDs, self.mol.sum(0))
        all_IDs = tuple(all_IDs)
        T_units, P_units, flow_units, N = self.display_units.get_units(T=T, P=P, flow=flow, N=N)
        basic_info += Stream._info_phaseTP(self, self.phases, T_units, P_units)
        len_ = len(all_IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs]
        maxlen = max(all_lengths + [8]) 

        index, factor = self._get_indexer_and_factor(flow_units)
        first_line = f' flow ({flow_units}): '
        first_line_spaces = len(first_line)*" "

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = factor * index[phase, all_IDs] 
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
        for phase, data in self.mol.iter_phase_data():
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
    