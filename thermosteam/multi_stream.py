# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from .stream import Stream, assert_same_chemicals
from .base.units_of_measure import get_dimensionality
from .thermal_condition import ThermalCondition
from .material_array import MolarFlow, MassFlow, VolumetricFlow, ChemicalMolarFlow
from .exceptions import DimensionError
from .settings import settings
from .equilibrium import VLE
from .utils import define_from

__all__ = ('MultiStream', )

shared_properties = (
    'thermo', 'mixture', 'chemicals', 'thermal_condition', 'T', 'P', 'cost', 'get_property',
    'molar_data', 'mass_data', 'volumetric_data', 'molar_flow', 'mass_flow', 'volumetric_flow',
    'bubble_point', 'dew_point', 'T_bubble', 'T_dew', 'P_bubble', 'P_dew', 'Hf', 'Hc',
    'net_molar_flow', 'net_mass_flow', 'empty', 'show', 'display_units', '_ipython_display_',
)
@define_from(Stream, shared_properties)
class MultiStream:
    __slots__ = ('_molar_flow', '_TP', '_thermo', '_streams', '_vle', 'price')
    
    def __init__(self, flow=(), T=298.15, P=101325., phases='lg', units=None,
                 thermo=None, price=None, **phase_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_flow(flow, phases, thermo.chemicals, phase_flows)
        self._streams = {}
        if 'l' in phases and 'g' in phases:
            self._vle = VLE(self._molar_flow, self._TP, thermo=self._thermo)
        else:
            self._vle = None
        self.price = price
        if units:
            dimensionality = get_dimensionality(units)
            if dimensionality == MolarFlow.units.dimensionality:
                self.molar_flow.set_data(self.molar_data, units)
            elif dimensionality == MassFlow.units.dimensionality:
                self.mass_flow.set_data(self.molar_data, units)
            elif dimensionality == VolumetricFlow.units.dimensionality:
                self.volumetric_flow.set_data(self.molar_data, units)
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dimensionality}'")
                
    def _load_flow(self, flow, phases, chemicals, phase_flows):
        if flow is ():
            if phase_flows:
                molar_flow = MolarFlow(phases, chemicals=chemicals, **phase_flows)
            else:
                molar_flow = MolarFlow.blank(phases, chemicals)
        else:
            assert not phase_flows, ("may specify either 'flow' or "
                                    "'phase_flows', but not both")
            if isinstance(flow, MolarFlow):
                molar_flow = flow
            else:
                molar_flow = MolarFlow.from_data(flow, phases, chemicals)
        self._molar_flow = molar_flow
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._molar_flow = ChemicalMolarFlow.from_data(self.molar_flow[phase],
                                                             (phase,), 
                                                             chemicals=self.chemicals)
            stream._TP = self._TP
            stream._thermo = self._thermo
            streams[phase] = stream
        return stream
    
    ### Stream data ###
    
    @property
    def phases(self):
        return self._molar_flow._phases
    
    
    ### Net flow properties ###
    
    @property
    def net_volumetric_flow(self):
        return self.mixture.xV_at_TP(self.molar_flow.iter_phase_data(), self._TP).sum()

    @property
    def H(self):
        return self.mixture.xH_at_TP(self.molar_flow.iter_phase_data(), self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self.molar_flow.iter_phase_data(), H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self.molar_flow.iter_phase_data(), self._TP)
    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self._molar_flow['l'], self._TP)
    
    @property
    def C(self):
        return self.mixture.xCp_at_TP(self.molar_flow.iter_phase_data(), self._TP)
    
    ### Composition properties ###
    
    @property
    def molar_composition(self):
        molar_flow = self.molar_data
        net_molar_flow = molar_flow.sum()
        return molar_flow / net_molar_flow if net_molar_flow else molar_flow.copy()
    @property
    def mass_composition(self):
        mass_flow = self.chemicals.MW * self.molar_data
        net_mass_flow = mass_flow.sum()
        return mass_flow / net_mass_flow if net_mass_flow else mass_flow
    @property
    def volumetric_composition(self):
        volumetric_flow = self.volumetric_data.values()
        net_volumetric_flow = volumetric_flow.sum()
        return volumetric_flow / net_volumetric_flow if net_volumetric_flow else volumetric_flow
    
    @property
    def V(self):
        return self.mixture.xV_at_TP(self._molar_flow.iter_phase_composition(), self._TP)
        
    @property
    def kappa(self):
        return self.mixture.xkappa_at_TP(self._molar_flow.iter_phase_composition(), self._TP)
        
    @property
    def Cp(self):
        return self.mixture.xCp_at_TP(self._molar_flow.iter_phase_composition(), self._TP)
        
    @property
    def mu(self):
        return self.mixture.xmu_at_TP(self._molar_flow.iter_phase_composition(), self._TP)

    @property
    def sigma(self):
        molar_flow = self._molar_flow['l']
        net = molar_flow.sum()
        if net:
            return self.mixture.xsigma_at_TP(molar_flow / net, self._TP)
        else:
            return 0
    @property
    def epsilon(self):
        molar_flow = self._molar_flow['l']
        net = molar_flow.sum()
        if net:
            return self.mixture.epsilon_at_TP(molar_flow / net, self._TP)
        else:
            return 0
        
    ### Methods ###
        
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        multi = []; single = []; isa = isinstance
        for i in others:
            (multi if isa(i, MultiStream) else single).append(i)
        self.empty()
        for i in single:
            self.molar_flow[i.phase] += i.molar_data    
        self.molar_data[:] += sum([i.molar_data for i in multi])
        self.H = sum([i.H for i in others])
        
    def split_to(self, s1, s2, split):
        molar_data = self.molar_data.sum(0)
        s1.molar_data[:] = dummy = molar_data * split
        s2.molar_data[:] = molar_data - dummy
        
    def link_with(self, other):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        other._molar_flow._data_cache.clear()
        self._TP = other._TP
        self._molar_flow._data = other._molar_flow._data
        self._streams = other._streams
        self._vle = other._vle
            
    def unlink(self):
        molar_flow = self._molar_flow
        molar_flow._data_cache.clear()
        self._TP = TP = self._TP.copy()
        molar_flow._data = self._molar_flow._data.copy()
        self._vle = VLE(molar_flow, TP)
    
    def copy_like(self, other):
        self.molar_data[:] = other.molar_data
        self.TP.copy_like(other)
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new._thermo = self._thermo
        new._molar_flow = self._molar_flow.copy()
        new._TP = TP = self._TP.copy()
        new._vle = VLE(self.molar_data, TP)
        return new
    
    ### Equilibrium ###
    @property
    def vle(self):
        return self._vle
    
    @property
    def z_chemicals(self):
        molar_data = self.molar_data.sum(0)
        chemicals = self.chemicals
        indices = chemicals.equilibrium_indices(molar_data != 0)
        flow = molar_data[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        z = flow / netflow  
        chemicals_tuple = chemicals.tuple
        return z, [chemicals_tuple[i] for i in indices]
    
    @property
    def equilibrium_chemicals(self):
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        molar_data = self.molar_data.sum(0)
        indices = chemicals.equilibrium_indices(molar_data != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def z(self):
        molar_data = self.molar_data.sum(0)
        indices = self.chemicals.equilibrium_indices(molar_data != 0)
        flow = molar_data[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        return flow / netflow  
    
    ### Representation ###
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .material_array import nonzeros
        IDs = self.chemicals.IDs
        basic_info = f"{type(self).__name__}:\n"
        all_IDs, _ = nonzeros(self.chemicals.IDs, self.molar_flow.to_chemical_array(data=True))
        all_IDs = tuple(all_IDs)
        T_units, P_units, flow_units, N = self.display_units.get_units(T=T, P=P, flow=flow, N=N)
        basic_info += Stream._info_phaseTP(self, self.phases, T_units, P_units)
        len_ = len(all_IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs]
        maxlen = max(all_lengths + [8]) 

        # Get flow
        flow_dim = get_dimensionality(flow_units)
        if flow_dim == MolarFlow.units.dimensionality:
            flow = 'molar_flow'
        elif flow_dim == MassFlow.units.dimensionality:
            flow = 'mass_flow'
        elif flow_dim == VolumetricFlow.units.dimensionality:
            flow = 'volumetric_flow'
        else:
            raise DimensionError(f"dimensions for flow units must be in "
                                 f"molar, mass or volumetric flow rates, not '{flow_dim}'")
        first_line = ' ' + flow + f' ({flow_units}):'
        first_line_spaces = len(first_line)*" "
        flow = getattr(self, flow)

        # Set up chemical data for all phases
        phases_flowrates_info = ''
        for phase in self.phases:
            phase_data = flow.get_data(flow_units, phase, all_IDs)
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
    
    def __repr__(self):
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
        return (f"{type(self).__name__}({phases}, T={self.T:.2f}, "
                f"P={self.P:.6g}{price}{phase_data})")
    
del shared_properties