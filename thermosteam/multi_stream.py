# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from .stream import Stream, assert_same_chemicals
from .phase_container import new_phase_container
from .base.units_of_measure import get_dimensionality
from .thermal_condition import ThermalCondition
from .material_array import MolarFlow, MassFlow, VolumetricFlow, ChemicalMolarFlow
from .exceptions import DimensionError
from .settings import settings
from .equilibrium import VLE, BubblePoint, DewPoint

__all__ = ('MultiStream', )

class MultiStream:
    __slots__ = ('_molar_flow', '_TP', '_thermo', '_streams', 'price', '_vle')
    
    display_units = Stream.display_units
    
    def __init__(self, flow=(), T=298.15, P=101325., phases='lg', units=None,
                 thermo=None, price=None, **phase_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_flow(flow, phases, thermo.chemicals, phase_flows)
        self._streams = {}
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
        
    def get_property(self, name, units=None):
        value = getattr(self, name)
        if units:
            mixture_property = getattr(self.mixture, name)
            factor = mixture_property.units.to(units)
            return value * factor
        else:
            return value
    
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
    def TP(self):
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
    def phases(self):
        return self._molar_flow._phases
    
    @property
    def molar_data(self):
        return self._molar_flow._data
    @property
    def mass_data(self):
        return self.mass_flow._data
    @property
    def volumetric_data(self):
        return self.volumetric_flow._data
    
    @property
    def molar_flow(self):
        return self._molar_flow
    @property
    def mass_flow(self):
        return self._molar_flow.by_mass()
    @property
    def volumetric_flow(self):
        return self._molar_flow.by_volume(self._TP)
    
    ### Net flow properties ###
    
    @property
    def cost(self):
        return self.price * self.net_mass_flow
    
    @property
    def net_molar_flow(self):
        return self.molar_data.sum()
    @property
    def net_mass_flow(self):
        return (self.chemicals.MW * self.molar_data).sum()
    @property
    def net_volumetric_flow(self):
        return self.mixture.xV_at_TP(self.molar_flow.phase_data, self._TP).sum()

    @property
    def H(self):
        return self.mixture.xH_at_TP(self.molar_flow.phase_data, self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self.molar_flow.phase_data, H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self.molar_flow.phase_data, self._TP)
    
    @property
    def Hf(self):
        return (self.chemicals.Hf * self.molar_data).sum()
    @property
    def Hc(self):
        return (self.chemicals.Hc * self.molar_data).sum()    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self._molar_flow['l'], self._TP)
    
    @property
    def C(self):
        return self.mixture.xCp_at_TP(self.molar_flow.phase_data, self._TP)
    
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
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return self.mixture.xV_at_TP(zip(self.phases, molar_data / net_molar_data), self._TP)
        else:
            return 0
    @property
    def kappa(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return self.mixture.xkappa_at_TP(zip(self.phases, molar_data / net_molar_data), self._TP)
        else:
            return 0
    @property
    def Cp(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return self.mixture.xCp_at_TP(zip(self.phases, molar_data / net_molar_data), self._TP)
        else:
            return 0
    @property
    def mu(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return self.mixture.xmu_at_TP(zip(self.phases, molar_data / net_molar_data), self._TP)
        else:
            return 0
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
            return self.mixture.xepsilon_at_TP(molar_flow / net, self._TP)
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
        
    def link_with(self, other, TP=True, flow=True, phase=True):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        other._molar_flow._data_cache.clear()
        if TP:
            self._TP = other._TP
        if flow:
            self._molar_flow._data = other._molar_flow._data
        if phase:
            self._molar_flow._phase = other._molar_flow._phase
            
    def unlink(self):
        self._molar_flow._data_cache.clear()
        self._TP = self._TP.copy()
        self._molar_flow._data = self._molar_flow._data.copy()
        self._molar_flow._phase = new_phase_container(self._molar_flow._phase)
    
    def copy_like(self, other):
        self.molar_data[:] = other.molar_data
        self.TP.copy_like(other)
    
    def copy(self):
        cls = self.__class__
        new = cls.__new__(cls)
        new._thermo = self._thermo
        new._molar_flow = self._molar_flow.copy()
        new._TP = self._TP.copy()
        return new
    
    def empty(self):
        self.molar_data[:] = 0
    
    ### Equilibrium ###
    @property
    def vle(self):
        return VLE(self._molar_flow, self._TP, thermo=self._thermo)
    
    @property
    def equilibrium_chemicals(self):
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.equilibrium_indices(self.molar_data.sum(0) != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def equilibrium_composition(self):
        molar_data = self.molar_data
        indices = self.chemicals.equilibrium_indices(molar_data != 0)
        flow = molar_data[:, indices].sum(0)
        netflow = flow.sum()
        assert netflow, "no equlibrium chemicals present"
        return flow / netflow  
    
    @property
    def bubble_point(self):
        bp = BubblePoint(self.equilibrium_chemicals, self._thermo)
        return bp
    
    @property
    def dew_point(self):
        bp = DewPoint(self.equilibrium_chemicals, self._thermo)
        return bp
        
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
    
    def __repr__(self):
        from .utils import repr_kwarg, repr_couples
        IDs = self.chemicals.IDs
        phase_data = []
        for phase, data in self.molar_flow.phase_data:
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