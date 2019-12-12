# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from .stream import Stream
from .base.units_of_measure import get_dimensionality
from .thermal_condition import ThermalCondition
from .material_array import MolarFlow, MassFlow, VolumetricFlow, ChemicalMolarFlow
from .exceptions import DimensionError
from .settings import settings

__all__ = ('MultiStream', )

class MultiStream:
    __slots__ = ('_molar_flow', '_TP', '_thermo', '_streams', 'price')
    
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
        return self.mixture.xV_at_TP(self.phase_data, self._TP).sum()

    @property
    def H(self):
        return self.mixture.xH_at_TP(self.phase_data, self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self.phase, self.molar_data, H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.xS_at_TP(self.phase_data, self._TP)
    
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
        return self._get_flow_property('xCp')
    
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
        volumetric_flow = self._get_flow_property(self.mixture.V)
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