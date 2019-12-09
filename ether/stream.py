# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
from .material_array import nonzeros
from .base.units_of_measure import get_dimensionality, _Q
from .base.display_units import DisplayUnits
from .exceptions import DimensionError
from .settings import settings
from .material_array import ChemicalMolarFlow, ChemicalMassFlow, ChemicalVolumetricFlow
from .thermal_condition import ThermalCondition
from .utils import Cache

__all__ = ('ChemicalStream',)


# %%

class ChemicalStream:
    __slots__ = ('_molar_flow', '_mass_flow_cache', '_volumetric_flow_cache',
                 '_thermal_condition', '_thermo', 'price')
    

    #: [DisplayUnits] Units of measure for IPython display
    display_units = DisplayUnits(T='K', P='Pa',
                                 flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                 N=5)

    def __init__(self, flow=(), phase='l', T=298.15, P=101325., units=None,
                 price=0., thermo=None, **chemical_flows):
        self._thermal_condition = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_flow(flow, phase, thermo.chemicals, chemical_flows)
        if units:
            dimensionality = get_dimensionality(units)
            if dimensionality == ChemicalMolarFlow.units.dimensionality:
                self.molar_flow.set_data(self.molar_flow._data, units)
            elif dimensionality == ChemicalMassFlow.units.dimensionality:
                self.mass_flow.set_data(self.molar_flow._data, units)
            elif dimensionality == ChemicalVolumetricFlow.units.dimensionality:
                self.volumetric_flow.set_data(self.molar_flow._data, units)
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dimensionality}'")
    
    def _load_flow(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if flow:
            assert not chemical_flows, ("may specify either 'flow' or "
                                        "'chemical_flows', but not both")
            molar_flow = ChemicalMolarFlow.from_data(flow, phase, chemicals)
        elif chemical_flows:
            molar_flow = ChemicalMolarFlow(chemicals, phase, **chemical_flows)
        else:
            molar_flow = ChemicalMolarFlow.blank(phase, chemicals)
        self._molar_flow = molar_flow
        self._mass_flow_cache = Cache(molar_flow.as_chemical_mass_flow)
        self._volumetric_flow_cache = Cache(molar_flow.as_chemical_volumetric_flow,
                                            (self._thermal_condition,))

    def get_flow_property(self, name, units=None):
        mixture_property = getattr(self._thermo.mixture, name)
        value = self._get_flow_property(mixture_property)
        return value * mixture_property.units.to(units) if units else value
    
    def get_composition_property(self, name, units=None):
        mixture_property = getattr(self._thermo.mixture, name)
        value = self._get_composition_property(mixture_property)
        return value * mixture_property.units.to(units) if units else value
    
    def get_single_phase_flow_property(self, name, units=None):
        mixture_property = getattr(self._thermo.mixture, name)
        value = self._get_single_phase_flow_property(mixture_property)
        return value * mixture_property.units.to(units) if units else value
    
    def get_single_phase_composition_property(self, name, units=None):
        mixture_property = getattr(self._thermo.mixture, name)
        value = self._get_single_phase_composition_property(mixture_property)
        return value * mixture_property.units.to(units) if units else value
    
    def _get_flow_property(self, mixture_property):
        return mixture_property.at_thermal_condition(self.phase,
                                                     self.molar_data,
                                                     self.thermal_condition)
    
    def _get_composition_property(self, mixture_property):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  mixture_property.at_thermal_condition(self.phase,
                                                          molar_data / net_molar_data,
                                                          self.thermal_condition)
        else:
            return 0
    
    def _get_single_phase_flow_property(self, mixture_property):
        return mixture_property.at_thermal_condition(self.molar_data,
                                                     self.thermal_condition)
    
    def _get_single_phase_composition_property(self, mixture_property):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  mixture_property.at_thermal_condition(molar_data / net_molar_data,
                                                          self.thermal_condition)
        else:
            return 0
    
    ### Stream data ###
    
    @property
    def thermo(self):
        return self._thermo
    
    @property
    def thermal_condition(self):
        return self._thermal_condition

    @property
    def T(self):
        return self._thermal_condition.T
    @T.setter
    def T(self, T):
        self._thermal_condition.T = T
    
    @property
    def P(self):
        return self._thermal_condition.P
    @P.setter
    def P(self, P):
        self._thermal_condition.P = P
    
    @property
    def phase(self):
        return self._molar_flow.phase
    @phase.setter
    def phase(self, phase):
        self._molar_flow.phase = phase
    
    @property
    def molar_data(self):
        return self._molar_flow._data
    @property
    def molar_flow(self):
        return self._molar_flow
    @property
    def mass_flow(self):
        return self._mass_flow_cache()
    @property
    def volumetric_flow(self):
        return self._volumetric_flow_cache()
    
    ### Net flow properties ###
    
    @property
    def net_molar_flow(self):
        return self.molar_data.sum()
    @property
    def net_mass_flow(self):
        return (self._thermo.chemicals.MW * self.molar_data).sum()
    @property
    def net_volumetric_flow(self):
        return self._get_flow_property(self._thermo.mixture.V)
    
    @property
    def H(self):
        return self._get_flow_property(self._thermo.mixture.H)
    @H.setter
    def H(self, H):
        self.T = self._thermo.mixture.H.solve_T(self.phase, self.molar_data, H, self.T)
    
    @property
    def S(self):
        return self._get_flow_property(self._thermo.mixture.S)
    
    @property
    def Hf(self):
        return (self._thermo.chemicals.Hf * self.molar_data).sum()
    @property
    def Hc(self):
        return (self._thermo.chemicals.Hc * self.molar_data).sum()    
    @property
    def Hvap(self):
        return self._get_single_phase_flow_property(self._thermo.mixture.Hvap)
    
    @property
    def C(self):
        return self._get_flow_property(self._thermo.mixture.Cp)
    
    ### Composition properties ###
    
    @property
    def V(self):
        return self._get_composition_property(self._thermo.mixture.V)
    @property
    def k(self):
        return self._get_composition_property(self._thermo.mixture.k)
    @property
    def Cp(self):
        return self._get_composition_property(self._thermo.mixture.Cp)
    @property
    def mu(self):
        return self._get_composition_property(self._thermo.mixture.mu)
    @property
    def sigma(self):
        return self._get_single_phase_composition_property(self._thermo.mixture.sigma)
    @property
    def epsilon(self):
        return self._get_single_phase_composition_property(self._thermo.mixture.epsilon)
    
    ### Representation ###
    
    def _info_phaseTP(self, phases, T_units, P_units):
        T = _Q(self.T,'K').to(T_units).magnitude
        P = _Q(self.P, 'Pa').to(P_units).magnitude
        return f" phase: '{phases}', T: {T:.5g} {T_units}, P: {P:.6g} {P_units}\n"
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        basic_info = f"{type(self).__name__}:\n"
        IDs = self._thermo.chemicals.IDs
        data = self.molar_flow.data
        IDs, data = nonzeros(IDs, data)
        IDs = tuple(IDs)
        T_units, P_units, flow_units, N = self.display_units.get_units(T=T, P=P, flow=flow, N=N)
        basic_info += self._info_phaseTP(self.phase, T_units, P_units)
        len_ = len(IDs)
        if len_ == 0:
            return basic_info + ' flow: 0' 
        
        # Start of third line (flow rates)
        flow_dim = get_dimensionality(flow_units)
        if flow_dim == ChemicalMolarFlow.units.dimensionality:
            flow = 'molar_flow'
        elif flow_dim == ChemicalMassFlow.units.dimensionality:
            flow = 'mass_flow'
        elif flow_dim == ChemicalVolumetricFlow.units.dimensionality:
            flow = 'volumetric_flow'
        else:
            raise DimensionError(f"dimensions for flow units must be in "
                                 f"molar, mass or volumetric flow rates, not '{flow_dim}'")
        beginning = f' {flow} ({flow_units}): '
            
        # Remaining lines (all flow rates)
        new_line_spaces = len(beginning) * ' '
        flow_array = getattr(self, flow).get_data(flow_units, IDs)        
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