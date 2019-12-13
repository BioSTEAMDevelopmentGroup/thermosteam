# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 02:34:56 2019

@author: yoelr
"""
from .base.units_of_measure import get_dimensionality, _Q
from .base.display_units import DisplayUnits
from .exceptions import DimensionError
from .settings import settings
from .material_array import ChemicalMolarFlow, ChemicalMassFlow, ChemicalVolumetricFlow
from .thermal_condition import ThermalCondition
from .phase_container import new_phase_container
from .equilibrium import BubblePoint, DewPoint

__all__ = ('Stream',)


# %% Utilities

def assert_same_chemicals(stream, others):
    chemicals = stream.chemicals
    assert all([chemicals == i.chemicals for i in others]), "chemicals must match to mix streams"


# %%

class Stream:
    __slots__ = ('_molar_flow', '_TP', '_thermo', 'price')
    

    #: [DisplayUnits] Units of measure for IPython display
    display_units = DisplayUnits(T='K', P='Pa',
                                 flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                 N=5)

    def __init__(self, flow=(), phase='l', T=298.15, P=101325., units=None,
                 price=0., thermo=None, **chemical_flows):
        self._TP = ThermalCondition(T, P)
        self._thermo = thermo = thermo or settings.get_thermo(thermo)
        self._load_flow(flow, phase, thermo.chemicals, chemical_flows)
        self.price = price
        if units:
            dimensionality = get_dimensionality(units)
            if dimensionality == ChemicalMolarFlow.units.dimensionality:
                self.molar_flow.set_data(self.molar_data, units)
            elif dimensionality == ChemicalMassFlow.units.dimensionality:
                self.mass_flow.set_data(self.molar_data, units)
            elif dimensionality == ChemicalVolumetricFlow.units.dimensionality:
                self.volumetric_flow.set_data(self.molar_data, units)
            else:
                raise DimensionError(f"dimensions for flow units must be in molar, "
                                     f"mass or volumetric flow rates, not '{dimensionality}'")
    
    def _load_flow(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if flow is ():
            if chemical_flows:
                molar_flow = ChemicalMolarFlow(phase, chemicals=chemicals, **chemical_flows)
            else:
                molar_flow = ChemicalMolarFlow.blank(phase, chemicals)
        else:
            assert not chemical_flows, ("may specify either 'flow' or "
                                        "'chemical_flows', but not both")
            if isinstance(flow, ChemicalMolarFlow):
                molar_flow = flow 
                molar_flow.phase = phase
            else:
                molar_flow = ChemicalMolarFlow.from_data(flow, phase, chemicals)
        self._molar_flow = molar_flow

    ### Property getters ###

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
        return self._molar_flow.phase
    @phase.setter
    def phase(self, phase):
        self._molar_flow.phase = phase
    
    @property
    def molar_data(self):
        return self._molar_flow._data
    @molar_data.setter
    def molar_data(self, value):
        if self.molar_data is not value:
            raise AttributeError("cannot replace attribute")
    
    @property
    def mass_data(self):
        return self.mass_flow._data
    @mass_data.setter
    def mass_data(self, value):
        if self.mass_data is not value:
            raise AttributeError("cannot replace attribute")
    
    @property
    def volumetric_data(self):
        return self.volumetric_flow._data
    @volumetric_data.setter
    def volumetric_data(self, value):
        if self.volumetric_data is not value:
            raise AttributeError("cannot replace attribute")
        
    @property
    def molar_flow(self):
        return self._molar_flow
    @molar_flow.setter
    def molar_flow(self, value):
        if self._molar_flow is not value:
            raise AttributeError("cannot replace attribute")
    
    @property
    def mass_flow(self):
        return self._molar_flow.by_mass()
    @mass_flow.setter
    def mass_flow(self, value):
        if self.mass_flow is not value:
            raise AttributeError("cannot replace attribute")
    
    @property
    def volumetric_flow(self):
        return self._molar_flow.by_volume(self._TP)
    @volumetric_flow.setter
    def volumetric_flow(self, value):
        if self.volumetric_flow is not value:
            raise AttributeError("cannot replace attribute")
    
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
        return self.mixture.V_at_TP(self.phase, self.molar_data, self._TP)
    
    @property
    def H(self):
        return self.mixture.H_at_TP(self.phase, self.molar_data, self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.solve_T(self.phase, self.molar_data, H, self.T, self.P)

    @property
    def S(self):
        return self.mixture.S_at_TP(self.phase, self.molar_data, self._TP)
    
    @property
    def Hf(self):
        return (self.chemicals.Hf * self.molar_data).sum()
    @property
    def Hc(self):
        return (self.chemicals.Hc * self.molar_data).sum()    
    @property
    def Hvap(self):
        return self.mixture.Hvap_at_TP(self.molar_data, self._TP)
    
    @property
    def C(self):
        return self.mixture.Cp_at_TP(self.molar_data, self._TP)
    
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
        volumetric_flow = self.volumetric_data
        net_volumetric_flow = volumetric_flow.sum()
        return volumetric_flow / net_volumetric_flow if net_volumetric_flow else volumetric_flow
    
    @property
    def V(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.V_at_TP(self.phase, molar_data / net_molar_data, self._TP)
        else:
            return 0
    @property
    def kappa(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.kappa_at_TP(self.phase, molar_data / net_molar_data, self._TP)
        else:
            return 0
    @property
    def Cp(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.Cp_at_TP(self.phase, molar_data / net_molar_data, self._TP)
        else:
            return 0
    @property
    def mu(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.mu_at_TP(self.phase, molar_data / net_molar_data, self._TP)
        else:
            return 0
    @property
    def sigma(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.sigma_at_TP(molar_data / net_molar_data, self._TP)
        else:
            return 0
    @property
    def epsilon(self):
        molar_data = self.molar_data
        net_molar_data = molar_data.sum()
        if net_molar_data:
            return  self.mixture.epsilon_at_TP(molar_data / net_molar_data, self._TP)
        else:
            return 0
    
    ### Stream methods ###
    
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        isa = isinstance
        self.molar_data[:] = sum([i.molar_data if isa(i, Stream)
                                  else i.molar_data.sum(0) for i in others])
        self.H = sum([i.H for i in others])
    
    def split_to(self, s1, s2, split):
        molar_data = self.molar_data
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
        self.phase = other.phase
    
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
    def z_chemicals(self):
        molar_data = self.molar_data
        chemicals = self.chemicals
        indices = chemicals.equilibrium_indices(molar_data != 0)
        flow = molar_data[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        z = flow / netflow  
        chemicals_tuple = chemicals.tuple
        return z, [chemicals_tuple[i] for i in indices]
    
    @property
    def equilibrim_chemicals(self):
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.equilibrium_indices(self.molar_data != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def z(self):
        molar_data = self.molar_data
        indices = self.chemicals.equilibrium_indices(molar_data != 0)
        flow = molar_data[indices]
        netflow = flow.sum()
        assert netflow, "no equilibrium chemicals present"
        return flow / netflow  
    
    @property
    def bubble_point(self):
        bp = BubblePoint(self.equilibrim_chemicals, self._thermo)
        return bp
    
    @property
    def dew_point(self):
        bp = DewPoint(self.equilibrim_chemicals, self._thermo)
        return bp
    
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
    
    ### Representation ###
    
    def _info_phaseTP(self, phase, T_units, P_units):
        T = _Q(self.T,'K').to(T_units).magnitude
        P = _Q(self.P, 'Pa').to(P_units).magnitude
        s = '' if isinstance(phase, str) else 's'
        return f" phase{s}: {repr(phase)}, T: {T:.5g} {T_units}, P: {P:.6g} {P_units}\n"
    
    def _info(self, T, P, flow, N):
        """Return string with all specifications."""
        from .material_array import nonzeros
        basic_info = f"{type(self).__name__}:\n"
        IDs = self.chemicals.IDs
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
    
    def __repr__(self):
        from .utils import repr_kwargs, repr_kwarg
        chemical_flows = repr_kwargs(self.chemicals.IDs, self.molar_data)
        price = repr_kwarg('price', self.price)
        return (f"{type(self).__name__}(phase={repr(self.phase)}, T={self.T:.2f}, "
                f"P={self.P:.6g}{price}{chemical_flows})")
        
        