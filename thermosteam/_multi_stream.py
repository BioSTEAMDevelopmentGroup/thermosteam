# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from ._stream import Stream
from ._thermal_condition import ThermalCondition
from .indexer import MolarFlowIndexer
from ._phase import phase_tuple
from . import equilibrium as eq
from . import utils
import numpy as np

__all__ = ('MultiStream', )

def get_phase_fraction(stream, phases):
    all_phases = stream.phases
    phase_fraction = 0.
    F_mol = stream.F_mol
    imol = stream.imol
    if not F_mol: return 0.
    for phase in phases:
        if phase in all_phases:
            phase_fraction += imol[phase].sum() 
    return phase_fraction / F_mol

class MultiStream(Stream):
    """
    Create a MultiStream object that defines material flow rates for multiple
    phases along with its thermodynamic state. Thermodynamic and transport
    properties of a stream are available as properties, while thermodynamic
    equilbrium (e.g. VLE, and bubble and dew points) are available as methods. 

    Parameters
    ----------
    ID : str, optional
        A unique identification. If ID is None, stream will not be registered.
        If no ID is given, stream will be registered with a unique ID.
    flow : 2d array, optional
        All flow rates corresponding to `phases` by row and chemical IDs by column.
    T : float
        Temperature [K]. Defaults to 298.15.
    P : float
        Pressure [Pa]. Defaults to 101325.
    phases : tuple['g', 'l', 's', 'L', 'S']
        Tuple denoting the phases present. Defaults to ('g', 'l').
    units : str, optional
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid). Defaults to 'kmol/hr'.
    price : float, optional
        Price per unit mass [USD/kg]. Defaults to 0.
    total_flow : float, optional
        Total flow rate.
    thermo : :class:`~thermosteam.Thermo`
        Thermodynamic equilibrium package. Defaults to `thermosteam.settings.get_thermo()`.
    **phase_flow : tuple[str, float]
        phase-(ID, flow) pairs.
    
    Examples
    --------
    Before creating streams, first set the chemicals:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    
    Create a multi phase stream, defining the thermodynamic condition and
    flow rates:
        
    >>> s1 = tmo.MultiStream(ID='s1',T=298.15, P=101325,
    ...                      l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
    >>> s1.show(flow='kg/hr') # Use the show method to select units of display
    MultiStream: s1
     phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
     flow (kg/hr): (l) Water    20
                       Ethanol  10
    
    The temperature and pressure are stored as attributes:
    
    >>> (s1.T, s1.P)
    (298.15, 101325.0)
    
    Unlike Stream objects, the `mol` attribute does not store data, it simply
    returns the total flow rate of each chemical. Setting an element of the
    array raises an error to prevent the wrong assumption that the data is
    linked:
    
    >>> s1.mol
    array([1.11 , 0.217])
    >>> s1.mol[0] = 1
    Traceback (most recent call last):
    ValueError: assignment destination is read-only
    
    All flow rates are stored in the `imol` attribute:
    
    >>> s1.imol.show() # Molar flow rates [kmol/hr]
    MolarFlowIndexer (kmol/hr):
     (l) Water     1.11
         Ethanol   0.2171
    >>> # Index a single chemical in the liquid phase
    >>> s1.imol['l', 'Water']
    1.1101687012358397
    >>> # Index multiple chemicals in the liquid phase
    >>> s1.imol['l', ('Ethanol', 'Water')]
    array([0.217, 1.11 ])
    >>> # Index the vapor phase
    >>> s1.imol['g']
    array([0., 0.])
    >>> # Index flow of chemicals summed across all phases
    >>> s1.imol['Ethanol', 'Water']
    array([0.217, 1.11 ])
    
    Note that overall chemical flows in MultiStream objects cannot be set like
    with Stream objects:
        
    >>> # s1.imol['Ethanol', 'Water'] = [1, 0]
    Traceback (most recent call last):
    IndexError: multiple phases present; must include phase key to set chemical data
    
    Chemical flows must be set by phase:

    >>> s1.imol['l', ('Ethanol', 'Water')] = [1, 0]
    
    The most convinient way to get and set flow rates is through the 
    `get_flow` and `set_flow` methods:
    
    >>> # Set flow
    >>> key = ('l', 'Water')
    >>> s1.set_flow(1, 'gpm', key)
    >>> s1.get_flow('gpm', key)
    1.0
    >>> # Set multiple flows
    >>> key = ('l', ('Ethanol', 'Water'))
    >>> s1.set_flow([10, 20], 'kg/hr', key)
    >>> s1.get_flow('kg/hr', key)
    array([10., 20.])
    
    Chemical flows across all phases can be retrieved if no phase is
    given:
        
    >>> s1.get_flow('kg/hr', ('Ethanol', 'Water'))
    array([10., 20.])
    
    However, setting chemical data requires the phase to be specified:
        
    >>> s1.set_flow([10, 20], 'kg/hr', ('Ethanol', 'Water'))
    Traceback (most recent call last):
    IndexError: multiple phases present; must include phase key to set chemical data
    
    Note that for both Stream and MultiStream objects, `mol`, `imol`, 
    and `get_flow` return chemical flows across all phases when given only 
    chemical IDs.
    
    Vapor-liquid equilibrium can be performed by setting 2 degrees of freedom
    from the following list:
    
    * T - Temperature [K]
    * P - Pressure [Pa]
    * V - Vapor fraction
    * H - Enthalpy [kJ/hr]
        
    >>> s1.vle(P=101325, T=365)
    
    Each phase can be accessed separately too:
    
    >>> s1['l'].show()
    Stream: 
     phase: 'l', T: 365 K, P: 101325 Pa
     flow (kmol/hr): Water    0.619
                     Ethanol  0.0238
    >>> s1['g'].show()
    Stream: 
     phase: 'g', T: 365 K, P: 101325 Pa
     flow (kmol/hr): Water    0.491
                     Ethanol  0.193
    
    Note that the phase cannot be changed:
    
    >>> s1['g'].phase = 'l'
    Traceback (most recent call last):
    AttributeError: phase is locked
    
    """
    __slots__ = ()
    def __init__(self, ID="", flow=(), T=298.15, P=101325., phases=None, 
                 units=None, price=0, total_flow=None, thermo=None, 
                 characterization_factors=None, **phase_flows):
        self.characterization_factors = {} if characterization_factors is None else {}
        self._thermal_condition = ThermalCondition(T, P)
        thermo = self._load_thermo(thermo)
        chemicals = thermo.chemicals
        self.price = price
        phases = set(phase_flows).union(['l', 'g']) if phases is None else phases
        if units:
            name, factor = self._get_flow_name_and_factor(units)
            if name == 'mass':
                group_wt_compositions = chemicals._group_wt_compositions
                for phase, chemical_flows in phase_flows.items():
                    new_chemical_flows = []
                    for item in chemical_flows:
                        cID, value = item
                        if cID in group_wt_compositions:
                            compositions = group_wt_compositions[cID]
                            chemical_group = chemicals[cID]
                            for i in range(len(chemical_group)):
                                item = (chemical_group[i]._ID, value * compositions[i])
                                new_chemical_flows.append(item)
                        else:
                            new_chemical_flows.append(item)
                    phase_flows[phase] = new_chemical_flows
            elif name == 'vol':
                group_wt_compositions = chemicals._group_wt_compositions
                for chemical_flows in phase_flows.values():
                    for item in chemical_flows:
                        cID, value = item
                        if cID in group_wt_compositions:
                            raise ValueError(f"cannot set volumetric flow by chemical group '{i}'")
            self._init_indexer(flow, phases, chemicals, phase_flows)
            flow = getattr(self, 'i' + name)
            material_data = self._imol._data / factor
            if total_flow: material_data *= total_flow / material_data.sum()
            flow._data[:] = material_data
        else:
            self._init_indexer(flow, phases, chemicals, phase_flows)
            if total_flow: self._imol._data *= total_flow / self.F_mol
        self._sink = self._source = None
        self.reset_cache()
        self._register(ID)
        self._islinked = False
        self._user_equilibrium = None
        
    def reset_flow(self, total_flow=None, units=None, phases=None, **phase_flows):
        """
        Convinience method for resetting flow rate data.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 1)])
        >>> s1.reset_flow(g=[('Ethanol', 1)], phases='lgs', units='kg/hr', total_flow=2)
        >>> s1.show('cwt')
        MultiStream: s1
         phases: ('g', 'l', 's'), T: 298.15 K, P: 101325 Pa
         composition: (g) Ethanol  1
                          -------  2 kg/hr
        
        """
        imol = self._imol
        imol.empty()
        self.phases = set(phase_flows).union(['l', 'g']) if phases is None else phases
        if phase_flows:
            for phase, data in phase_flows.items():
                keys, values = zip(*data)
                self.set_flow(values, units, (phase, keys))
        if total_flow:
            self.set_total_flow(total_flow, units)
        
    def _init_indexer(self, flow, phases, chemicals, phase_flows):
        if flow == ():
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
    
    def _get_property_cache(self, name, flow=False):
        property_cache = self._property_cache
        thermal_condition = self._thermal_condition
        imol = self._imol
        data = imol._data
        total = data.sum()
        if total == 0.: return 0.
        composition = data / total
        literal = (imol._phases, thermal_condition._T, thermal_condition._P)
        last_literal, last_composition, last_total = self._property_cache_key
        if literal == last_literal and (composition - last_composition).all():
            prop = property_cache.get(name)
            if not prop: return prop
            if flow:
                return prop * total / last_total
            else:
                return prop
        else:
            self._property_cache_key = (literal, composition, total)
            property_cache.clear()
            return None
    
    def reset_cache(self):
        """Reset cache regarding equilibrium methods."""
        super().reset_cache()
        if hasattr(self, '_streams'): 
            for i in self._streams.values(): i.reset_cache()
        else:
            self._streams = {}
        self._vle_cache = eq.VLECache(self._imol,
                                      self._thermal_condition, 
                                      self._thermo,
                                      self._bubble_point_cache,
                                      self._dew_point_cache)
        self._lle_cache = eq.LLECache(self._imol,
                                      self._thermal_condition, 
                                      self._thermo)
        self._sle_cache = eq.SLECache(self._imol,
                                      self._thermal_condition, 
                                      self._thermo)
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._ID = stream._sink = stream._source = None
            stream._islinked = False
            stream._imol = self._imol.get_phase(phase)
            stream._thermal_condition = self._thermal_condition
            stream._thermo = self._thermo
            stream._bubble_point_cache =  self._bubble_point_cache
            stream._dew_point_cache = self._dew_point_cache
            stream._property_cache = {}
            stream.characterization_factors = {}
            stream._property_cache_key = None, None, None
            stream._user_equilibrium = None
            streams[phase] = stream
        return stream
    
    def __iter__(self):
        for i in self.phases: yield self[i]
    
    def __len__(self):
        return len(self.phases)
    
    def shares_flow_rate_with(self, other):
        """
        Return whether other stream shares data with this one.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> s1 = tmo.MultiStream('s1')
        >>> other = s1.flow_proxy()
        >>> s1.shares_flow_rate_with(other)
        True
        >>> s1 = tmo.MultiStream('s1', phases=('l', 'g'))
        >>> s1.shares_flow_rate_with(s1['g'])
        True
        >>> s2 = tmo.MultiStream('s2', phases=('l', 'g'))
        >>> s2.shares_flow_rate_with(s1['g'])
        False
        >>> s1.shares_flow_rate_with(s2)
        False
        
        """
        imol = self._imol
        other_imol = other._imol
        if imol.__class__ is other_imol.__class__ and imol._data is other_imol._data:
            shares_data = True
        elif isinstance(other, Stream):
            phase = other.phase
            substreams = self._streams
            if phase in substreams:
                substream = substreams[phase]
                shares_data = other.shares_flow_rate_with(substream)
            else:
                shares_data = False
        else:
            shares_data = False
        return shares_data
    
    ### Property getters ###
    
    def get_flow(self, units, key=...):
        """
        Return an array of flow rates in requested units.
        
        Parameters
        ----------
        units : str
            Units of measure.
        key : tuple(phase, IDs), phase, or IDs
            * phase: str, ellipsis, or missing.
            * IDs: str, tuple(str), ellipisis, or missing.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s1.get_flow('kg/hr', ('l', 'Water'))
        20.0

        """
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        return factor * indexer[key]
    
    def set_flow(self, data, units, key=...):
        """
        Set flow rates in given units.

        Parameters
        ----------
        data : 1d ndarray or float
            Flow rate data.
        units : str
            Units of measure.
        key : tuple(phase, IDs), phase, or IDs
            * phase: str, ellipsis, or missing.
            * IDs: str, tuple(str), ellipisis, or missing.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s1.set_flow(10, 'kg/hr', ('l', 'Water'))
        >>> s1.get_flow('kg/hr', ('l', 'Water'))
        10.0

        """
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        indexer[key] = np.asarray(data, dtype=float) / factor
    
    ### Stream data ###
    
    @property
    def phases(self):
        """tuple[str] All phases avaiable."""
        return self._imol._phases
    @phases.setter
    def phases(self, phases):
        if len(phases) == 1:
            self.phase = phases[0]
        phases = phase_tuple(phases)
        if phases != self.phases:
            self._imol = self._imol.to_material_indexer(phases)
            self.reset_cache()
    
    ### Flow properties ###
            
    @property
    def mol(self):
        """[Array] Chemical molar flow rates (total of all phases)."""
        mol = self._imol._data.sum(0)
        mol.setflags(0)
        return mol
    @property
    def mass(self):
        """[Array] Chemical mass flow rates (total of all phases)."""
        mass = self.mol * self.chemicals.MW
        mass.setflags(0)
        return mass
    @property
    def vol(self):
        """[Array] Chemical volumetric flow rates (total of all phases)."""
        vol = self.ivol._data.sum(0)
        vol.setflags(0)
        return vol
    
    ### Net flow properties ###
    
    @property
    def H(self):
        """[float] Enthalpy flow rate in kJ/hr."""
        H = self._get_property_cache('H', True)
        if H is None:
            self._property_cache['H'] = H = self.mixture.xH(self._imol, *self._thermal_condition)
        return H
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self._imol, H, *self._thermal_condition)

    @property
    def S(self):
        """[float] Absolute entropy flow rate in kJ/hr."""
        S = self._get_property_cache('S', True)
        if S is None:
            self._property_cache['S'] = S = self.mixture.xS(self._imol, *self._thermal_condition)
        return S
    @property
    def C(self):
        """[float] Heat capacity flow rate in kJ/hr."""
        C = self._get_property_cache('C', True)
        if C is None:
            self._property_cache['C'] = C = self.mixture.xCn(self._imol, self.T)
        return C
    @property
    def F_vol(self):
        """[float] Total volumetric flow rate in m3/hr."""
        return 1000. * self.V * self.F_mol
    @F_vol.setter
    def F_vol(self, value):
        F_vol = self.F_vol
        if not F_vol: raise AttributeError("undefined composition; cannot set flow rate")
        self.ivol._data[:] *= value / F_vol / 1000.
    
    @property
    def Hvap(self):
        """[float] Enthalpy of vaporization flow rate in kJ/hr."""
        mol = self._imol['l']
        T = self._thermal_condition._T
        Hvap = self._get_property_cache('Hvap')
        if Hvap is None:
            self._property_cache['Hvap'] = Hvap = sum([
                i*j.Hvap(T) for i,j in zip(mol, self.chemicals)
                if i and not j.locked_state
            ])
        return Hvap
    
    ### Composition properties ###
    
    @property
    def vapor_fraction(self):
        """Molar vapor fraction."""
        return get_phase_fraction(self, 'gG')
    @property
    def liquid_fraction(self):
        """Molar liquid fraction."""
        return get_phase_fraction(self, 'lL')
    @property
    def solid_fraction(self):
        """Molar solid fraction."""
        return get_phase_fraction(self, 'sS')
    
    @property
    def V(self):
        """[float] Molar volume [m^3/mol]."""
        V = self._get_property_cache('V')
        if V is None:
            self._property_cache['V'] = V = self.mixture.xV(self._imol.iter_composition(), *self._thermal_condition)
        return V
    @property
    def kappa(self):
        """[float] Thermal conductivity [W/m/k]."""
        kappa = self._get_property_cache('kappa')
        if kappa is None:
            self._property_cache['kappa'] = kappa = self.mixture.xkappa(self._imol.iter_composition(), *self._thermal_condition)
        return kappa      
    @property
    def Cn(self):
        """[float] Molar heat capacity [J/mol/K]."""
        Cn = self._get_property_cache('Cn')
        if Cn is None:
            self._property_cache['Cn'] = Cn = self.mixture.xCn(self._imol.iter_composition(), self.T)
        return Cn
    @property
    def mu(self):
        """[float] Hydrolic viscosity [Pa*s]."""
        mu = self._get_property_cache('mu')
        if mu is None:
            self._property_cache['mu'] = mu = self.mixture.xmu(self._imol.iter_composition(), *self._thermal_condition)
        return mu
    @property
    def sigma(self):
        """[float] Surface tension [N/m]."""
        mol = self._imol['l']
        sigma = self._get_property_cache('sigma')
        if sigma is None:
            self._property_cache['sigma'] = sigma = self.mixture.sigma(mol / mol.sum(), *self._thermal_condition)
        return sigma
    @property
    def epsilon(self):
        """[float] Relative permittivity [-]."""
        mol = self._imol['l']
        epsilon = self._get_property_cache('epsilon')
        if epsilon is None:
            self._property_cache['epsilon'] = epsilon = self.mixture.epsilon(mol / mol.sum(), *self._thermal_condition)
        return epsilon
        
    ### Methods ###
    
    def split_to(self, s1, s2, split, energy_balance=True):
        """
        Split molar flow rate from this stream to two others given
        the split fraction or an array of split fractions.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> s = tmo.Stream('s', Water=20, Ethanol=10, units='kg/hr')
        >>> s.phases = ('l', 'g')
        >>> s1 = tmo.Stream('s1')
        >>> s2 = tmo.Stream('s2')
        >>> split = chemicals.kwarray(dict(Water=0.5, Ethanol=0.1))
        >>> s.split_to(s1, s2, split)
        >>> s1.show(flow='kg/hr')
        MultiStream: s1
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    10
                           Ethanol  1
        
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    10
                           Ethanol  9
        
        
        """
        phases = self.phases
        if energy_balance or isinstance(s1, MultiStream) or isinstance(s2, MultiStream):
            s1.phases = phases
            s2.phases = phases
            for phase in phases: self[phase].split_to(s1[phase], s2[phase], split)
        else:
            Stream.split_to(self, s1, s2, split)
            return
        if energy_balance:
            tc1 = s1._thermal_condition
            tc2 = s2._thermal_condition
            tc = self._thermal_condition
            tc1._T = tc2._T = tc._T
            tc1._P = tc2._P = tc._P
    
    def copy_flow(self, other, phase=..., IDs=..., *, remove=False, exclude=False):
        """
        Copy flow rates of another stream to self.
        
        Parameters
        ----------
        other : Stream
            Flow rates will be copied from here.
        phase : str or Ellipsis
        IDs=None : iterable[str], defaults to all chemicals.
            Chemical IDs. 
        remove=False: bool, optional
            If True, copied chemicals will be removed from `stream`.
        exclude=False: bool, optional
            If True, exclude designated chemicals when copying.
        
        Notes
        -----
        Works just like <Stream>.copy_flow, but the phase must be specified.
        
        Examples
        --------
        Initialize streams:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s2 = tmo.MultiStream('s2')
        
        Copy all flows:
        
        >>> s2.copy_flow(s1)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    20
                           Ethanol  10
        
        Reset and copy just water flow:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, IDs='Water')
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water  20
        
        Reset and copy all flows except water:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, IDs='Water', exclude=True)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Ethanol  10
        
        Cut and paste flows:
        
        >>> s2.copy_flow(s1, remove=True)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    20
                           Ethanol  10
        >>> s1.show()
        MultiStream: s1
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow: 0
         
        The other stream can also be a single phase stream (doesn't have to be a MultiStream object):
                                                            
        Initialize streams:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.MultiStream('s2')
        
        Copy all flows:
        
        >>> s2.copy_flow(s1)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    20
                           Ethanol  10
        
        Reset and copy just water flow:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, IDs='Water')
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water  20
        
        Reset and copy all flows except water:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, IDs='Water', exclude=True)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Ethanol  10
        
        Cut and paste flows:
        
        >>> s2.copy_flow(s1, remove=True)
        >>> s2.show(flow='kg/hr')
        MultiStream: s2
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow (kg/hr): (l) Water    20
                           Ethanol  10
        >>> s1.show()
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow: 0
        
        """
        if self.chemicals is not other.chemicals:
            raise ValueError('other stream must have the same chemicals defined to copy flow')
        other_ismultistream = isinstance(other, MultiStream)
        if other_ismultistream: self.phases = [*self.phases, *other.phases]
        IDs_index = self.chemicals.get_index(IDs)
        phase_index = self.imol.get_phase_index(phase)
        data = self.imol.data
        other_data = other.imol.data
        if exclude:
            data = self.imol.data
            other_data = other.imol.data
            original_data = data.copy()
            if other_ismultistream:
                data[:] = other_data
                data[phase_index, IDs_index] = original_data[phase_index, IDs_index]
                if remove:
                    excluded_data = other_data[phase_index, IDs_index]
                    other_data[:] = 0.
                    other_data[phase_index, IDs_index] = excluded_data
            else:
                other_phase_index = self.imol.get_phase_index(other.phase)
                data[other_phase_index, :] = other_data
                data[phase_index, IDs_index] = original_data[phase_index, IDs_index]
                if remove and (phase is ... or phase_index == other_phase_index):
                    excluded_data = other_data[IDs_index]
                    other_data[:] = 0.
                    other_data[IDs_index] = excluded_data   
        elif other_ismultistream:
            data[phase_index, IDs_index] = other_data[phase_index, IDs_index]
            if remove: other_data[phase_index, IDs_index] = 0.
        else:
            data[:] = 0.
            other_phase_index = self.imol.get_phase_index(other.phase)
            if phase is ... or phase_index == other_phase_index:
                data[other_phase_index, IDs_index] = other_data[IDs_index]
                if remove: other_data[IDs_index] = 0.
    
    def get_normalized_mol(self, IDs):
        """
        Return normalized molar fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals to be normalized.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='kmol/hr')
        >>> s1.get_normalized_mol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_vol(self, IDs):
        """
        Return normalized mass fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals to be normalized.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='m3/hr')
        >>> s1.get_normalized_vol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.ivol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_mass(self, IDs):
        """
        Return normalized mass fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals to be normalized.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='kg/hr')
        >>> s1.get_normalized_mass(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imass[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_molar_composition(self, IDs):
        """
        Return molar composition of given chemicals.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='kmol/hr')
        >>> s1.get_molar_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        return self.imol[..., IDs].sum(0)/self.F_mol
    
    def get_mass_composition(self, IDs):
        """
        Return mass composition of given chemicals.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='kg/hr')
        >>> s1.get_mass_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        return self.imass[..., IDs].sum(0)/self.F_mass
    
    def get_volumetric_composition(self, IDs):
        """
        Return volumetric composition of given chemicals.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='m3/hr')
        >>> s1.get_volumetric_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        return self.ivol[..., IDs].sum(0)/self.F_vol
    
    def get_concentration(self, phase, IDs):
        """
        Return concentration of given chemicals in kmol/m3.

        Parameters
        ----------
        IDs : tuple[str]
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='m3/hr')
        >>> s1.get_concentration('l', ('Water', 'Ethanol'))
        array([27.672,  4.265])

        """
        return self.imol[phase, IDs] / self.F_vol
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        """[VLE] An object that can perform vapor-liquid equilibrium on the stream."""
        phases = self.phases
        if 'l' not in phases or 'g' not in phases: 
            self.phases = [*phases, 'l', 'g']
        return self._vle_cache.retrieve()
    @property
    def lle(self):
        """[LLE] An object that can perform liquid-liquid equilibrium on the stream."""
        phases = self.phases
        if 'l' not in phases or 'L' not in phases: 
            self.phases = [*phases, 'l', 'L']
        return self._lle_cache.retrieve()
    @property
    def sle(self):
        """[SLE] An object that can perform solid-liquid equilibrium on the stream."""
        phases = self.phases
        if 's' not in phases or 'l' not in phases: 
            self.phases = [*phases, 's', 'l']
        return self._sle_cache.retrieve()
    
    ### Casting ###
    
    def as_stream(self):
        """Convert MultiStream to Stream."""
        phase = self.phase
        N_phase = len(phase)
        if N_phase == 1:
            self.phase = phase
        elif N_phase == 0:
            self.phase = self.phases[0]
        else:
            raise RuntimeError('multiple phases present; cannot convert to single phase stream')
    
    @property
    def phase(self):
        imol = self._imol
        return ''.join([phases[0] for phases in ('gG', 'lL', 'sS') if not imol.phases_are_empty(phases)])
    @phase.setter
    def phase(self, phase):
        if len(phase) > 1: self.phases = phase
        self._imol = self._imol.to_chemical_indexer(phase)
        self._streams.clear()
        self.__class__ = Stream
    
    ### Representation ###
    
    def _info(self, layout, T, P, flow, composition, N, IDs):
        """Return string with all specifications."""
        flow, composition, N = self._translate_layout(layout, flow, composition, N)
        from .indexer import nonzeros
        if not IDs: IDs = self.chemicals.IDs
        basic_info = self._basic_info()
        all_IDs, _ = nonzeros(IDs, self.imol[IDs])
        all_IDs = tuple(all_IDs)
        display_units = self.display_units
        T_units = T or display_units.T
        P_units = P or display_units.P
        flow_units = flow or display_units.flow
        N_max = display_units.N if N is None else N
        if composition is None: composition = display_units.composition
        basic_info += Stream._info_phaseTP(self, self.phases, T_units, P_units)
        N_all_IDs = len(all_IDs)
        if N_all_IDs == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs]
        maxlen = max(all_lengths) 

        name, factor = self._get_flow_name_and_factor(flow_units)
        indexer = getattr(self, 'i' + name)
        if composition:
            first_line = " composition:"
        else:
            first_line = f' flow ({flow_units}):'
        first_line_spaces = len(first_line)*" "

        # Set up chemical data for all phases
        phases_flow_rates_info = ''
        for phase in self.phases:
            phase_data = factor * indexer[phase, all_IDs] 
            IDs, data = nonzeros(all_IDs, phase_data)
            if not IDs: continue
            if composition:
                total_flow = data.sum()
                data = data/total_flow
        
            # Get basic structure for phase data            
            beginning = (first_line or first_line_spaces) + f' ({phase}) '
            first_line = False
            new_line = '\n' + len(beginning) * ' '

            # Set chemical data
            flow_rates = ''
            N_IDs = len(data)
            lengths = [len(i) for i in IDs]
            too_many_chemicals = N_IDs > N_max
            N = N_max if too_many_chemicals else N_IDs
            for i in range(N):
                spaces = ' ' * (maxlen - lengths[i])
                if i: flow_rates += new_line    
                flow_rates += f'{IDs[i]} ' + spaces + f' {data[i]:.3g}'
            if too_many_chemicals:
                flow_rates += new_line + '...'
            if composition:
                dashes = '-' * maxlen
                flow_rates += f"{new_line}{dashes}  {total_flow:.3g} {flow_units}"
            # Put it together
            phases_flow_rates_info += beginning + flow_rates + '\n'
            
        return basic_info + phases_flow_rates_info.rstrip('\n')
    
    def print(self):
        """
        Print in a format that you can use recreate the stream.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream(ID='s1',T=298.15, P=101325,
        ...                      l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s1.print()
        MultiStream(ID='s1', phases=('g', 'l'), T=298.15, P=101325, l=[('Water', 1.11), ('Ethanol', 0.2171)])
        
        """
        
        IDs = self.chemicals.IDs
        phase_data = []
        for phase, data in self.imol:
            IDdata = utils.repr_couples(", ", IDs, data)
            if IDdata:
                phase_data.append(f"{phase}=[{IDdata}]")
        dlim = ", "
        phase_data = dlim.join(phase_data)
        phases = f'phases={self.phases}'
        if phase_data:
            phase_data = dlim + phase_data
        price = utils.repr_kwarg('price', self.price)
        print(f"{type(self).__name__}(ID={repr(self.ID)}, {phases}, T={self.T:.2f}, "
              f"P={self.P:.6g}{price}{phase_data})")
    