# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from __future__ import annotations
from ._thermo import Thermo
from ._stream import Stream
from ._thermal_condition import ThermalCondition
from .indexer import MolarFlowIndexer
from ._phase import phase_tuple
from . import equilibrium as eq
from . import utils
from .indexer import nonzeros
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Optional, Sequence, Dict, Tuple

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

@utils.registered_franchise(Stream)
class MultiStream(Stream):
    """
    Create a MultiStream object that defines material flow rates for multiple
    phases along with its thermodynamic state. Thermodynamic and transport
    properties of a stream are available as properties, while thermodynamic
    equilbrium (e.g. VLE, and bubble and dew points) are available as methods. 

    Parameters
    ----------
    ID : 
        A unique identification. If ID is None, stream will not be registered.
        If no ID is given, stream will be registered with a unique ID.
    flow : 
        All flow rates corresponding to `phases` by row and chemical IDs by column.
    T : 
        Temperature [K]. Defaults to 298.15.
    P : 
        Pressure [Pa]. Defaults to 101325.
    phases : 
        Tuple denoting the phases present. Defaults to ('g', 'l').
    units : 
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid). Defaults to 'kmol/hr'.
    price : 
        Price per unit mass [USD/kg]. Defaults to 0.
    total_flow : 
        Total flow rate.
    thermo : 
        Thermodynamic equilibrium package. Defaults to `thermosteam.settings.get_thermo()`.
    vlle :
        Whether to run rigorous phase equilibrium to determine phases. 
        Defaults to False.
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
    sparse([1.11 , 0.217])
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
    1.1101
    >>> # Index multiple chemicals in the liquid phase
    >>> s1.imol['l', ('Ethanol', 'Water')]
    array([0.217, 1.11 ])
    >>> # Index the vapor phase
    >>> s1.imol['g']
    sparse([0., 0.])
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
    flow (kmol/hr): Water    0.617
                    Ethanol  0.0238
    >>> s1['g'].show()
    Stream: 
    phase: 'g', T: 365 K, P: 101325 Pa
    flow (kmol/hr): Water    0.493
                    Ethanol  0.193
    
    Note that the phase cannot be changed:
    
    >>> s1['g'].phase = 'l'
    Traceback (most recent call last):
    AttributeError: phase is locked
    
    """
    __slots__ = ()
    def __init__(self, 
                 ID: Optional[str]="",
                 flow: Optional[Sequence]=(), 
                 T: Optional[float]=298.15, 
                 P: Optional[float]=101325.,
                 phases: Optional[Sequence[str]]=None, 
                 units: Optional[str]=None, 
                 price: Optional[float]=0,
                 total_flow: Optional[float]=None, 
                 thermo: Optional[Thermo]=None, 
                 characterization_factors: Optional[Dict[str, float]]=None, 
                 vlle: Optional[bool]=False, 
                 **phase_flows: Tuple[str, float]):
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
            material_data = self._imol.data / factor
            if total_flow: material_data *= total_flow / material_data.sum()
            flow.data[:] = material_data
        else:
            self._init_indexer(flow, phases, chemicals, phase_flows)
            if total_flow: self._imol.data *= total_flow / self.F_mol
        self._sink = self._source = None
        self.reset_cache()
        self._register(ID)
        if vlle: self.vlle(T, P)
        
    @classmethod
    def from_streams(cls, streams, thermo=None):
        if not streams: raise ValueError('at least one stream must be passed')
        self = cls.__new__(cls)
        self._streams = streams_by_phase = {i.phase: i for i in streams}
        phases = phase_tuple(streams_by_phase)
        N_streams = len(streams)
        if len(phases) != N_streams: raise ValueError('each stream must have a different phase')
        base, *others = streams
        self.characterization_factors = {}
        self._thermal_condition = base._thermal_condition
        for i in others: i._thermal_condition = base._thermal_condition
        self._load_thermo(thermo or base.thermo)
        self.price = 0
        self._imol = MolarFlowIndexer.from_data(
            [streams_by_phase[i]._imol.data for i in phases], phases, 
            chemicals=base.chemicals
        )
        self._sink = self._source = None
        self.reset_cache()
        self._register(None)
        return self
        
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
        composition (%): (g) Ethanol  100
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
    
    def _get_property(self, name, flow=False, nophase=False):
        property_cache = self._property_cache
        thermal_condition = self._thermal_condition
        imol = self._imol
        data = imol.data
        total = data.sum()
        if total == 0.: 
            return 0. if flow else None
        else:
            composition = data / total
            composition_key = [i.dct for i in composition.rows]
            if nophase:
                literal = (thermal_condition._T, thermal_condition._P)
            else:
                literal = (imol._phases, thermal_condition._T, thermal_condition._P)
            last_literal, last_composition_key = self._property_cache_key
            if literal == last_literal and (composition_key == last_composition_key):
                if name in property_cache:
                    value = property_cache.get(name)
                    return value * total if flow else value
            else:
                property_cache.clear()
            self._property_cache_key = (literal, [i.copy() for i in composition_key])
            if nophase:
                calculate = getattr(self.mixture, name)
                self._property_cache[name] = value = calculate(
                    composition.sum(axis=0), *self.thermal_condition
                )
            else:
                calculate = getattr(self.mixture, 'x' + name)
                self._property_cache[name] = value = calculate(
                    zip(imol._phases, composition), *self.thermal_condition
                )
            return value * total if flow else value
    
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
            stream._imol = self._imol.get_phase(phase)
            stream._thermal_condition = self._thermal_condition
            stream._thermo = self._thermo
            stream._bubble_point_cache =  self._bubble_point_cache
            stream._dew_point_cache = self._dew_point_cache
            stream._property_cache = {}
            stream.characterization_factors = {}
            stream._property_cache_key = None, None
            streams[phase] = stream
        return stream
    
    def __iter__(self):
        for i in self.phases: yield self[i]
    
    def __len__(self):
        return len(self.phases)
    
    ### Property getters ###
    
    def get_flow(self, units: str, key=...):
        """
        Return an array of flow rates in requested units.
        
        Parameters
        ----------
        units : 
            Units of measure.
        key : 
            Index key as (phase, IDs), phase, or IDs where:
            
            * phase is a str, ellipsis, or missing.
            * IDs is a str, Sequence[str], ellipisis, or missing.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s1.get_flow('kg/hr', ('l', 'Water'))
        20.0
        
        >>> s1.get_flow('kg/hr')
        sparse([20., 10.])

        """
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        return factor * indexer[key]
    
    def set_flow(self, data: NDArray[float]|float, units: str, key=...):
        """
        Set flow rates in given units.

        Parameters
        ----------
        data : 
            Flow rate data.
        units :
            Units of measure.
        key : 
            Index key as (phase, IDs), phase, or IDs where:
            
            * phase is a str, ellipsis, or missing.
            * IDs is a str, Sequence[str], ellipisis, or missing.

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
    def phases(self) -> tuple[str, ...]:
        """All phases avaiable."""
        return self._imol._phases
    @phases.setter
    def phases(self, phases):
        phases = set(phases)
        if len(phases) == 1:
            self.phase, = phases
        phases = phase_tuple(phases)
        if phases != self.phases:
            self._imol = self._imol.to_material_indexer(phases)
            self.reset_cache()
    
    ### Flow properties ###
            
    @property
    def mol(self) -> NDArray[float]:
        """Chemical molar flow rates (total of all phases)."""
        mol = self._imol.data.sum(0)
        mol.read_only = True
        return mol
    @property
    def mass(self) -> NDArray[float]:
        """Chemical mass flow rates (total of all phases)."""
        mass = self.mol * self.chemicals.MW
        mass.read_only = True
        return mass
    @property
    def vol(self) -> NDArray[float]:
        """Chemical volumetric flow rates (total of all phases)."""
        vol = self.ivol.data.sum(0)
        vol.read_only = True
        return vol
    
    ### Net flow properties ###
    
    @property
    def H(self) -> float:
        """Enthalpy flow rate [kJ/hr]."""
        return self._get_property('H', True)
    @H.setter
    def H(self, H):
        if not H and self.isempty(): return
        self.T = self.mixture.xsolve_T_at_HP(
            self._imol, H, *self._thermal_condition
        )

    @property
    def h(self) -> float:
        """Specific enthalpy in kJ/kmol."""
        return self._get_property('H')
    @h.setter
    def h(self, h: float):
        if not h and self.isempty(): return
        self.T = self.mixture.xsolve_T_at_HP(
            self._imol.iter_composition(), h, *self._thermal_condition
        )

    @property
    def S(self) -> float:
        """Absolute entropy flow rate [kJ/hr]."""
        return self._get_property('S', True)
    @S.setter
    def S(self, S):
        if not S and self.isempty(): return
        self.T = self.mixture.xsolve_T_at_SP(
            self._imol, S, *self._thermal_condition
        )
    
    ### Composition properties ###
    
    @property
    def vapor_fraction(self) -> float:
        """Molar vapor fraction."""
        return self.imol['g'].sum() / F_mol if 'g' in self.phases and (F_mol:=self.F_mol) != 0. else 0.
        
    @property
    def liquid_fraction(self) -> float:
        """Molar liquid fraction."""
        return get_phase_fraction(self, 'lL')
    @property
    def solid_fraction(self) -> float:
        """Molar solid fraction."""
        return get_phase_fraction(self, 'sS')
    
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
    
    def copy_like(self, other):
        """
        Copy all conditions of another stream.

        Examples
        --------
        Copy data from another stream with the same property package:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> s2 = tmo.Stream('s2', Water=2, units='kg/hr')
        >>> s1.copy_like(s2)
        >>> s1.show(flow='kg/hr')
        MultiStream: s1
        phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
        flow (kg/hr): (l) Water  2
         
        Copy data from another stream with a different property package:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10)], units='kg/hr')
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> s2 = tmo.Stream('s2', Water=2, units='kg/hr')
        >>> s1.copy_like(s2)
        >>> s1.show(flow='kg/hr')
        MultiStream: s1
        phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
        flow (kg/hr): (l) Water  2

        """
        self._imol.copy_like(other._imol)
        self._thermal_condition.copy_like(other._thermal_condition)
    
    def copy_flow(self, 
                  other: Stream, 
                  phase: Optional[Sequence[str]|str]=...,
                  IDs: Optional[Sequence[str]|str]=..., *,
                  remove: Optional[bool]=False,
                  exclude: Optional[bool]=False):
        """
        Copy flow rates of another stream to self.
        
        Parameters
        ----------
        other : 
            Flow rates will be copied from here.
        IDs : 
            Chemical IDs. Defaults to all chemicals.
        remove : 
            If True, copied chemicals will be removed from `stream`.
        exclude :
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
    
    def get_normalized_mol(self, IDs: Sequence[str]):
        """
        Return normalized molar fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : 
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
    
    def get_normalized_vol(self, IDs: Sequence[str]):
        """
        Return normalized mass fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : 
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
    
    def get_normalized_mass(self, IDs: Sequence[str]):
        """
        Return normalized mass fractions of given chemicals. The sum of the result is always 1.

        Parameters
        ----------
        IDs : 
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
    
    def get_molar_composition(self, IDs: Sequence[str]):
        """
        Return molar composition of given chemicals.

        Parameters
        ----------
        IDs : 
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
    
    def get_mass_composition(self, IDs: Sequence[str]):
        """
        Return mass composition of given chemicals.

        Parameters
        ----------
        IDs : 
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
    
    def get_volumetric_composition(self, IDs: Sequence[str]):
        """
        Return volumetric composition of given chemicals.

        Parameters
        ----------
        IDs : 
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
    
    def get_concentration(self, phase: str, IDs: Sequence[str]|str, units: Optional[str]=None):
        """
        Return concentration of given chemicals in kmol/m3.

        Parameters
        ----------
        phase :
        IDs : 
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.MultiStream('s1', l=[('Water', 20), ('Ethanol', 10), ('Methanol', 10)], units='m3/hr')
        >>> s1.get_concentration('l', ('Water', 'Ethanol'))
        array([27.673,  4.261])

        >>> s1.get_concentration('l', ('Water', 'Ethanol'), 'g/L')
        array([498.532, 196.291])

        """
        F_vol = self.F_vol
        if F_vol == 0.: return 0.
        if units is None:
            return self.imol[IDs] / F_vol 
        else:
            num, denum = units.split('/')
            return self.get_flow(num + '/hr', (phase, IDs)) / self.get_total_flow(denum + '/hr')
        
        return self.imol[phase, IDs] / self.F_vol
    
    ### Equilibrium ###
    
    # @property
    # def equilibrium(self):
    #     """[Equilibrium] An object that can perform equilibrium on the stream."""
    #     phases = self.phases
    #     if phases == ('g', 'l'):
    #         cache = self._vle_cache
    #     elif phases == ('l', 'L'):
    #         cache = self._lle_cache
    #     elif phases == ('l', 's'):
    #         cache = self._sle_cache
    #     else:
    #         return None
    #     return cache.retrieve()
        
    @property
    def vle(self) -> eq.VLE:
        """An object that can perform vapor-liquid equilibrium on the stream."""
        phases = self.phases
        if 'l' not in phases or 'g' not in phases: 
            self.phases = [*phases, 'l', 'g']
        return self._vle_cache.retrieve()
    @property
    def lle(self) -> eq.LLE:
        """An object that can perform liquid-liquid equilibrium on the stream."""
        phases = self.phases
        if 'l' not in phases or 'L' not in phases: 
            self.phases = [*phases, 'l', 'L']
        return self._lle_cache.retrieve()
    @property
    def sle(self) -> eq.SLE:
        """An object that can perform solid-liquid equilibrium on the stream."""
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
    
    def reduce_phases(self):
        """Remove empty phases."""
        self.phase = self.phase
    
    @property
    def phase(self) -> str:
        imol = self._imol
        return ''.join([phases[0] for phases in ('g', 'lL', 'sS') if not imol.phases_are_empty(phases)])
    @phase.setter
    def phase(self, phase):
        N_phase = len(phase)
        if N_phase > 1: 
            self.phases = phase
        else:
            if N_phase == 0: phase = 'l' # Default phase for streams
            self._imol = self._imol.to_chemical_indexer(phase)
            self._streams.clear()
            self.__class__ = Stream
    
    ### Representation ###
    
    def _info_str(self, units, notation, composition, N_max, all_IDs, indexer, factor):
        """Return string with all specifications."""
        basic_info = self._basic_info()
        basic_info += Stream._info_phaseTP(self, self.phases, units, notation)
        flow_units = units['flow']
        flow_notation = notation['flow']
        N_all_IDs = len(all_IDs)
        if N_all_IDs == 0:
            return basic_info + ' flow: 0' 

        # Length of chemical column
        all_lengths = [len(i) for i in all_IDs[:N_max]]
        maxlen = max(all_lengths) + 2
        
        if composition:
            first_line = "composition (%):"
        else:
            first_line = f'flow ({flow_units}):'
        first_line_spaces = len(first_line)*" "

        # Set up chemical data for all phases
        phases_flow_rates_info = ''
        for phase in self.phases:
            phase_data = factor * indexer[phase, all_IDs] 
            IDs, data = nonzeros(all_IDs, phase_data)
            data = np.array(data)
            if not IDs: continue
            if composition:
                total_flow = data.sum()
                data = 100 * data/total_flow
        
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
                flow_rates += f'{IDs[i]}' + spaces + f'{data[i]:{flow_notation}}'
            if too_many_chemicals:
                spaces = ' ' * (maxlen - 3)
                flow_rates += new_line + '...' + spaces + f'{data[N_max:].sum():{flow_notation}}'
            if composition:
                dashes = '-' * (maxlen - 2)
                flow_rates += f"{new_line}{dashes}  {total_flow:{flow_notation}} {flow_units}"
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
    
