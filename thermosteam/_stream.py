# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numpy as np
import thermosteam as tmo
from . import indexer
from . import equilibrium as eq
from . import functional as fn
from . import units_of_measure as thermo_units
from .exceptions import DimensionError, DomainError
from chemicals.elements import array_to_atoms, symbol_to_index
from . import utils

__all__ = ('Stream', )

# %% Utilities

mol_units = indexer.ChemicalMolarFlowIndexer.units
mass_units = indexer.ChemicalMassFlowIndexer.units
vol_units = indexer.ChemicalVolumetricFlowIndexer.units

class StreamData:
    __slots__ = ('_imol', '_T', '_P', '_phases')
    
    def __init__(self, imol, thermal_condition, phases):
        self._imol = imol.copy()
        self._T = thermal_condition.T
        self._P = thermal_condition.P
        self._phases = phases
        
    
# %%

@utils.thermo_user
@utils.registered(ticket_name='s')
class Stream:
    """
    Create a Stream object that defines material flow rates
    along with its thermodynamic state. Thermodynamic and transport
    properties of a stream are available as properties, while
    thermodynamic equilbrium (e.g. VLE, and bubble and dew points)
    are available as methods. 

    Parameters
    ----------
    ID='' : str
        A unique identification. If ID is None, stream will not be registered.
        If no ID is given, stream will be registered with a unique ID.
    flow=() : Iterable[float]
        All flow rates corresponding to chemical `IDs`.
    phase='l' : 'l', 'g', or 's'
        Either gas (g), liquid (l), or solid (s).
    T=298.15 : float
        Temperature [K].
    P=101325 : float
        Pressure [Pa].
    units='kmol/hr' : str
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid).
    price=0 : float
        Price per unit mass [USD/kg].
    thermo=None : :class:`~thermosteam.Thermo`
        Thermo object to initialize input and output streams. Defaults to
        `biosteam.settings.get_thermo()`.
    **chemical_flows : float
        ID - flow pairs.
    
    Examples
    --------
    Before creating a stream, first set the chemicals:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    
    Create a stream, defining the thermodynamic condition and flow rates:
        
    >>> s1 = tmo.Stream(ID='s1',
    ...                 Water=20, Ethanol=10, units='kg/hr',
    ...                 T=298.15, P=101325, phase='l')
    >>> s1.show(flow='kg/hr') # Use the show method to select units of display
    Stream: s1
     phase: 'l', T: 298.15 K, P: 101325 Pa
     flow (kg/hr): Water    20
                   Ethanol  10
    >>> s1.show(composition=True, flow='kg/hr') # Its also possible to show by composition
    Stream: s1
     phase: 'l', T: 298.15 K, P: 101325 Pa
     composition: Water    0.667
                  Ethanol  0.333
                  -------  30 kg/hr
    
    All flow rates are stored as an array in the `mol` attribute:
    
    >>> s1.mol # Molar flow rates [kmol/hr]
    array([1.11 , 0.217])
    
    Mass and volumetric flow rates are available as property arrays:
        
    >>> s1.mass
    property_array([<Water: 20 kg/hr>, <Ethanol: 10 kg/hr>])
    >>> s1.vol
    property_array([<Water: 0.02006 m^3/hr>, <Ethanol: 0.012722 m^3/hr>])
    
    These arrays work just like ordinary arrays, but the data is linked to the molar flows:
    
    >>> # Mass flows are always up to date with molar flows
    >>> s1.mol[0] = 1
    >>> s1.mass[0]
    <Water: 18.015 kg/hr>
    >>> # Changing mass flows changes molar flows
    >>> s1.mass[0] *= 2
    >>> s1.mol[0]
    2.0
    >>> # Property arrays act just like normal arrays
    >>> s1.mass + 2
    array([38.031, 12.   ])
    
    The temperature, pressure and phase are attributes as well:
    
    >>> (s1.T, s1.P, s1.phase)
    (298.15, 101325.0, 'l')
    
    The most convinient way to get and set flow rates is through
    the `get_flow` and `set_flow` methods:
    
    >>> # Set flow
    >>> s1.set_flow(1, 'gpm', 'Water')
    >>> s1.get_flow('gpm', 'Water')
    1.0
    >>> # Set multiple flows
    >>> s1.set_flow([10, 20], 'kg/hr', ('Ethanol', 'Water'))
    >>> s1.get_flow('kg/hr', ('Ethanol', 'Water'))
    array([10., 20.])
    
    It is also possible to index using IDs through the
    `imol`, `imass`, and `ivol` indexers:
    
    >>> s1.imol.show()
    ChemicalMolarFlowIndexer (kmol/hr):
     (l) Water    1.11
         Ethanol  0.2171
    >>> s1.imol['Water']
    1.1101687012358397
    >>> s1.imol['Ethanol', 'Water']
    array([0.217, 1.11 ])
    
    Thermodynamic properties are available as stream properties:
    
    >>> s1.H # Enthalpy (kJ/hr)
    0.0
    
    Note that the reference enthalpy is 0.0 at the reference
    temperature of 298.15 K, and pressure of 101325 Pa.
    Retrive the enthalpy at a 10 degC above the reference.
    
    >>> s1.T += 10
    >>> s1.H
    1083.467954...
    
    Other thermodynamic properties are temperature and pressure dependent as well:
    
    >>> s1.rho # Density [kg/m3]
    908.8914226...
    
    It may be more convinient to get properties with different units:
        
    >>> s1.get_property('rho', 'g/cm3')
    0.90889142...
    
    It is also possible to set some of the properties in different units:
        
    >>> s1.set_property('T', 40, 'degC')
    >>> s1.T
    313.15
    
    Bubble point and dew point computations can be performed through stream methods:
        
    >>> bp = s1.bubble_point_at_P() # Bubble point at constant pressure
    >>> bp
    BubblePointValues(T=357.09, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.49 0.51])
    
    The bubble point results contain all results as attributes:
    
    >>> bp.T # Temperature [K]
    357.088...
    >>> bp.y # Vapor composition
    array([0.49, 0.51])
    
    Vapor-liquid equilibrium can be performed by setting 2 degrees of freedom from the following list: `T` [Temperature; in K], `P` [Pressure; in Pa], `V` [Vapor fraction], `H` [Enthalpy; in kJ/hr].
    
    Set vapor fraction and pressure of the stream:
        
    >>> s1.vle(P=101325, V=0.5)
    >>> s1.show()
    MultiStream: s1
     phases: ('g', 'l'), T: 364.8 K, P: 101325 Pa
     flow (kmol/hr): (g) Water    0.472
                         Ethanol  0.192
                     (l) Water    0.638
                         Ethanol  0.0255
    
    Note that the stream is a now a MultiStream object to manage multiple phases.
    Each phase can be accessed separately too:
    
    >>> s1['l'].show()
    Stream: 
     phase: 'l', T: 364.8 K, P: 101325 Pa
     flow (kmol/hr): Water    0.638
                     Ethanol  0.0255
    
    >>> s1['g'].show()
    Stream: 
     phase: 'g', T: 364.8 K, P: 101325 Pa
     flow (kmol/hr): Water    0.472
                     Ethanol  0.192
    
    We can convert a MultiStream object back to a Stream object by setting the phase:
        
    >>> s1.phase = 'l'
    >>> s1.show(flow='kg/hr')
    Stream: s1
     phase: 'l', T: 364.8 K, P: 101325 Pa
     flow (kg/hr): Water    20
                   Ethanol  10
    
    """
    __slots__ = ('_ID', '_imol', '_thermal_condition', '_thermo', '_streams',
                 '_bubble_point_cache', '_dew_point_cache',
                 '_vle_cache', '_lle_cache', '_sle_cache',
                 '_sink', '_source', '_price', '_link')
    line = 'Stream'
    
    #: [DisplayUnits] Units of measure for IPython display (class attribute)
    display_units = thermo_units.DisplayUnits(T='K', P='Pa',
                                              flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                              composition=False,
                                              N=7)

    _flow_cache = {}

    def __init__(self, ID= '', flow=(), phase='l', T=298.15, P=101325.,
                 units='kmol/hr', price=0., thermo=None, **chemical_flows):
        self._thermal_condition = tmo.ThermalCondition(T, P)
        thermo = self._load_thermo(thermo)
        self._init_indexer(flow, phase, thermo.chemicals, chemical_flows)
        self.price = price
        if units != 'kmol/hr':
            name, factor = self._get_flow_name_and_factor(units)
            flow = getattr(self, name)
            flow[:] = self.mol / factor
        self._sink = self._source = None # For BioSTEAM
        self._init_cache()
        self._register(ID)
        self._link = None

    def as_stream(self):
        """Does nothing."""

    def get_data(self):
        """
        Return a StreamData object containing data on material flow rates, 
        temperature, pressure, and phase(s).
        
        See Also
        --------
        Stream.set_data
        
        Examples
        --------
        Get and set data from stream at different conditions
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> stream = tmo.Stream('stream', Water=10)
        >>> data = stream.get_data()
        >>> stream.vle(V=0.5, P=101325)
        >>> data_vle = stream.get_data()
        >>> stream.set_data(data)
        >>> stream.show()
        Stream: stream
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kmol/hr): Water  10
        >>> stream.set_data(data_vle)
        >>> stream.show()
        MultiStream: stream
         phases: ('g', 'l'), T: 373.12 K, P: 101325 Pa
         flow (kmol/hr): (g) Water  5
                         (l) Water  5
        
        Note that only StreamData objects are valid for this method:
        
        >>> stream.set_data({'T': 298.15})
        Traceback (most recent call last):
        ValueError: stream_data must be a StreamData object; not dict
        
        """
        return StreamData(self.imol, self.thermal_condition, self.phases)

    def set_data(self, stream_data):
        """
        Set material flow rates, temperature, pressure, and phase(s) through a 
        StreamData object

        See Also
        --------
        Stream.get_data

        """
        if isinstance(stream_data, StreamData):
            self.phases = stream_data._phases
            self._imol.copy_like(stream_data._imol)
            self._thermal_condition.copy_like(stream_data)
        else:
            raise ValueError(f'stream_data must be a StreamData object; not {type(stream_data).__name__}')
        
    @property
    def price(self):
        """[float] Price of stream per unit mass [USD/kg."""
        return self._price
    @price.setter
    def price(self, price):
        self._price = float(price)
    
    def isempty(self):
        """
        Return whether or not stream is empty.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> stream = tmo.Stream()
        >>> stream.isempty()
        True
        
        """
        return (self._imol._data == 0.).all()

    @property
    def vapor_fraction(self):
        """Molar vapor fraction."""
        return 1.0 if self.phase in 'gG' else 0.0
    @property
    def liquid_fraction(self):
        """Molar liquid fraction."""
        return 1.0 if self.phase in 'lL' else 0.0
    @property
    def solid_fraction(self):
        """Molar solid fraction."""
        return 1.0 if self.phase in 'sS' else 0.0

    def isfeed(self):
        """Return whether stream has a sink but no source."""
        return bool(self._sink and not self._source)

    def isproduct(self):
        """Return whether stream has a source but no sink."""
        return bool(self._source and not self._sink)

    @property
    def main_chemical(self):
        """[str] ID of chemical with the largest mol fraction in stream."""
        return self.chemicals.tuple[self.mol.argmax()].ID

    def disconnect(self):
        """Disconnect stream from unit operations."""
        sink = self._sink
        source = self._source
        if sink:
            ins = sink.ins
            index = ins.index(self)
            ins[index] = None
        if source:
            outs = source.outs
            index = outs.index(self)
            outs[index] = None
    
    def _init_indexer(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if len(flow) == 0:
            if chemical_flows:
                imol = indexer.ChemicalMolarFlowIndexer(phase, chemicals=chemicals, **chemical_flows)
            else:
                imol = indexer.ChemicalMolarFlowIndexer.blank(phase, chemicals)
        else:
            assert not chemical_flows, ("may specify either 'flow' or "
                                        "'chemical_flows', but not both")
            if isinstance(flow, indexer.ChemicalMolarFlowIndexer):
                imol = flow 
                imol.phase = phase
            else:
                imol = indexer.ChemicalMolarFlowIndexer.from_data(
                    np.asarray(flow, dtype=float), phase, chemicals)
        self._imol = imol

    def _init_cache(self):
        self._bubble_point_cache = eq.BubblePointCache()
        self._dew_point_cache = eq.DewPointCache()

    @classmethod
    def _get_flow_name_and_factor(cls, units):
        cache = cls._flow_cache
        if units in cache:
            name, factor = cache[units]
        else:
            dimensionality = thermo_units.get_dimensionality(units)
            if dimensionality == mol_units.dimensionality:
                name = 'mol'
                factor = mol_units.conversion_factor(units)
            elif dimensionality == mass_units.dimensionality:
                name = 'mass'
                factor = mass_units.conversion_factor(units)
            elif dimensionality == vol_units.dimensionality:
                name = 'vol'
                factor = vol_units.conversion_factor(units)
            else:
                raise DimensionError("dimensions for flow units must be in molar, "
                                     "mass or volumetric flow rates, not "
                                    f"'{dimensionality}'")
            cache[units] = name, factor
        return name, factor

    ### Property getters ###

    def get_atomic_flow(self, symbol):
        """
        Return flow rate of atom in kmol / hr given the atomic symbol.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> stream = tmo.Stream(Water=1)
        >>> stream.get_atomic_flow('H') # kmol/hr of H
        2.0
        >>> stream.get_atomic_flow('O') # kmol/hr of O
        1.0
        
        """
        return (self.chemicals.formula_array[symbol_to_index[symbol], :] * self.mol).sum()

    def get_atomic_flows(self):
        """
        Return dictionary of atomic flow rates in kmol / hr.
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> stream = tmo.Stream(Water=1)
        >>> stream.get_atomic_flows()
        {'H': 2.0, 'O': 1.0}
        
        """
        return array_to_atoms(self.chemicals.formula_array @ self.mol)

    def get_flow(self, units, key=...):
        """
        Return an flow rates in requested units.
        
        Parameters
        ----------
        units : str
            Units of measure.
        key : tuple[str] or str, optional
            Chemical identifiers.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.get_flow('kg/hr', 'Water')
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
        key : Iterable[str] or str, optional
            Chemical identifiers.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream(ID='s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.set_flow(10, 'kg/hr', 'Water')
        >>> s1.get_flow('kg/hr', 'Water')
        10.0

        """
        name, factor = self._get_flow_name_and_factor(units)
        indexer = getattr(self, 'i' + name)
        indexer[key] = np.asarray(data, dtype=float) / factor
    
    def get_total_flow(self, units):
        """
        Get total flow rate in given units.

        Parameters
        ----------
        units : str
            Units of measure.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.get_total_flow('kg/hr')
        30.0

        """
        name, factor = self._get_flow_name_and_factor(units)
        flow = getattr(self, 'F_' + name)
        return factor * flow
    
    def set_total_flow(self, value, units):
        """
        Set total flow rate in given units keeping the composition constant.

        Parameters
        ----------
        value : float
            New total flow rate.
        units : str
            Units of measure.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.set_total_flow(1.0,'kg/hr')
        >>> s1.get_total_flow('kg/hr')
        0.9999999999999999

        """
        name, factor = self._get_flow_name_and_factor(units)
        setattr(self, 'F_' + name, value / factor)
    
    def get_property(self, name, units):
        """
        Return property in requested units.

        Parameters
        ----------
        name : str
            Name of stream property.
        units : str
            Units of measure.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.get_property('sigma', 'N/m') # Surface tension
        0.063780393

        """
        value = getattr(self, name)
        units_dct = thermo_units.stream_units_of_measure
        if name in units_dct:
            original_units = units_dct[name]
        else:
            raise ValueError(f"'{name}' is not thermodynamic property")
        return original_units.convert(value, units)
    
    def set_property(self, name, value, units):
        """
        Set property in given units.

        Parameters
        ----------
        name : str
            Name of stream property.
        value : str
            New value of stream property.
        units : str
            Units of measure.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.set_property('P', 2, 'atm')
        >>> s1.P
        202650.0

        """
        units_dct = thermo_units.stream_units_of_measure
        if name in units_dct:
            original_units = units_dct[name]
        else:
            raise ValueError(f"no property with name '{name}'")
        value = original_units.unconvert(value, units)
        setattr(self, name, value)
    
    ### Stream data ###

    @property
    def source(self):
        """[Unit] Outlet location."""
        return self._source
    @property
    def sink(self):
        """[Unit] Inlet location."""
        return self._sink

    @property
    def thermal_condition(self):
        """
        [ThermalCondition] Contains the temperature and pressure conditions 
        of the stream.
        """
        return self._thermal_condition

    @property
    def T(self):
        """[float] Temperature in Kelvin."""
        return self._thermal_condition.T
    @T.setter
    def T(self, T):
        self._thermal_condition.T = T
    
    @property
    def P(self):
        """[float] Pressure in Pascal."""
        return self._thermal_condition.P
    @P.setter
    def P(self, P):
        self._thermal_condition.P = P
    
    @property
    def phase(self):
        """Phase of stream."""
        return self._imol.phase
    @phase.setter
    def phase(self, phase):
        self._imol.phase = phase
    
    @property
    def mol(self):
        """[array] Molar flow rates in kmol/hr."""
        return self._imol._data
    @mol.setter
    def mol(self, value):
        mol = self.mol
        if mol is not value:
            mol[:] = value
    
    @property
    def mass(self):
        """[property_array] Mass flow rates in kg/hr."""
        return self.imass._data
    @mass.setter
    def mass(self, value):
        mass = self.mass
        if mass is not value:
            mass[:] = value
    
    @property
    def vol(self):
        """[property_array] Volumetric flow rates in m3/hr."""
        return self.ivol._data
    @vol.setter
    def vol(self, value):
        vol = self.vol
        if vol is not value:
            vol[:] = value
        
    @property
    def imol(self):
        """[Indexer] Flow rate indexer with data in kmol/hr."""
        return self._imol
    @property
    def imass(self):
        """[Indexer] Flow rate indexer with data in kg/hr."""
        return self._imol.by_mass()
    @property
    def ivol(self):
        """[Indexer] Flow rate indexer with data in m3/hr."""
        return self._imol.by_volume(self._thermal_condition)
    
    ### Net flow properties ###
    
    @property
    def cost(self):
        """[float] Total cost of stream in USD/hr."""
        return self.price * self.F_mass
    
    @property
    def F_mol(self):
        """[float] Total molar flow rate in kmol/hr."""
        return self._imol._data.sum()
    @F_mol.setter
    def F_mol(self, value):
        F_mol = self.F_mol
        if not F_mol: raise AttributeError("undefined composition; cannot set flow rate")
        self._imol._data[:] *= value/F_mol
    @property
    def F_mass(self):
        """[float] Total mass flow rate in kg/hr."""
        return (self.chemicals.MW * self.mol).sum()
    @F_mass.setter
    def F_mass(self, value):
        F_mass = self.F_mass
        if not F_mass: raise AttributeError("undefined composition; cannot set flow rate")
        self.imol._data[:] *= value/F_mass
    @property
    def F_vol(self):
        """[float] Total volumetric flow rate in m3/hr."""
        return 1000. * self.mixture.V(self.phase, self.mol, *self._thermal_condition)
    @F_vol.setter
    def F_vol(self, value):
        F_vol = self.F_vol
        if not F_vol: raise AttributeError("undefined composition; cannot set flow rate")
        self.imol._data[:] *= value / F_vol
    
    @property
    def H(self):
        """[float] Enthalpy flow rate in kJ/hr."""
        return self.mixture.H(self.phase, self.mol, *self._thermal_condition)
    @H.setter
    def H(self, H:float):
        self.T = self.mixture.solve_T(self.phase, self.mol, H, *self._thermal_condition)

    @property
    def S(self):
        """[float] Absolute entropy flow rate in kJ/hr."""
        return self.mixture.S(self.phase, self.mol, *self._thermal_condition)
    
    @property
    def Hnet(self):
        """[float] Total enthalpy flow rate (including heats of formation) in kJ/hr."""
        return self.H + self.Hf
    
    @property
    def Hf(self):
        """[float] Enthalpy of formation flow rate in kJ/hr."""
        return (self.chemicals.Hf * self.mol).sum()
    @property
    def LHV(self):
        """[float] Lower heating value flow rate in kJ/hr."""
        return (self.chemicals.LHV * self.mol).sum()    
    @property
    def HHV(self):
        """[float] Higher heating value flow rate in kJ/hr."""
        return (self.chemicals.HHV * self.mol).sum()    
    @property
    def Hvap(self):
        """[float] Enthalpy of vaporization flow rate in kJ/hr."""
        mol = self.mol
        condition = self._thermal_condition
        return sum([i*j.Hvap(condition.T) for i,j in zip(mol, self.chemicals) if i and not j.locked_state])
    
    @property
    def C(self):
        """[float] Heat capacity flow rate in kJ/K/hr."""
        return self.mixture.Cn(self.phase, self.mol, self.T)
    
    ### Composition properties ###
    
    @property
    def z_mol(self):
        """[1d array] Molar composition."""
        mol = self.mol
        z = mol / mol.sum()
        z.setflags(0)
        return z
    @property
    def z_mass(self):
        """[1d array] Mass composition."""
        mass = self.chemicals.MW * self.mol
        z = mass / mass.sum()
        z.setflags(0)
        return z
    @property
    def z_vol(self):
        """[1d array] Volumetric composition."""
        vol = 1. * self.vol
        z = vol / vol.sum()
        z.setflags(0)
        return z
    
    @property
    def MW(self):
        """[float] Overall molecular weight."""
        return self.F_mass / self.F_mol
    @property
    def V(self):
        """[float] Molar volume [m^3/mol]."""
        mol = self.mol
        return self.mixture.V(self.phase, mol / mol.sum(), *self._thermal_condition)
    @property
    def kappa(self):
        """[float] Thermal conductivity [W/m/k]."""
        mol = self.mol
        return self.mixture.kappa(self.phase, mol / mol.sum(), *self._thermal_condition)
    @property
    def Cn(self):
        """[float] Molar heat capacity [J/mol/K]."""
        mol = self.mol
        return self.mixture.Cn(self.phase, mol / mol.sum(), self.T)
    @property
    def mu(self):
        """[float] Hydrolic viscosity [Pa*s]."""
        mol = self.mol
        return self.mixture.mu(self.phase, mol / mol.sum(), *self._thermal_condition)
    @property
    def sigma(self):
        """[float] Surface tension [N/m]."""
        mol = self.mol
        return self.mixture.sigma(mol / mol.sum(), *self._thermal_condition)
    @property
    def epsilon(self):
        """[float] Relative permittivity [-]."""
        mol = self.mol
        return self.mixture.epsilon(mol / mol.sum(), *self._thermal_condition)
    
    @property
    def Cp(self):
        """[float] Heat capacity [J/g/K]."""
        return self.Cn / self.MW
    @property
    def alpha(self):
        """[float] Thermal diffusivity [m^2/s]."""
        return fn.alpha(self.kappa, self.rho, self.Cp * 1000.)
    @property
    def rho(self):
        """[float] Density [kg/m^3]."""
        return fn.V_to_rho(self.V, self.MW)
    @property
    def nu(self):
        """[float] Kinematic viscosity [-]."""
        return fn.mu_to_nu(self.mu, self.rho)
    @property
    def Pr(self):
        """[float] Prandtl number [-]."""
        return fn.Pr(self.Cp * 1000, self.kappa, self.mu)
    
    ### Stream methods ###
    
    @property
    def available_chemicals(self):
        """list[Chemical] All chemicals with nonzero flow."""
        return [i for i, j in zip(self.chemicals, self.mol) if j]
    
    def in_thermal_equilibrium(self, other):
        """
        Return whether or not stream is in thermal equilibrium with
        another stream.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> stream = Stream(Water=1, T=300)
        >>> other = Stream(Water=1, T=300)
        >>> stream.in_thermal_equilibrium(other)
        True
        
        """
        return self._thermal_condition.in_equilibrium(other._thermal_condition)
    
    @classmethod
    def sum(cls, streams, ID=None, thermo=None):
        """
        Return a new Stream object that represents the sum of all given streams.
        
        Examples
        --------
        Sum two streams:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s_sum = tmo.Stream.sum([s1, s1], 's_sum')
        >>> s_sum.show(flow='kg/hr')
        Stream: s_sum
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    40
                       Ethanol  20
        
        Sum two streams with new property package:
            
        >>> thermo = tmo.Thermo(['Water', 'Ethanol', 'Methanol'], cache=True)
        >>> s_sum = tmo.Stream.sum([s1, s1], 's_sum', thermo)
        >>> s_sum.show(flow='kg/hr')
        Stream: s_sum
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    40
                       Ethanol  20
        """
        if streams:
            new = streams[0].copy(ID)
            new.mix_from(streams)
        else:
            new = cls(ID, thermo=thermo)
        return new
    
    def mix_from(self, others, energy_balance=True):
        """
        Mix all other streams into this one, ignoring its initial contents.
        
        Examples
        --------
        Mix two streams with the same thermodynamic property package:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = s1.copy('s2')
        >>> s1.mix_from([s1, s2])
        >>> s1.show(flow='kg/hr')
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    40
                       Ethanol  20
        
        It's also possible to mix streams with different property packages
        so long as all chemicals are defined in the mixed stream's property 
        package:
        
        >>> tmo.settings.set_thermo(['Water'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=40, units='kg/hr')
        >>> tmo.settings.set_thermo(['Ethanol'], cache=True) 
        >>> s2 = tmo.Stream('s2', Ethanol=20, units='kg/hr')
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s_mix = tmo.Stream('s_mix')
        >>> s_mix.mix_from([s1, s2])
        >>> s_mix.show(flow='kg/hr')
        Stream: s_mix
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    40
                       Ethanol  20
        
        Mixing emptry streams if fine too:
            
        >>> s1.empty(); s2.empty(); s_mix.mix_from([s1, s2])
        >>> s_mix.show()
        Stream: s_mix
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow: 0
        
        """
        others = [i for i in others if i]
        N_others = len(others)
        if N_others == 0:
            self.empty()
        elif N_others == 1:
            self.copy_like(others[0])
        else:
            self._imol.mix_from([i._imol for i in others])
            if not energy_balance: return
            H = sum([i.H for i in others])
            if self.isempty():
                self.T = np.mean([i.T for i in others])
            elif isinstance(self, tmo.MultiStream):
                self.H = H
            else:
                try: self.H = H
                except: # pragma: no cover
                    phase = self.phase.lower()
                    if phase == 'g':
                        # Maybe too little heat, liquid must be present
                        self.phase = 'l'
                    elif phase == 'l':
                        # Maybe too much heat, gas must be present
                        self.phase = 'g'
                    else:
                        phases = ''.join([i.phase for i in others])
                        self.phases = tuple(set(phases))
                        self._imol.mix_from([i._imol for i in others])
                    try: 
                        self.H = H
                    except:
                        phases = ''.join([i.phase for i in others])
                        self.phases = tuple(set(phases))
                        self._imol.mix_from([i._imol for i in others])
                        self.H = H
            
    def split_to(self, s1, s2, split):
        """
        Split molar flow rate from this stream to two others given
        the split fraction or an array of split fractions.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
        >>> tmo.settings.set_thermo(chemicals)
        >>> s = tmo.Stream('s', Water=20, Ethanol=10, units='kg/hr')
        >>> s1 = tmo.Stream('s1')
        >>> s2 = tmo.Stream('s2')
        >>> split = chemicals.kwarray(dict(Water=0.5, Ethanol=0.1))
        >>> s.split_to(s1, s2, split)
        >>> s1.show(flow='kg/hr')
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    10
                       Ethanol  1
        
        >>> s2.show(flow='kg/hr')
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    10
                       Ethanol  9
        
        
        """
        mol = self.mol
        s1.mol[:] = dummy = mol * split
        s2.mol[:] = mol - dummy
        
    def link_with(self, other, flow=True, phase=True, TP=True):
        """
        Link with another stream.
        
        Parameters
        ----------
        other : Stream
        flow : bool, defaults to True
            Whether to link the flow rate data.
        phase : bool, defaults to True
            Whether to link the phase.
        TP : bool, defaults to True
            Whether to link the temperature and pressure.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2')
        >>> s2.link_with(s1)
        >>> s1.mol is s2.mol
        True
        >>> s2.thermal_condition is s1.thermal_condition
        True
        >>> s1.phase = 'g'
        >>> s2.phase
        'g'
        
        """
        if not isinstance(other._imol, self._imol.__class__):
            at_unit = f" at unit {self.source}" if self.source is other.sink else ""
            raise RuntimeError(f"stream {self} cannot link with stream {other}" + at_unit
                               + "; streams must have the same class to link")
        if self._link and not (self.source is other.sink or self.sink is other.source):
            raise RuntimeError(f"stream {self} cannot link with stream {other};"
                               f" {self} already linked with {self._link}")
        if TP and flow and (phase or self._imol._data.ndim == 2):
            self._imol._data_cache = other._imol._data_cache
        else:
            self._imol._data_cache.clear()
        
        if TP:
            self._thermal_condition = other._thermal_condition
        if flow:
            self._imol._data = other._imol._data
        if phase and self._imol._data.ndim == 1:
            self._imol._phase = other._imol._phase
        
        self._link = other
        other._link = self
            
    def unlink(self):
        """
        Unlink stream from other streams.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2')
        >>> s2.link_with(s1)
        >>> s1.unlink()
        >>> s2.mol is s1.mol
        False
        
        """
        self._imol._data_cache.clear()
        self._thermal_condition = self._thermal_condition.copy()
        self._imol._data = self._imol._data.copy()
        self._imol._phase = self._imol._phase.copy()
        self._init_cache()
        self._link = None
    
    def copy_like(self, other):
        """
        Copy all conditions of another stream.

        Examples
        --------
        Copy data from another stream with the same property package:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2', Water=2, units='kg/hr')
        >>> s1.copy_like(s2)
        >>> s1.show(flow='kg/hr')
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water  2
         
        Copy data from another stream with a different property package:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> s2 = tmo.Stream('s2', Water=2, units='kg/hr')
        >>> s1.copy_like(s2)
        >>> s1.show(flow='kg/hr')
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water  2

        """
        if isinstance(other, tmo.MultiStream):
            phase = other.phase
            if len(phase) == 1:
                imol = other._imol.to_chemical_indexer(phase)
            else:
                self.phases = other.phases
                imol = other._imol
        else:
            imol = other._imol
        self._imol.copy_like(imol)
        self._thermal_condition.copy_like(other._thermal_condition)
    
    def copy_thermal_condition(self, other):
        """
        Copy thermal conditions (T and P) of another stream.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=2, units='kg/hr')
        >>> s2 = tmo.Stream('s2', Water=1, units='kg/hr', T=300.00)
        >>> s1.copy_thermal_condition(s2)
        >>> s1.show(flow='kg/hr')
        Stream: s1
         phase: 'l', T: 300 K, P: 101325 Pa
         flow (kg/hr): Water  2
        """
        self._thermal_condition.copy_like(other._thermal_condition)
    
    def copy_flow(self, stream, IDs=..., *, remove=False, exclude=False):
        """
        Copy flow rates of stream to self.
        
        Parameters
        ----------
        stream : Stream
            Flow rates will be copied from here.
        IDs=... : Iterable[str], defaults to all chemicals.
            Chemical IDs. 
        remove=False: bool, optional
            If True, copied chemicals will be removed from `stream`.
        exclude=False: bool, optional
            If True, exclude designated chemicals when copying.
        
        Examples
        --------
        Initialize streams:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2')
        
        Copy all flows:
        
        >>> s2.copy_flow(s1)
        >>> s2.show(flow='kg/hr')
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    20
                       Ethanol  10
        
        Reset and copy just water flow:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, 'Water')
        >>> s2.show(flow='kg/hr')
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water  20
        
        Reset and copy all flows except water:
        
        >>> s2.empty()
        >>> s2.copy_flow(s1, 'Water', exclude=True)
        >>> s2.show(flow='kg/hr')
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Ethanol  10
        
        Cut and paste flows:
        
        >>> s2.copy_flow(s1, remove=True)
        >>> s2.show(flow='kg/hr')
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    20
                       Ethanol  10
        
        >>> s1.show()
        Stream: s1
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow: 0
         
        Its also possible to copy flows from a multistream:
        
        >>> s1.phases = ('g', 'l')
        >>> s1.imol['g', 'Water'] = 10
        >>> s2.copy_flow(s1, remove=True)
        >>> s2.show()
        Stream: s2
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kmol/hr): Water  10
        >>> s1.show()
        MultiStream: s1
         phases: ('g', 'l'), T: 298.15 K, P: 101325 Pa
         flow: 0
         
        """
        chemicals = self.chemicals
        mol = stream.mol
        if exclude:
            IDs = chemicals.get_index(IDs)
            index = np.ones(chemicals.size, dtype=bool)
            index[IDs] = False
        else:
            index = chemicals.get_index(IDs)
        
        self.mol[index] = mol[index]
        if remove: 
            if isinstance(stream, tmo.MultiStream):
                stream.imol.data[:, index] = 0
            else:
                mol[index] = 0
    
    def copy(self, ID=None):
        """
        Return a copy of the stream.

        Examples
        --------
        Create a copy of a new stream:
        
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1_copy = s1.copy('s1_copy')
        >>> s1_copy.show(flow='kg/hr')
        Stream: s1_copy
         phase: 'l', T: 298.15 K, P: 101325 Pa
         flow (kg/hr): Water    20
                       Ethanol  10
        
        Warnings
        --------
        Prices are not copied.
        
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new._link = new._sink = new._source = None
        new._thermo = self._thermo
        new._imol = self._imol.copy()
        new._thermal_condition = self._thermal_condition.copy()
        new._init_cache()
        new.price = 0
        new.ID = ID
        return new
    __copy__ = copy
    
    def flow_proxy(self, ID=None):
        """
        Return a new stream that shares flow rate data with this one.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = s1.flow_proxy()
        >>> s2.mol is s1.mol
        True
        
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new.ID = new._sink = new._source = None
        new.price = 0
        new._thermo = self._thermo
        new._imol = imol = self._imol._copy_without_data()
        imol._data = self._imol._data
        new._thermal_condition = self._thermal_condition.copy()
        new._init_cache()
        new._link = self
        return new
    
    def proxy(self, ID=None):
        """
        Return a new stream that shares all data with this one.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = s1.proxy()
        >>> s2.imol is s1.imol and s2.thermal_condition is s1.thermal_condition
        True
        
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new.ID = None
        new._sink = new._source = None
        new.price = self.price
        new._thermo = self._thermo
        new._imol = self._imol
        new._thermal_condition = self._thermal_condition
        try: new._vle_cache = self._vle_cache
        except AttributeError: pass
        new._link = self
        return new
    
    def empty(self):
        """Empty stream flow rates.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s1.empty()
        >>> s1.F_mol
        0.0
        
        """
        self._imol.empty()
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        """[VLE] An object that can perform vapor-liquid equilibrium on the stream."""
        self.phases = ('g', 'l')
        return self.vle

    @property
    def lle(self):
        """[LLE] An object that can perform liquid-liquid equilibrium on the stream."""
        self.phases = ('L', 'l')
        return self.lle
    
    @property
    def sle(self):
        """[SLE] An object that can perform solid-liquid equilibrium on the stream."""
        self.phases = ('s', 'l')
        return self.sle

    @property
    def vle_chemicals(self):
        """list[Chemical] Chemicals cabable of liquid-liquid equilibrium."""
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.get_vle_indices(self.mol != 0)
        return [chemicals_tuple[i] for i in indices]
    
    @property
    def lle_chemicals(self):
        """list[Chemical] Chemicals cabable of vapor-liquid equilibrium."""
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.get_lle_indices(self.mol != 0)
        return [chemicals_tuple[i] for i in indices]
    
    def get_bubble_point(self, IDs=None):
        """
        Return a BubblePoint object capable of computing bubble points.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.get_bubble_point()
        BubblePoint([Water, Ethanol])
        
        """
        chemicals = self.chemicals[IDs] if IDs else self.vle_chemicals
        bp = self._bubble_point_cache(chemicals, self._thermo)
        return bp
    
    def get_dew_point(self, IDs=None):
        """
        Return a DewPoint object capable of computing dew points.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.get_dew_point()
        DewPoint([Water, Ethanol])
        
        """
        chemicals = self.chemicals.retrieve(IDs) if IDs else self.vle_chemicals
        dp = self._dew_point_cache(chemicals, self._thermo)
        return dp
    
    def bubble_point_at_T(self, T=None, IDs=None):
        """
        Return a BubblePointResults object with all data on the bubble point at constant temperature.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.bubble_point_at_T()
        BubblePointValues(T=350.00, P=76622, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.486 0.514])
        
        """
        bp = self.get_bubble_point(IDs)
        z = self.get_normalized_mol(bp.IDs)
        return bp(z, T=T or self.T)
    
    def bubble_point_at_P(self, P=None, IDs=None):
        """
        Return a BubblePointResults object with all data on the bubble point at constant pressure.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.bubble_point_at_P()
        BubblePointValues(T=357.09, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.49 0.51])
        
        """
        bp = self.get_bubble_point(IDs)
        z = self.get_normalized_mol(bp.IDs)
        return bp(z, P=P or self.P)
    
    def dew_point_at_T(self, T=None, IDs=None):
        """
        Return a DewPointResults object with all data on the dew point
        at constant temperature.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all
            chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.dew_point_at_T()
        DewPointValues(T=350.00, P=48991, IDs=('Water', 'Ethanol'), z=[0.836 0.164], x=[0.984 0.016])
        
        """
        dp = self.get_dew_point(IDs)
        z = self.get_normalized_mol(dp.IDs)
        return dp(z, T=T or self.T)
    
    def dew_point_at_P(self, P=None, IDs=None):
        """
        Return a DewPointResults object with all data on the dew point
        at constant pressure.
        
        Parameters
        ----------
        IDs : Iterable[str], optional
            Chemicals that participate in equilibrium. Defaults to all
            chemicals in equilibrium.
            
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.dew_point_at_P()
        DewPointValues(T=368.66, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], x=[0.984 0.016])
        
        """
        dp = self.get_dew_point(IDs)
        z = self.get_normalized_mol(dp.IDs)
        return dp(z, P=P or self.P)
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kmol/hr')
        >>> s1.get_normalized_mol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imol[IDs]
        z_sum = z.sum()
        if not z_sum: raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kg/hr')
        >>> s1.get_normalized_mass(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imass[IDs]
        z_sum = z.sum()
        if not z_sum: raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_normalized_vol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.ivol[IDs]
        z_sum = z.sum()
        if not z_sum: raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kmol/hr')
        >>> s1.get_molar_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_mol = self.F_mol
        if not F_mol: raise RuntimeError(f'{repr(self)} is empty')
        return self.imol[IDs] / F_mol
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kg/hr')
        >>> s1.get_mass_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_mass = self.F_mass
        if not F_mass: raise RuntimeError(f'{repr(self)} is empty')
        return self.imass[IDs] / F_mass
    
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_volumetric_composition(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_vol = self.F_vol
        if not F_vol: raise RuntimeError(f'{repr(self)} is empty')
        return self.ivol[IDs] / F_vol
    
    def get_concentration(self, IDs):
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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_concentration(('Water', 'Ethanol'))
        array([27.671,  4.266])

        """
        F_vol = self.F_vol
        if not F_vol: raise RuntimeError(f'{repr(self)} is empty')
        return self.imol[IDs] / F_vol
    
    @property
    def P_vapor(self):
        """Vapor pressure of liquid."""
        chemicals = self.vle_chemicals
        F_l = eq.LiquidFugacities(chemicals, self.thermo)
        IDs = tuple([i.ID for i in chemicals])
        x = self.get_molar_composition(IDs)
        if x.sum() < 1e-12: return 0
        return F_l(x, self.T).sum()
    
    def receive_vent(self, other, accumulate=False, fraction=1.0):
        """
        Receive vapors from another stream as if in equilibrium.

        Notes
        -----
        Disregards energy balance. Assumes liquid composition will not change
        significantly.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol', tmo.Chemical('N2', phase='g')], cache=True)
        >>> tmo.settings.set_thermo(chemicals) 
        >>> s1 = tmo.Stream('s1', N2=10, units='m3/hr', phase='g', T=330)
        >>> s2 = tmo.Stream('s2', Water=10, Ethanol=2, T=330)
        >>> s1.receive_vent(s2, accumulate=True)
        >>> s1.show(flow='kmol/hr')
        Stream: s1
         phase: 'g', T: 330 K, P: 101325 Pa
         flow (kmol/hr): Water    0.0557
                         Ethanol  0.0616
                         N2       0.369
        """
        # TODO: Add energy balance 
        assert self.phase == 'g', 'stream must be a gas to receive vent'
        chemicals = other.vle_chemicals
        light_indices = other.chemicals._light_indices
        if accumulate:
            self.mol[light_indices] += other.mol[light_indices]
        else:
            self.mol[light_indices] = other.mol[light_indices]
        other.mol[light_indices] = 0
        F_l = eq.LiquidFugacities(chemicals, other.thermo)
        IDs = tuple([i.ID for i in chemicals])
        x = other.get_molar_composition(IDs)
        T = self.T
        P = self.P
        f_l = F_l(x, T)
        y = f_l / P
        imol = self.imol
        mol_old = imol[IDs]
        mol_new = fraction * self.F_mol * y
        if accumulate:
            imol[IDs] += mol_new            
        else:
            imol[IDs] = mol_new
        other.imol[IDs] += mol_old - mol_new 
        index = other.mol < 0.
        self.mol[index] += other.mol[index]
        other.mol[index] = 0
        
    ### Casting ###
    
    @property
    def link(self):
        """
        [Stream] Data on the thermal condition and material flow rates may 
        be shared with this stream.
        """
        return self._link
    
    @property
    def phases(self):
        """tuple[str] All phases present."""
        return (self.phase,)
    @phases.setter
    def phases(self, phases):
        if self._link: raise RuntimeError('cannot convert linked stream')
        if len(phases) == 1:
            self.phase = phases[0]
        else:
            self.__class__ = tmo.MultiStream
            self._imol = self._imol.to_material_indexer(phases)
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
    
    ### Representation ###
    
    def diagram(self, file=None, format='png'): # pragma: no cover
        from biosteam._digraph import make_digraph, save_digraph
        units = [i for i in (self.source, self.sink) if i]
        streams = sum([i.ins + i.outs for i in units], [])
        f = make_digraph(units, set(streams))
        save_digraph(f, file, format)
    
    def _basic_info(self):
        return f"{type(self).__name__}: {self.ID or ''}\n"
    
    def _info_phaseTP(self, phase, T_units, P_units):
        T = thermo_units.convert(self.T, 'K', T_units)
        P = thermo_units.convert(self.P, 'Pa', P_units)
        s = '' if isinstance(phase, str) else 's'
        return f" phase{s}: {repr(phase)}, T: {T:.5g} {T_units}, P: {P:.6g} {P_units}\n"
    
    def _source_info(self):
        source = self.source
        return f"{source}-{source.outs.index(self)}" if source else self.ID
    
    def _info(self, T, P, flow, composition, N, IDs):
        """Return string with all specifications."""
        from .indexer import nonzeros
        basic_info = self._basic_info()
        if not IDs:
            IDs = self.chemicals.IDs
            data = self.imol.data
        else:
            data = self.imol[IDs]
        IDs, data = nonzeros(IDs, data)
        IDs = tuple(IDs)
        display_units = self.display_units
        T_units = T or display_units.T
        P_units = P or display_units.P
        flow_units = flow or display_units.flow
        N_max = display_units.N if N is None else N
        composition = display_units.composition if composition is None else composition 
        basic_info += self._info_phaseTP(self.phase, T_units, P_units)
        N_IDs = len(IDs)
        if N_IDs == 0:
            return basic_info + ' flow: 0' 
        
        # Start of third line (flow rates)
        name, factor = self._get_flow_name_and_factor(flow_units)
        indexer = getattr(self, 'i' + name)
            
        # Remaining lines (all flow rates)
        flow_array = factor * indexer[IDs]
        if composition:
            total_flow = flow_array.sum()
            beginning = " composition: "
            new_line = '\n' + 14 * ' '
            flow_array = flow_array/total_flow
        else:
            beginning = f' flow ({flow_units}): '
            new_line = '\n' + len(beginning) * ' '
        flow_rates = ''
        lengths = [len(i) for i in IDs]
        maxlen = max(lengths) + 2
        too_many_chemicals = N_IDs > N_max
        N = N_max if too_many_chemicals else N_IDs
        for i in range(N):
            spaces = ' ' * (maxlen - lengths[i])
            if i: flow_rates += new_line
            flow_rates += IDs[i] + spaces + f'{flow_array[i]:.3g}'
        if too_many_chemicals: flow_rates += new_line + '...'
        if composition:
            dashes = '-' * (maxlen - 2)
            flow_rates += f"{new_line}{dashes}  {total_flow:.3g} {flow_units}"
        return (basic_info 
              + beginning
              + flow_rates)

    def show(self, T=None, P=None, flow=None, composition=None, N=None, IDs=None):
        """Print all specifications.
        
        Parameters
        ----------
        T : str, optional
            Temperature units.
        P : str, optional
            Pressure units.
        flow : str, optional
            Flow rate units.
        composition : bool, optional
            Whether to show composition.
        N : int, optional
            Number of compounds to display.
        IDs : tuple[str], optional
            IDs of compounds to display. Defaults to all chemicals
            .
        Notes
        -----
        Default values are stored in `Stream.display_units`.
        
        """
        print(self._info(T, P, flow, composition, N, IDs))
    _ipython_display_ = show
    
    def print(self, units=None):
        """
        Print in a format that you can use recreate the stream.
        
        Parameters
        ----------
        units : str, optional
            Units of measure for material flow rates. Defaults to 'kmol/hr'
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream(ID='s1',
        ...                 Water=20, Ethanol=10, units='kg/hr',
        ...                 T=298.15, P=101325, phase='l')
        >>> s1.print(units='kg/hr')
        Stream(ID='s1', phase='l', T=298.15, P=101325, Water=20, Ethanol=10, units='kg/hr')
        >>> s1.print() # Units default to kmol/hr
        Stream(ID='s1', phase='l', T=298.15, P=101325, Water=1.11, Ethanol=0.2171, units='kmol/hr')
        
        """
        if not units:
            units = 'kmol/hr'
            flow = self.mol
        else:
            flow = self.get_flow(units)
        chemical_flows = utils.repr_IDs_data(self.chemicals.IDs, flow)
        price = utils.repr_kwarg('price', self.price)
        print(f"{type(self).__name__}(ID={repr(self.ID)}, phase={repr(self.phase)}, T={self.T:.2f}, "
              f"P={self.P:.6g}{price}{chemical_flows}, units={repr(units)})")
