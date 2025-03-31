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
import pandas as pd
import numpy as np
import thermosteam as tmo
import flexsolve as flx
from thermosteam import functional as fn, Thermo
from . import indexer
from . import equilibrium as eq
from . import units_of_measure as UofM
from .exceptions import DimensionError, InfeasibleRegion
from chemicals.elements import array_to_atoms, symbol_to_index
from . import utils
from .indexer import nonzeros
from typing import TYPE_CHECKING
from ._phase import valid_phases
from .network import AbstractStream
from .nodes import VariableNode
if TYPE_CHECKING:
    from .base import SparseVector, SparseArray
    from numpy.typing import NDArray
    from typing import Optional, Sequence, Callable
# from .constants import g

MaterialIndexer = tmo.indexer.MaterialIndexer

__all__ = ('Stream',)

# %% Utilities

impact_indicator_basis = tmo.units_of_measure.UnitsOfMeasure('kg')
mol_units = indexer.ChemicalMolarFlowIndexer.units
mass_units = indexer.ChemicalMassFlowIndexer.units
vol_units = indexer.ChemicalVolumetricFlowIndexer.units


class StreamData:
    __slots__ = ('_flow', '_T', '_P', '_phases')

    def __init__(self, imol, thermal_condition, phases):
        self._flow = imol.data.copy()
        self._T = thermal_condition._T
        self._P = thermal_condition._P
        self._phases = phases


class TemporaryPhase:
    __slots__ = ('stream', 'original', 'temporary')

    def __init__(self, stream, original, temporary):
        self.stream = stream
        self.original = original
        self.temporary = temporary

    def __enter__(self):
        stream = self.stream
        stream.phase = self.temporary
        return stream

    def __exit__(self, type, exception, traceback):
        self.stream.phase = self.original
        if exception:
            raise exception


class TemporaryStream:
    __slots__ = ('stream', 'data', 'flow', 'T', 'P', 'phase')

    def __init__(self, stream, flow, T, P, phase):
        self.stream = stream
        self.data = stream.get_data()
        self.flow = flow
        self.T = T
        self.P = P
        self.phase = phase

    def __enter__(self):
        stream = self.stream
        if self.flow is not None:
            stream.imol.data[:] = self.flow
        if self.T is not None:
            stream.T = self.T
        if self.P is not None:
            stream.P = self.P
        return stream

    def __exit__(self, type, exception, traceback):
        self.stream.set_data(self.data)
        if exception:
            raise exception


class Equations:
    __slots__ = ('material', 'energy')

    def __init__(self):
        self.material = []
        self.energy = []

    def __repr__(self):
        return f"{type(self).__name__}(material={[i.__name__ for i in self.material]}(), energy={[i.__name__ for i in self.energy]}())"


# %%

@utils.define_units_of_measure(UofM.stream_units_of_measure)
@utils.thermo_user
class Stream(AbstractStream):
    """
    Create a Stream object that defines material flow rates
    along with its thermodynamic state. Thermodynamic and transport
    properties of a stream are available as properties, while
    thermodynamic equilbrium (e.g. VLE, and bubble and dew points)
    are available as methods. 

    Parameters
    ----------
    ID : 
        A unique identification. If ID is None, stream will not be registered.
        If no ID is given, stream will be registered with a unique ID.
    flow : 
        All flow rates corresponding to defined chemicals.
    phase : 
        'g' for gas, 'l' for liquid, and 's' for solid. Defaults to 'l'.
    T : 
        Temperature [K]. Defaults to 298.15.
    P : 
        Pressure [Pa]. Defaults to 101325.
    units : 
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid). Defaults to 'kmol/hr'.
    price : 
        Price per unit mass [USD/kg]. Defaults to 0.
    total_flow : 
        Total flow rate.
    thermo : 
        Thermo object to initialize input and output streams. Defaults to
        :meth:`settings.thermo <thermosteam._settings.ProcessSettings.thermo>`.
    characterization_factors : 
        Characterization factors for life cycle assessment.
    vlle : 
        Whether to run rigorous phase equilibrium to determine phases. 
        Defaults to False.
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
    composition (%): Water    66.7
                     Ethanol  33.3
                     -------  30 kg/hr

    All flow rates are stored as a sparse array in the `mol` attribute.
    These arrays work just like numpy arrays, but are more scalable
    (saving memory and increasing speed) for sparse chemical data:

    >>> s1.mol # Molar flow rates [kmol/hr]
    sparse([1.11 , 0.217])

    Mass and volumetric flow rates are also available for convenience:

    >>> s1.mass
    sparse([20., 10.])
    >>> s1.vol
    sparse([0.02 , 0.013])

    The data of these arrays are linked to the molar flows:

    >>> # Mass flows are always up to date with molar flows
    >>> s1.mol[0] = 1
    >>> s1.mass[0]
    18.015
    >>> # Changing mass flows changes molar flows
    >>> s1.mass[0] *= 2
    >>> s1.mol[0]
    2.0
    >>> # New arrays are not linked to molar flows
    >>> s1.mass + 2
    sparse([38.031, 12.   ])

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
    1083.46

    Other thermodynamic properties are temperature and pressure dependent as well:

    >>> s1.rho # Density [kg/m3]
    909.14

    It may be more convinient to get properties with different units:

    >>> s1.get_property('rho', 'g/cm3')
    0.9091

    It is also possible to set some of the properties in different units:

    >>> s1.set_property('T', 40, 'degC')
    >>> s1.T
    313.15

    Bubble point and dew point computations can be performed through stream methods:

    >>> bp = s1.bubble_point_at_P() # Bubble point at constant pressure
    >>> bp
    BubblePointValues(T=357.14, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.492 0.508])

    The bubble point results contain all results as attributes:

    >>> tmo.docround(bp.T) # Temperature [K]
    357.1442
    >>> bp.y # Vapor composition
    array([0.49, 0.51])

    Vapor-liquid equilibrium can be performed by setting 2 degrees of freedom from the following list: `T` [Temperature; in K], `P` [Pressure; in Pa], `V` [Vapor fraction], `H` [Enthalpy; in kJ/hr].

    Set vapor fraction and pressure of the stream:

    >>> s1.vle(P=101325, V=0.5)
    >>> s1.show()
    MultiStream: s1
    phases: ('g', 'l'), T: 364.78 K, P: 101325 Pa
    flow (kmol/hr): (g) Water    0.472
                        Ethanol  0.191
                    (l) Water    0.638
                        Ethanol  0.0257

    Note that the stream is a now a MultiStream object to manage multiple phases.
    Each phase can be accessed separately too:

    >>> s1['l'].show()
    Stream: 
    phase: 'l', T: 364.78 K, P: 101325 Pa
    flow (kmol/hr): Water    0.638
                    Ethanol  0.0257

    >>> s1['g'].show()
    Stream: 
    phase: 'g', T: 364.78 K, P: 101325 Pa
    flow (kmol/hr): Water    0.472
                    Ethanol  0.191

    We can convert a MultiStream object back to a Stream object by setting the phase:

    >>> s1.phase = 'l'
    >>> s1.show(flow='kg/hr')
    Stream: s1
    phase: 'l', T: 364.78 K, P: 101325 Pa
    flow (kg/hr): Water    20
                  Ethanol  10

    """
    __slots__ = (
        '_imol', '_thermal_condition', '_streams',
        '_vle_cache', '_lle_cache', '_sle_cache',
        '_price', '_property_cache_key',
        '_property_cache', 'characterization_factors',
        'equations',
        '_original',
        '_F_node',
        '_E_node',
        # '_velocity', '_height'
    )

    #: Units of measure for IPython display (class attribute)
    display_units = UofM.DisplayUnits(T='K', P='Pa',
                                      flow=('kmol/hr', 'kg/hr', 'm3/hr'),
                                      composition=False,
                                      sort=False,
                                      N=7)

    display_notation = UofM.DisplayNotation(T='.5g', P='.6g', flow='.3g')

    _units_of_measure = UofM.stream_units_of_measure

    _flow_cache = {}

    def __init__(self, ID: Optional[str] = '',
                 flow: Sequence[float] | Sequence[tuple[str, float]] = (),
                 phase: Optional[str] = 'l',
                 T: Optional[float] = 298.15,
                 P: Optional[float] = 101325.,
                 units: Optional[str] = None,
                 price: Optional[float] = 0.,
                 total_flow: Optional[float] = None,
                 thermo: Optional[Thermo] = None,
                 characterization_factors: Optional[dict[str, float]] = None,
                 vlle: Optional[bool] = False,
                 # velocity=0., height=0.,
                 **chemical_flows: float):
        self.equations: list[Callable] = Equations()
        #: Characterization factors for life cycle assessment [impact/kg].
        self.characterization_factors: dict[str, float] = {
        } if characterization_factors is None else {}
        self._thermal_condition = tmo.ThermalCondition(T, P)
        thermo = self._load_thermo(thermo)
        chemicals = thermo.chemicals
        self.price = price
        # self.velocity = velocity
        # self.height = height
        if units:
            name, factor = self._get_flow_name_and_factor(units)
            if name == 'mass':
                group_wt_compositions = chemicals._group_wt_compositions
                for cID in tuple(chemical_flows):
                    if cID in group_wt_compositions:
                        compositions = group_wt_compositions[cID]
                        group_flow = chemical_flows.pop(cID)
                        chemical_group = chemicals[cID]
                        for i in range(len(chemical_group)):
                            chemical_flows[chemical_group[i]._ID] = group_flow * \
                                compositions[i]
            elif name == 'vol':
                group_wt_compositions = chemicals._group_wt_compositions
                for cID in chemical_flows:
                    if cID in group_wt_compositions:
                        raise ValueError(
                            f"cannot set volumetric flow by chemical group '{i}'")
            self._init_indexer(flow, phase, chemicals, chemical_flows)
            mol = self.mol
            flow = getattr(self, name)
            if total_flow is not None:
                mol *= total_flow / mol.sum()
            material_data = mol / factor
            flow[:] = material_data
        else:
            self._init_indexer(flow, phase, chemicals, chemical_flows)
            if total_flow:
                mol = self.mol
                mol *= total_flow / mol.sum()
        self._sink = self._source = None
        self.reset_cache()
        self._register(ID)
        if vlle:
            self.vlle(T, P)
            data = self._imol.data
            self.phases = [j for i, j in enumerate(
                self.phases) if data[i].any()]

    def temporary(self, flow=None, T=None, P=None, phase=None):
        return TemporaryStream(self, flow, T, P, phase)

    def temporary_phase(self, phase):
        return TemporaryPhase(self, self.phase, phase)

    @classmethod
    def from_data(cls, data, ID=None, price=0., characterization_factors=None, thermo=None):
        self = cls.__new__(cls)
        self.__init__(
            ID,
            characterization_factors=characterization_factors,
            price=price,
            thermo=thermo,
        )
        self.set_data(data)
        return self

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, key):
        phase = self.phase
        if key.lower() == phase.lower():
            return self
        raise tmo.UndefinedPhase(phase)

    def __reduce__(self):
        return self.from_data, (self.get_data(), self._ID, self._price, self.characterization_factors, self._thermo)

    # Phenomena-oriented simulation
    @property
    def E_node(self):
        return self.source.E_node if self.source else None

    @property
    def F_node(self):
        source = self.source
        if source is None:
            return None
        if hasattr(self, '_F_node'):
            self._F_node.value = self.mol.to_array()
            return self._F_node
        self._F_node = var = VariableNode(f"{self.ID}.F",
            self.mol.to_array(),
            *source.get_connected_material_nodes(self),
            # TODO: Use get_connected_material_nodes here too
            *[i.sink.overall_material_balance_node for i in source.outs if i.sink],
        )
        return var

    @property
    def material_reference(self):
        imol = self._imol
        return (imol._parent, imol._phase)

    @property
    def material_equations(self):
        return self.equations.material

    @property
    def energy_equations(self):
        return self.equations.energy

    def material_balance(self, f=None):
        self.material_equations.append(f)
        return f

    def _create_material_balance_equations(self):
        return [i() for i in self.material_equations]

    def _create_energy_departure_equations(self):
        return [i() for i in self.energy_equations]

    def _update_energy_departure_coefficient(self, coefficients):
        source = self.source
        if source is None or not source._recycle_system:
            return
        if not source._get_energy_departure_coefficient:
            raise NotImplementedError(
                f'{source!r} has no method `_get_energy_departure_coefficient`')
        coeff = source._get_energy_departure_coefficient(self)
        if coeff is None:
            return
        key, value = coeff
        coefficients[key] = value

    def scale(self, scale):
        """
        Multiply flow rate by given scale.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=1)
        >>> s1.scale(100)
        >>> s1.F_mol
        100.0

        """
        self._imol.data *= scale
    rescale = scale

    def reset(self, T=None, P=None, **flows):
        """
        Convinience method for resetting flow rate and thermal condition data.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=1)
        >>> s1.reset(T=300.1, Ethanol=1, phase='g', units='kg/hr', total_flow=2)
        >>> s1.show('cwt')
        Stream: s1
        phase: 'g', T: 300.1 K, P: 101325 Pa
        composition (%): Ethanol  100
                         -------  2 kg/hr

        """
        self.reset_flow(**flows)
        if T is not None:
            self.T = T
        if P is not None:
            self.P = P

    def reset_flow(self, phase=None, units=None, total_flow=None, **chemical_flows):
        """
        Convinience method for resetting flow rate data.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=1)
        >>> s1.reset_flow(Ethanol=1, phase='g', units='kg/hr', total_flow=2)
        >>> s1.show('cwt')
        Stream: s1
        phase: 'g', T: 298.15 K, P: 101325 Pa
        composition (%): Ethanol  100
                         -------  2 kg/hr

        """
        imol = self._imol
        imol.empty()
        if phase:
            imol.phase = phase
        if chemical_flows:
            keys, values = zip(*chemical_flows.items())
            if units is None:
                self.imol[keys] = values
            else:
                self.set_flow(values, units, keys)
        if total_flow:
            if units is None:
                self.F_mol = total_flow
            else:
                self.set_total_flow(total_flow, units)

    def get_CF(self, key: str, basis: Optional[str] = None, units: Optional[str] = None):
        """
        Returns the life-cycle characterization factor on a kg basis given the
        impact indicator key.

        Parameters
        ----------
        key : 
            Key of impact indicator.
        basis :
            Basis of characterization factor. Mass is the only valid dimension (for now). 
            Defaults to 'kg'.
        units :
            Units of impact indicator. Before using this argument, the default units 
            of the impact indicator should be defined with 
            :meth:`settings.define_impact_indicator <thermosteam._settings.ProcessSettings.define_impact_indicator>`.
            Units must also be dimensionally consistent with the default units.

        """
        try:
            value = self.characterization_factors[key]
        except:
            return 0.
        if units is not None:
            original_units = tmo.settings.get_impact_indicator_units(key)
            value = original_units.convert(value, units)
        if basis is not None:
            value /= impact_indicator_basis.conversion_factor(basis)
        return value

    def set_CF(self, key: str, value: float, basis: Optional[str] = None, units: Optional[str] = None):
        """
        Set the life-cycle characterization factor on a kg basis given the 
        impact indicator key and the units of measure.

        Parameters
        ----------
        key : 
            Key of impact indicator.
        value : 
            Characterization factor value.
        basis :
            Basis of characterization factor. Mass is the only valid dimension (for now). 
            Defaults to 'kg'.
        units :
            Units of impact indicator. Before using this argument, the default units 
            of the impact indicator should be defined with 
            :meth:`settings.define_impact_indicator <thermosteam._settings.ProcessSettings.define_impact_indicator>`.
            Units must also be dimensionally consistent with the default units.

        """
        if units is not None:
            original_units = tmo.settings.get_impact_indicator_units(key)
            value = original_units.unconvert(value, units)
        if basis is not None:
            value *= impact_indicator_basis.conversion_factor(basis)
        self.characterization_factors[key] = value

    def get_impact(self, key):
        """Return hourly rate of the impact indicator given the key."""
        cfs = self.characterization_factors
        return cfs[key] * self.F_mass if key in cfs else 0.

    def empty_negative_flows(self):
        """
        Replace flows of all components with negative values with 0.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=1, Ethanol=-1)
        >>> s1.empty_negative_flows()
        >>> s1.show()
        Stream: s1
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kmol/hr): Water  1

        """
        self._imol.data.remove_negatives()

    def shares_flow_rate_with(self, other):
        """
        Return whether other stream shares data with this one.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> s1 = tmo.Stream('s1')
        >>> other = s1.flow_proxy()
        >>> s1.shares_flow_rate_with(other)
        True
        >>> s1 = tmo.MultiStream('s1', phases=('l', 'g'))
        >>> s1['g'].shares_flow_rate_with(s1)
        True
        >>> s2 = tmo.MultiStream('s2', phases=('l', 'g'))
        >>> s1['g'].shares_flow_rate_with(s2)
        False
        >>> s1['g'].shares_flow_rate_with(s2['g'])
        False
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
        return self._imol.data.shares_data_with(other._imol.data)

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
        return StreamData(self._imol, self._thermal_condition, self.phases)

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
            self._imol.data.copy_like(stream_data._flow)
            self._thermal_condition.copy_like(stream_data)
        else:
            raise ValueError('stream_data must be a StreamData object; '
                            f'not {type(stream_data).__name__}')

    @property
    def price(self) -> float:
        """Price of stream per unit mass [USD/kg]."""
        return self._price

    @price.setter
    def price(self, price):
        if np.isfinite(price):
            self._price = float(price)
        else:
            raise AttributeError(f'price must be finite, not {price}')

    # @property
    # def velocity(self) -> float:
    #     """Velocity of stream [m/s]."""
    #     return self._velocity
    # @velocity.setter
    # def velocity(self, velocity):
    #     if np.isfinite(velocity):
    #         self._velocity = float(velocity)
    #     else:
    #         raise AttributeError(f'velocity must be finite, not {velocity}')

    # @property
    # def height(self) -> float:
    #     """Relative height of stream [m]."""
    #     return self._height
    # @height.setter
    # def height(self, height):
    #     if np.isfinite(height):
    #         self._height = float(height)
    #     else:
    #         raise AttributeError(f'height must be finite, not {height}')

    # @property
    # def potential_energy(self) -> float:
    #     """Potential energy flow rate [kW]"""
    #     return (g * self.height * self.F_mass) / 3.6e6

    # @property
    # def kinetic_energy(self):
    #     """Kinetic energy flow rate [kW]"""
    #     return 0.5 * self.F_mass / 3.6e6 * self._velocity * self._velocity

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
        return self._imol.isempty()

    def sanity_check(self):
        """
        Raise an InfeasibleRegion error if flow rates are infeasible.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> s1 = tmo.Stream('s1')
        >>> s1.sanity_check()
        >>> s1.mol[0] = -1.
        >>> s1.sanity_check()
        Traceback (most recent call last):
        InfeasibleRegion: negative material flow rate is infeasible

        """
        material = self._imol.data
        if material.has_negatives():
            raise InfeasibleRegion('negative material flow rate')

    @property
    def vapor_fraction(self) -> float:
        """Molar vapor fraction."""
        return 1.0 if self.phase in 'gG' else 0.0

    @property
    def liquid_fraction(self) -> float:
        """Molar liquid fraction."""
        return 1.0 if self.phase in 'lL' else 0.0

    @property
    def solid_fraction(self) -> float:
        """Molar solid fraction."""
        return 1.0 if self.phase in 'sS' else 0.0

    @property
    def main_chemical(self) -> str:
        """ID of chemical with the largest mol fraction in stream."""
        return self.chemicals.tuple[self.mol.argmax()].ID

    def _init_indexer(self, flow, phase, chemicals, chemical_flows):
        """Initialize molar flow rates."""
        if len(flow) == 0:
            parent = indexer.parent_indexer(chemicals)
            if chemical_flows:
                imol = indexer.ChemicalMolarFlowIndexer(
                    phase, chemicals=chemicals, parent=parent, **chemical_flows)
            else:
                imol = indexer.ChemicalMolarFlowIndexer.blank(
                    phase, chemicals, parent=parent)
        else:
            if chemical_flows:
                ValueError(
                    "may specify either 'flow' or 'chemical_flows', but not both")
            if isinstance(flow, indexer.ChemicalMolarFlowIndexer):
                imol = flow
                imol.phase = phase
            else:
                parent = indexer.parent_indexer(chemicals)
                imol = parent.to_chemical_indexer(phase)
                if isinstance(flow[0], str):
                    ID, flow = flow
                    imol[ID] = flow
                else:
                    try:
                        IDs, flow = zip(*flow)
                    except:
                        imol.data[:] = flow
                    else:
                        imol[IDs] = flow
        self._imol = imol

    def reset_cache(self):
        """Reset cache regarding equilibrium methods."""
        self._property_cache_key = None, None
        self._property_cache = {}

    @classmethod
    def _get_flow_name_and_factor(cls, units):
        cache = cls._flow_cache
        if units in cache:
            name, factor = cache[units]
        else:
            dimensionality = UofM.get_dimensionality(units)
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
        Return flow rate of atom [kmol / hr] given the atomic symbol.

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
        Return dictionary of atomic flow rates [kmol / hr].

        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water'], cache=True)
        >>> stream = tmo.Stream(Water=1)
        >>> stream.get_atomic_flows()
        {'H': 2.0, 'O': 1.0}

        """
        return array_to_atoms(self.chemicals.formula_array @ self.mol)

    def get_flow(self, units: str, key: Optional[Sequence[str] | str] = ...):
        """
        Return an flow rates in requested units.

        Parameters
        ----------
        units : 
            Units of measure.
        key : 
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

    def set_flow(self, data: NDArray[float] | float, units: str, key: Optional[Sequence[str] | str] = ...):
        """
        Set flow rates in given units.

        Parameters
        ----------
        data : 
            Flow rate data.
        units : 
            Units of measure.
        key : 
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

    def get_total_flow(self, units: str):
        """
        Get total flow rate in given units.

        Parameters
        ----------
        units : 
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

    def set_total_flow(self, value: float, units: str):
        """
        Set total flow rate in given units keeping the composition constant.

        Parameters
        ----------
        value : 
            New total flow rate.
        units : 
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

    ### Stream data ###

    def get_downstream_units(self, ends=None, facilities=True):
        """Return a set of all units downstream."""
        sink = self._sink
        units = sink.get_downstream_units(ends, facilities)
        units.add(sink)
        return units

    def get_upstream_units(self, ends=None, facilities=True):
        """Return a set of all units upstream."""
        source = self._source
        units = source.get_upstream_units(ends, facilities)
        units.add(source)
        return units

    @property
    def thermal_condition(self) -> tmo.ThermalCondition:
        """
        Contains the temperature and pressure conditions 
        of the stream.
        """
        return self._thermal_condition

    @property
    def T(self) -> float:
        """Temperature [K]."""
        return self._thermal_condition._T

    @T.setter
    def T(self, T):
        self._thermal_condition._T = float(T)

    @property
    def P(self) -> float:
        """Pressure [Pa]."""
        return self._thermal_condition._P

    @P.setter
    def P(self, P):
        self._thermal_condition._P = float(P)

    @property
    def phase(self) -> str:
        """Phase of stream."""
        return self._imol._phase

    @phase.setter
    def phase(self, phase):
        self._imol.phase = phase

    @property
    def mol(self) -> NDArray[float]:
        """Molar flow rates [kmol/hr]."""
        return self._imol.data

    @mol.setter
    def mol(self, value):
        mol = self.mol
        if mol is not value:
            mol[:] = value

    @property
    def mass(self) -> SparseVector | SparseArray:
        """Mass flow rates [kg/hr]."""
        return self.imass.data

    @mass.setter
    def mass(self, value):
        mass = self.mass
        if mass is not value:
            mass[:] = value

    @property
    def vol(self) -> SparseVector | SparseArray:
        """Volumetric flow rates [m3/hr]."""
        return self.ivol.data

    @vol.setter
    def vol(self, value):
        vol = self.vol
        if vol is not value:
            vol[:] = value

    @property
    def imol(self) -> indexer.Indexer:
        """Flow rate indexer with data [kmol/hr]."""
        return self._imol

    @property
    def imass(self) -> indexer.Indexer:
        """Flow rate indexer with data [kg/hr]."""
        return self._imol.by_mass()

    @property
    def ivol(self) -> indexer.Indexer:
        """Flow rate indexer with data [m3/hr]."""
        return self._imol.by_volume(self._thermal_condition)

    ### Net flow properties ###

    @property
    def cost(self) -> float:
        """Total cost of stream [USD/hr]."""
        return self.price * self.F_mass

    @property
    def F_mol(self) -> float:
        """Total molar flow rate [kmol/hr]."""
        return self._imol.data.sum()

    @F_mol.setter
    def F_mol(self, value):
        F_mol = self.F_mol
        if not F_mol:
            raise AttributeError("undefined composition; cannot set flow rate")
        self._imol.data *= value/F_mol

    @property
    def F_mass(self) -> float:
        """Total mass flow rate [kg/hr]."""
        return np.dot(self.chemicals.MW, self.mol)

    @F_mass.setter
    def F_mass(self, value):
        F_mass = self.F_mass
        if F_mass:
            self.imol.data *= value/F_mass
        elif value:
            raise AttributeError("undefined composition; cannot set flow rate")
        else:
            self.empty()

    @property
    def F_vol(self) -> float:
        """Total volumetric flow rate [m3/hr]."""
        F_mol = self.F_mol
        return 1000. * self.V * F_mol if F_mol else 0.

    @F_vol.setter
    def F_vol(self, value):
        F_vol = self.F_vol
        if not F_vol:
            raise AttributeError("undefined composition; cannot set flow rate")
        self.imol.data *= value / F_vol

    @property
    def H(self) -> float:
        """Enthalpy flow rate [kJ/hr]."""
        return self._get_property('H', flow=True)

    @H.setter
    def H(self, H: float):
        if not H and self.isempty():
            return
        try:
            self.T = self.mixture.solve_T_at_HP(
                self.phase, self.mol, H, *self._thermal_condition
            )
        except Exception as error:  # pragma: no cover
            phase = self.phase.lower()
            if phase == 'g':
                # Maybe too little heat, liquid must be present
                self.phase = 'l'
            elif phase == 'l':
                # Maybe too much heat, gas must be present
                self.phase = 'g'
            else:
                raise error
            self.T = self.mixture.solve_T_at_HP(
                self.phase, self.mol, H, *self._thermal_condition
            )

    @property
    def h(self) -> float:
        """Specific enthalpy [kJ/kmol]."""
        return self._get_property('H')

    @h.setter
    def h(self, h: float):
        if not h and self.isempty():
            return
        z_mol = self.z_mol
        try:
            self.T = self.mixture.solve_T_at_HP(
                self.phase, z_mol, h, *self._thermal_condition
            )
        except Exception as error:  # pragma: no cover
            phase = self.phase.lower()
            if phase == 'g':
                # Maybe too little heat, liquid must be present
                self.phase = 'l'
            elif phase == 'l':
                # Maybe too much heat, gas must be present
                self.phase = 'g'
            else:
                raise error
            self.T = self.mixture.solve_T_at_HP(
                self.phase, z_mol, h, *self._thermal_condition
            )

    @property
    def S(self) -> float:
        """Absolute entropy flow rate [kJ/hr/K]."""
        return self._get_property('S', flow=True)

    @S.setter
    def S(self, S: float):
        if not S and self.isempty():
            return
        try:
            self.T = self.mixture.solve_T_at_SP(
                self.phase, self.mol, S, *self._thermal_condition
            )
        except Exception as error:  # pragma: no cover
            phase = self.phase.lower()
            if phase == 'g':
                # Maybe too little heat, liquid must be present
                self.phase = 'l'
            elif phase == 'l':
                # Maybe too much heat, gas must be present
                self.phase = 'g'
            else:
                raise error
            self.S = self.mixture.solve_T_at_SP(
                self.phase, self.mol, S, *self._thermal_condition
            )

    @property
    def Hnet(self) -> float:
        """Total enthalpy flow rate (including heats of formation) [kJ/hr]."""
        return self.H + self.Hf

    @Hnet.setter
    def Hnet(self, Hnet):
        self.H = Hnet - self.Hf

    @property
    def Hf(self) -> float:
        """Enthalpy of formation flow rate [kJ/hr]."""
        return (self.chemicals.Hf * self.mol).sum()

    @property
    def LHV(self) -> float:
        """Lower heating value flow rate [kJ/hr]."""
        return (self.chemicals.LHV * self.mol).sum()

    @property
    def HHV(self) -> float:
        """Higher heating value flow rate [kJ/hr]."""
        return (self.chemicals.HHV * self.mol).sum()

    @property
    def Hvap(self) -> float:
        """Enthalpy of vaporization flow rate [kJ/hr]."""
        return self._get_property('Hvap', flow=True, nophase=True)

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
            composition_key = composition.dct
            if nophase:
                literal = (thermal_condition._T, thermal_condition._P)
            else:
                phase = imol._phase
                literal = (phase, thermal_condition._T, thermal_condition._P)
            last_literal, last_composition_key = self._property_cache_key
            if literal == last_literal and (composition_key == last_composition_key):
                if name in property_cache:
                    value = property_cache[name]
                    return value * total if flow else value
            else:
                property_cache.clear()
            self._property_cache_key = (literal, composition_key.copy())
            calculate = getattr(self.mixture, name)
            if nophase:
                property_cache[name] = value = calculate(
                    composition, *self._thermal_condition
                )
            else:
                property_cache[name] = value = calculate(
                    phase, composition, *self._thermal_condition
                )
            return value * total if flow else value

    @property
    def C(self) -> float:
        """Isobaric heat capacity flow rate [kJ/K/hr]."""
        return self._get_property('Cn', flow=True)

    ### Composition properties ###

    @property
    def z_mol(self) -> NDArray[float]:
        """Molar composition."""
        mol = self.mol
        z = mol / mol.sum()
        z = z.to_array()
        z.setflags(0)
        return z

    @property
    def z_mass(self) -> NDArray[float]:
        """Mass composition."""
        mass = self.chemicals.MW * self.mol
        F_mass = mass.sum()
        if F_mass == 0:
            z = mass
        else:
            z = mass / mass.sum()
        z.setflags(0)
        return z

    @property
    def z_vol(self) -> NDArray[float]:
        """Volumetric composition."""
        vol = self.vol.to_array()
        z = vol / vol.sum()
        z.setflags(0)
        return z

    @property
    def MW(self) -> float:
        """Overall molecular weight."""
        return self.mixture.MW(self.mol)

    @property
    def V(self) -> float:
        """Molar volume [m^3/mol]."""
        return self._get_property('V')

    @property
    def kappa(self) -> float:
        """Thermal conductivity [W/m/k]."""
        return self._get_property('kappa')

    @property
    def Cn(self) -> float:
        """Molar isobaric heat capacity [J/mol/K]."""
        return self._get_property('Cn')

    @property
    def mu(self) -> float:
        """Hydrolic viscosity [Pa*s]."""
        return self._get_property('mu')

    @property
    def sigma(self) -> float:
        """Surface tension [N/m]."""
        return self._get_property('sigma', nophase=True)

    @property
    def epsilon(self) -> float:
        """Relative permittivity [-]."""
        return self._get_property('epsilon', nophase=True)

    @property
    def Cp(self) -> float:
        """Isobaric heat capacity [J/g/K]."""
        return self.Cn / self.MW

    @property
    def alpha(self) -> float:
        """Thermal diffusivity [m^2/s]."""
        return fn.alpha(self.kappa,
                        self.rho,
                        self.Cp * 1000.)

    @property
    def rho(self) -> float:
        """Density [kg/m^3]."""
        V = self.V
        if V is None:
            return V
        return fn.V_to_rho(V, self.MW)

    @property
    def nu(self) -> float:
        """Kinematic viscosity [m^2/s]."""
        mu = self.mu
        if mu is None:
            return mu
        return fn.mu_to_nu(mu, self.rho)

    @property
    def Pr(self) -> float:
        """Prandtl number [-]."""
        return fn.Pr(self.Cp * 1000,
                     self.kappa,
                     self.mu)

    ### Stream methods ###

    @property
    def available_chemicals(self) -> list[tmo.Chemical]:
        """All chemicals with nonzero flow."""
        chemicals = self.chemicals.tuple
        return [chemicals[i] for i in self.mol.nonzero_keys()]

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
    def sum(cls, streams, ID=None, thermo=None, energy_balance=True, vle=False):
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
        new = cls(ID, thermo=thermo)
        if streams:
            new.copy_thermal_condition(streams[0])
        new.mix_from(streams, energy_balance, vle)
        return new

    def separate_out(self, other, energy_balance=True):
        """
        Separate out given stream from this one.

        Examples
        --------
        Separate out another stream with the same thermodynamic property package:

        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=30, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2', Water=10, Ethanol=5, units='kg/hr')
        >>> s1.separate_out(s2)
        >>> s1.show(flow='kg/hr')
        Stream: s1
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kg/hr): Water    20
                      Ethanol  5

        It's also possible to separate out streams with different property packages
        so long as all chemicals are defined in the mixed stream's property 
        package:

        >>> tmo.settings.set_thermo(['Water'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=40, units='kg/hr')
        >>> tmo.settings.set_thermo(['Ethanol'], cache=True) 
        >>> s2 = tmo.Stream('s2', Ethanol=20, units='kg/hr')
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s_mix = tmo.Stream.sum([s1, s2], 's_mix')
        >>> s_mix.separate_out(s2)
        >>> s_mix.show(flow='kg/hr')
        Stream: s_mix
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kg/hr): Water  40

        Removing empty streams is fine too:

        >>> s1.empty(); s_mix.separate_out(s1)
        >>> s_mix.show(flow='kg/hr')
        Stream: s_mix
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kg/hr): Water  40

        """
        if other:
            if self is other:
                self.empty()
            if energy_balance:
                H_new = self.H - other.H
            self._imol.separate_out(other._imol)
            if energy_balance:
                self.H = H_new

    def mix_from(self, others, energy_balance=True, vle=False, Q=0., conserve_phases=False):
        """
        Mix all other streams into this one, ignoring its initial contents.

        Notes
        -----
        When streams at different pressures are mixed, BioSTEAM assumes valves 
        reduce the pressure of the streams being mixed to prevent backflow 
        (pressure needs to decrease in the direction of flow according to 
        Bernoulli's principle). The outlet pressure will be the minimum pressure
        of all streams being mixed.

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

        Mixing empty streams is fine too:

        >>> s1.empty(); s2.empty(); s_mix.mix_from([s1, s2])
        >>> s_mix.show()
        Stream: s_mix
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow: 0

        """
        streams = []
        isa = isinstance
        for i in others:
            if isa(i, Stream):
                if not i.isempty():
                    streams.append(i)
            elif i:
                Q += i.heat  # Must be a heat or power object, assume power turns to heat
        N_streams = len(streams)
        if N_streams == 0:
            self.empty()
        elif N_streams == 1:
            if energy_balance:
                self.copy_like(streams[0])
            else:
                self.copy_flow(streams[0])
        else:
            self.P = P = min([i.P for i in streams])
            if conserve_phases:
                phases = self.phase + ''.join([i.phase for i in others])
                self.phases = phases
            if vle:
                self._imol.mix_from([i._imol for i in streams])
                if energy_balance:
                    H = sum([i.H for i in streams], Q)
                    self.vle(H=H, P=P)
                else:
                    self.vle(T=self.T, P=P)
                self.reduce_phases()
            else:
                if energy_balance:
                    self._imol.mix_from([i._imol for i in streams])
                    H = sum([i.H for i in streams], Q)
                    if conserve_phases:
                        self.H = H
                    else:
                        try:
                            self.H = H
                        except:
                            self.phases = self.phase + \
                                ''.join([i.phase for i in others])
                            self._imol.mix_from([i._imol for i in streams])
                            self.H = H
                else:
                    self._imol.mix_from([i._imol for i in streams])

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
        chemicals = self.chemicals
        values = mol * split
        dummy = mol - values
        if energy_balance:
            tc1 = s1._thermal_condition
            tc2 = s2._thermal_condition
            tc = self._thermal_condition
            tc1._T = tc2._T = tc._T
            tc1._P = tc2._P = tc._P
            s1.phase = s2.phase = self.phase
        if s1.chemicals is chemicals:
            s1.mol[:] = values
        else:
            CASs, values = zip(*[(i, j)
                               for i, j in zip(chemicals.CASs, values) if j])
            s1.empty()
            s1._imol[CASs] = values
        values = dummy
        if s2.chemicals is chemicals:
            s2.mol[:] = values
        else:
            s2.empty()
            CASs, values = zip(*[(i, j)
                               for i, j in zip(chemicals.CASs, values) if j])
            s2._imol[CASs] = values

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
        if isinstance(other.imol, MaterialIndexer):
            phases = other.phases
            if len(phases) == 1:
                phase, = phases
                self.phase = phase
                self.mol.copy_like(other.imol[phase])
                return
            else:
                self.empty()
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

    def copy_phase(self, other):
        """Copy phase from another stream."""
        try:
            self._imol.phase = other._imol._phase
        except AttributeError as e:
            if isinstance(other, tmo.MultiStream):
                raise ValueError(
                    'cannot copy phase from stream with multiple phases')
            raise e from None

    def copy_flow(self,
                  other: Stream,
                  IDs: Optional[Sequence[str] | str] = ..., *,
                  remove: Optional[bool] = False,
                  exclude: Optional[bool] = False):
        """
        Copy flow rates of another stream to self.

        Parameters
        ----------
        other : 
            Flow rates will be copied from here.
        IDs : 
            Chemical IDs. 
        remove : 
            If True, copied chemicals will be removed from `stream`.
        exclude :
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

        Copy flows except except water and remove water:

        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, units='kg/hr')
        >>> s2 = tmo.Stream('s2')
        >>> s2.copy_flow(s1, 'Water', exclude=True, remove=True)
        >>> s1.show('wt')
        Stream: s1
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kg/hr): Water  20
        >>> s2.show('wt')
        Stream: s2
        phase: 'l', T: 298.15 K, P: 101325 Pa
        flow (kg/hr): Ethanol  10

        """
        other_mol = other.mol
        other_chemicals = other.chemicals
        chemicals = self.chemicals
        if IDs == ...:
            if exclude:
                return
            if chemicals is other_chemicals:
                self.mol[:] = other.mol
            else:
                self.empty()
                CASs = other_chemicals.CASs
                dct = other_mol.dct
                CASs = [CASs[i] for i in dct]
                values = list(dct.values())
                self.imol[CASs] = values
            if remove:
                if isinstance(other, tmo.MultiStream):
                    other.imol.data.clear()
                else:
                    other_mol.clear()
        else:
            if exclude:
                if isinstance(IDs, str):
                    if IDs in other_chemicals:
                        bad_index = other_chemicals.index(IDs)
                        other_index = [i for i in range(
                            other_chemicals.size) if i != bad_index]
                    else:
                        other_index = slice()
                else:
                    IDs = [i for i in IDs if i in other_chemicals]
                    bad_index = set(other_chemicals.indices(IDs))
                    if bad_index:
                        other_index = [i for i in range(
                            other_chemicals.size) if i not in bad_index]
                    else:
                        other_index = slice()
            else:
                other_index = other_chemicals.get_index(IDs)
            if chemicals is other_chemicals:
                self.mol[other_index] = other_mol[other_index]
            else:
                CASs = other_chemicals.CASs
                other_index = [
                    i for i in other_index if other_mol[i] or CASs[i] in chemicals]
                self.imol[tuple([CASs[i] for i in other_index])
                          ] = other_mol[other_index]
            if remove:
                if isinstance(other, tmo.MultiStream):
                    other.imol.data[:, other_index] = 0
                else:
                    other_mol[other_index] = 0

    def copy(self, ID=None, thermo=None):
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
        Prices, and LCA characterization factors are not copied.

        """
        cls = self.__class__
        if thermo is not None:
            new = cls(ID=ID, thermo=thermo)
            new.copy_like(self)
        else:
            new = cls.__new__(cls)
            new.equations = Equations()
            new._sink = new._source = None
            new.characterization_factors = {}
            new._thermo = self._thermo
            new._imol = self._imol.full_copy()
            new._thermal_condition = self._thermal_condition.copy()
            new.reset_cache()
            new.price = 0
            new.ID = ID
        return new
    __copy__ = copy

    def link_with(self, other: Stream):
        """
        Link with another stream.

        Parameters
        ----------
        other : 

        See Also
        --------
        :obj:`~Stream.flow_proxy`
        :obj:`~Stream.proxy`

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
        self._imol = other._imol
        self._thermal_condition = other._thermal_condition

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

        >>> s1.phases = s2.phases = ('l', 'g')
        >>> s2.link_with(s1)
        >>> s1.imol.data is s2.imol.data
        True
        >>> s1.unlink()
        >>> s1.imol.data is s2.imol.data
        False

        MultiStream phases cannot be unlinked:

        >>> s1 = tmo.MultiStream(None, phases=('l', 'g'))
        >>> s1['g'].unlink()
        Traceback (most recent call last):
        RuntimeError: phase is locked; stream cannot be unlinked

        """
        imol = self._imol
        if hasattr(imol, '_phase'):
            if imol._lock_phase:
                raise RuntimeError(
                    'phase is locked; stream cannot be unlinked')
        self._imol = imol.full_copy()
        self._thermal_condition = self._thermal_condition.copy()
        self.reset_cache()

    def flow_proxy(self, ID=None):
        """
        Return a new stream that shares flow rate data with this one.

        See Also
        --------
        :obj:`~Stream.link_with`
        :obj:`~Stream.proxy`

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
        new._ID = ID or ''
        new._sink = new._source = None
        new._price = 0
        new._thermo = self._thermo
        new._imol = imol = self._imol._copy_without_data()
        imol.data = self._imol.data
        new._thermal_condition = self._thermal_condition.copy()
        new.reset_cache()
        new.equations = Equations()
        new.characterization_factors = {}
        return new

    def proxy(self, ID=None):
        """
        Return a new stream that shares all data with this one.

        See Also
        --------
        :obj:`~Stream.link_with`
        :obj:`~Stream.flow_proxy`

        Warning
        -------
        Price and characterization factor data is not shared

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
        new._original = self
        new._ID = ID or ''
        new._sink = new._source = None
        new._price = self._price
        new._thermo = self._thermo
        new._imol = self._imol
        new._thermal_condition = self._thermal_condition
        new._property_cache = self._property_cache
        new._property_cache_key = self._property_cache_key
        new.equations = self.equations
        new.characterization_factors = self.characterization_factors
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
        0

        """
        self._imol.data.clear()

    ### Equilibrium ###

    @property
    def vle(self) -> eq.VLE:
        """An object that can perform vapor-liquid equilibrium on the stream."""
        if self.phase == 's':
            self.phase = 'l'
        self.phases = ('g', 'l')
        return self.vle

    @property
    def lle(self) -> eq.LLE:
        """An object that can perform liquid-liquid equilibrium on the stream."""
        if self.phase not in ('l', 'L'):
            self.phase = 'l'
        self.phases = ('L', 'l')
        return self.lle

    @property
    def sle(self) -> eq.SLE:
        """An object that can perform solid-liquid equilibrium on the stream."""
        if self.phase not in ('l', 's'):
            self.phase = 'l'
        self.phases = ('s', 'l')
        return self.sle

    def vlle(self, T, P):
        """
        Estimate vapor-liquid-liquid equilibrium.

        Warning
        -------
        This method may be as slow as 1 second.

        """
        self.phases = ('L', 'g', 'l')
        imol = self.imol
        vle = eq.VLE(imol,
                     self._thermal_condition,
                     self._thermo)
        lle = eq.LLE(imol,
                     self._thermal_condition,
                     self._thermo)
        data = self._imol.data
        LIQ, gas, liq = data
        liq += LIQ  # All flows must be in the 'l' phase for VLE
        LIQ[:] = 0.
        vle(T=T, P=P)
        if not gas.any():
            lle(T, P)
            return
        elif not liq.any():
            return
        lle(T, P)
        if not (LIQ.any() and liq.any()):
            return
        total_flow = data.sum()

        def f(x, done=[False]):
            if done[0]:
                return x
            data[:] = x
            lle(T=T, P=P)
            vle(T=T, P=P)
            liq[:], LIQ[:] = LIQ, liq.copy()
            vle(T=T, P=P)
            liq[:], LIQ[:] = LIQ, liq.copy()
            return data.to_array()
        flx.fixed_point(
            f, data / total_flow, xtol=1e-6,
            checkiter=False, checkconvergence=False,
            convergenceiter=10
        )
        if np.abs(liq - LIQ).sum() < 1e-6:
            liq += LIQ
            LIQ.clear()
        data *= total_flow

    @property
    def vle_chemicals(self) -> list[tmo.Chemical]:
        """Chemicals cabable of liquid-liquid equilibrium."""
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.get_vle_indices(self.mol.nonzero_keys())
        return [chemicals_tuple[i] for i in indices]

    @property
    def lle_chemicals(self) -> list[tmo.Chemical]:
        """Chemicals cabable of vapor-liquid equilibrium."""
        chemicals = self.chemicals
        chemicals_tuple = chemicals.tuple
        indices = chemicals.get_lle_indices(self.mol.nonzero_keys())
        return [chemicals_tuple[i] for i in indices]

    def get_bubble_point(self, IDs: Optional[Sequence[str]] = None):
        """
        Return a BubblePoint object capable of computing bubble points.

        Parameters
        ----------
        IDs : 
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.get_bubble_point()
        BubblePoint([Water, Ethanol])

        """
        return eq.BubblePoint(self.chemicals[IDs] if IDs else self.vle_chemicals, self._thermo)

    def get_dew_point(self, IDs: Optional[Sequence[str]] = None):
        """
        Return a DewPoint object capable of computing dew points.

        Parameters
        ----------
        IDs :
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.get_dew_point()
        DewPoint([Water, Ethanol])

        """
        return eq.DewPoint(self.chemicals[IDs] if IDs else self.vle_chemicals, self._thermo)

    def bubble_point_at_T(self, T: Optional[float] = None, IDs: Optional[Sequence[str]] = None):
        """
        Return a BubblePointResults object with all data on the bubble point at constant temperature.

        Parameters
        ----------
        T :
            Temperature [K].
        IDs : 
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.bubble_point_at_T()
        BubblePointValues(T=350.00, P=76463, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.488 0.512])

        """
        bp = self.get_bubble_point(IDs)
        z = self.get_normalized_mol(bp.IDs)
        return bp(z, T=T or self.T)

    def bubble_point_at_P(self, P: Optional[float] = None, IDs: Optional[Sequence[str]] = None):
        """
        Return a BubblePointResults object with all data on the bubble point at constant pressure.

        Parameters
        ----------
        IDs :
            Chemicals that participate in equilibrium. Defaults to all chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.bubble_point_at_P()
        BubblePointValues(T=357.14, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], y=[0.492 0.508])

        """
        bp = self.get_bubble_point(IDs)
        z = self.get_normalized_mol(bp.IDs)
        return bp(z, P=P or self.P)

    def dew_point_at_T(self, T: Optional[float] = None, IDs: Optional[Sequence[str]] = None):
        """
        Return a DewPointResults object with all data on the dew point
        at constant temperature.

        Parameters
        ----------
        IDs : 
            Chemicals that participate in equilibrium. Defaults to all
            chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.dew_point_at_T()
        DewPointValues(T=350.00, P=49058, IDs=('Water', 'Ethanol'), z=[0.836 0.164], x=[0.984 0.016])

        """
        dp = self.get_dew_point(IDs)
        z = self.get_normalized_mol(dp.IDs)
        return dp(z, T=T or self.T)

    def dew_point_at_P(self, P: Optional[float] = None, IDs: Optional[Sequence[str]] = None):
        """
        Return a DewPointResults object with all data on the dew point
        at constant pressure.

        Parameters
        ----------
        IDs : 
            Chemicals that participate in equilibrium. Defaults to all
            chemicals in equilibrium.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, T=350, units='kg/hr')
        >>> s1.dew_point_at_P()
        DewPointValues(T=368.62, P=101325, IDs=('Water', 'Ethanol'), z=[0.836 0.164], x=[0.983 0.017])

        """
        dp = self.get_dew_point(IDs)
        z = self.get_normalized_mol(dp.IDs)
        return dp(z, P=P or self.P)

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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kmol/hr')
        >>> s1.get_normalized_mol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imol[IDs]
        z_sum = z.sum()
        if not z_sum:
            raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum

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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kg/hr')
        >>> s1.get_normalized_mass(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.imass[IDs]
        z_sum = z.sum()
        if not z_sum:
            raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum

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
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_normalized_vol(('Water', 'Ethanol'))
        array([0.667, 0.333])

        """
        z = self.ivol[IDs]
        z_sum = z.sum()
        if not z_sum:
            raise RuntimeError(f'{repr(self)} is empty')
        return z / z_sum

    def get_molar_fraction(self, IDs: Sequence[str]):
        """
        Return molar fraction of given chemicals.

        Parameters
        ----------
        IDs : 
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kmol/hr')
        >>> s1.get_molar_fraction(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_mol = self.F_mol
        return self.imol[IDs] / F_mol if F_mol else 0.

    get_molar_composition = get_molar_fraction

    def get_mass_fraction(self, IDs: Sequence[str]):
        """
        Return mass fraction of given chemicals.

        Parameters
        ----------
        IDs : 
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='kg/hr')
        >>> s1.get_mass_fraction(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_mass = self.F_mass
        return self.imass[IDs] / F_mass if F_mass else 0.

    get_mass_composition = get_mass_fraction

    def get_volumetric_fraction(self, IDs: Sequence[str]):
        """
        Return volumetric fraction of given chemicals.

        Parameters
        ----------
        IDs : 
            IDs of chemicals.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_volumetric_fraction(('Water', 'Ethanol'))
        array([0.5 , 0.25])

        """
        F_vol = self.F_vol
        return self.ivol[IDs] / F_vol if F_vol else 0.

    get_volumetric_composition = get_volumetric_fraction

    def get_concentration(self, IDs: Sequence[str], units: Optional[str] = None):
        """
        Return concentration of given chemicals.

        Parameters
        ----------
        IDs : 
            IDs of chemicals.
        units :
            Units of measure. Defaults to kmol/m3.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True) 
        >>> s1 = tmo.Stream('s1', Water=20, Ethanol=10, Methanol=10, units='m3/hr')
        >>> s1.get_concentration(['Water', 'Ethanol']) # kg/m3
        array([27.673,  4.261])

        >>> s1.get_concentration(['Water', 'Ethanol'], 'g/L')
        array([498.532, 196.291])

        """
        F_vol = self.F_vol
        if F_vol == 0.:
            return 0.
        if units is None:
            return self.imol[IDs] / F_vol
        else:
            num, denum = units.split('/')
            return self.get_flow(num+'/hr', IDs) / self.get_total_flow(denum+'/hr')

    @property
    def P_vapor(self) -> float:
        """Vapor pressure of liquid."""
        chemicals = self.vle_chemicals
        F_l = eq.LiquidFugacities(chemicals, self.thermo)
        IDs = tuple([i.ID for i in chemicals])
        x = self.get_molar_fraction(IDs)
        if x.sum() < 1e-12:
            return 0
        return F_l(x, self.T).sum()

    def receive_vent(self, other, energy_balance=True, ideal=False):
        """
        Receive vapors from another stream by vapor-liquid equilibrium between 
        a gas and liquid stream assuming only a small amount of chemicals
        in vapor-liquid equilibrium is present

        Examples
        --------
        The energy balance is performed by default:

        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol', tmo.Chemical('N2', phase='g')], cache=True)
        >>> tmo.settings.set_thermo(chemicals) 
        >>> s1 = tmo.Stream('s1', N2=20, units='m3/hr', phase='g', T=330)
        >>> s2 = tmo.Stream('s2', Water=10, Ethanol=2, T=330)
        >>> s1.receive_vent(s2)
        >>> s1.show(flow='kmol/hr')
        Stream: s1
        phase: 'g', T: 323.12 K, P: 101325 Pa
        flow (kmol/hr): Water    0.0799
                        Ethanol  0.0887
                        N2       0.739

        Set energy balance to false to receive vent isothermally:

        >>> import thermosteam as tmo
        >>> chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol', tmo.Chemical('N2', phase='g')], cache=True)
        >>> tmo.settings.set_thermo(chemicals) 
        >>> s1 = tmo.Stream('s1', N2=20, units='m3/hr', phase='g', T=330)
        >>> s2 = tmo.Stream('s2', Water=10, Ethanol=2, T=330)
        >>> s1.receive_vent(s2, energy_balance=False)
        >>> s1.show(flow='kmol/hr')
        Stream: s1
        phase: 'g', T: 330 K, P: 101325 Pa
        flow (kmol/hr): Water    0.112
                        Ethanol  0.123
                        N2       0.739

        """
        assert self.phase == 'g', 'stream must be a gas to receive vent'
        thermo = self.thermo.ideal() if ideal else self.thermo
        T = self.T
        P = self.P
        ms = tmo.Stream(None, T=T, P=P, thermo=thermo)
        ms.mix_from([self, other], energy_balance=False)
        if energy_balance:
            ms.H = H = self.H + other.H
        ms.vle._setup()
        vapor = ms['g']
        liquid = ms['l']
        for chemical in ms.chemicals:
            try:
                Psat = chemical.Psat(T)
            except:
                continue
            ID = chemical.ID
            if Psat < P:
                liquid.imol[ID] = ms.imol[ID]
                vapor.imol[ID] = 0.
            else:
                vapor.imol[ID] = ms.imol[ID]
                liquid.imol[ID] = 0.
        chemicals = ms.vle_chemicals
        F_l = eq.LiquidFugacities(chemicals, thermo)
        IDs = tuple([i.ID for i in chemicals])
        x = other.get_molar_fraction(IDs)
        F_mol_vapor = vapor.F_mol
        mol = liquid.imol[IDs] + vapor.imol[IDs]
        if energy_balance:
            def equilibrium_approximation(T):
                f_l = F_l(x, T)
                y = f_l / P
                mol_v = F_mol_vapor * y
                vapor.imol[IDs] = mol_v
                liquid.imol[IDs] = mol - mol_v
                index = liquid.mol.negative_index()
                vapor.mol[index] += liquid.mol[index]
                liquid.mol[index] = 0
                ms.H = H
                return ms.T
            flx.wegstein(equilibrium_approximation, T, xtol=1e-4, maxiter=100)
        else:
            f_l = F_l(x, T)
            y = f_l / P
            mol_v = F_mol_vapor * y
            vapor.imol[IDs] = mol_v
            liquid.imol[IDs] = mol - mol_v
            index = liquid.mol.negative_index()
            vapor.mol[index] += liquid.mol[index]
            liquid.mol[index] = 0
        self.copy_like(vapor)
        other.copy_like(liquid)
        self.T = other.T = ms.T

    ### Casting ###

    @property
    def phases(self) -> tuple[str, ...]:
        """All phases present."""
        return (self.phase,)

    @phases.setter
    def phases(self, phases):
        phases = set(phases)
        if len(phases) == 1:
            self.phase, = phases
        else:
            self.__class__ = tmo.MultiStream
            self._imol = self._imol.to_material_indexer(phases)
            self._streams = {}
            self._vle_cache = eq.VLECache(self._imol,
                                          self._thermal_condition,
                                          self._thermo)
            self._lle_cache = eq.LLECache(self._imol,
                                          self._thermal_condition,
                                          self._thermo)
            self._sle_cache = eq.SLECache(self._imol,
                                          self._thermal_condition,
                                          self._thermo)

    def reduce_phases(self):
        """Remove empty phases."""

    ### Representation ###

    def _info_phaseTP(self, phase, units, notation):
        T_units = units['T']
        P_units = units['P']
        T = UofM.convert(self.T, 'K', T_units)
        P = UofM.convert(self.P, 'Pa', P_units)
        s = '' if isinstance(phase, str) else 's'
        return f"phase{s}: {repr(phase)}, T: {T:{notation['T']}} {T_units}, P: {P:{notation['P']}} {P_units}\n"

    def _translate_layout(self, layout, flow, composition, N, sort):
        if layout:
            layout = layout.replace(' ', '')
            if layout[-1] == 's':
                sort = True
                layout = layout[:-1]
            if layout[0] in ('c', '%'):
                composition = True
                layout = layout[1:]
            if layout.startswith('wt'):
                flow = 'kg/hr'
                layout = layout[2:]
            elif layout.startswith('mol'):
                flow = 'kmol/hr'
                layout = layout[3:]
            elif layout.startswith('vol'):
                flow = 'm3/hr'
                layout = layout[3:]
            elif layout.isdigit():
                flow = 'kmol/hr'
            else:
                raise ValueError(
                    "`layout` must have the form "
                    "{'c' or ''}{'wt', 'mol' or 'vol'}{# or ''}{'s' or ''};"
                    "for example: 'cwt100s' corresponds to compostion=True, "
                    "flow='kg/hr', N=100, sort=True"
                )
            if layout and layout[0] == '%':
                composition = True
                layout = layout[1:]
            if layout.isdigit():
                N = int(layout)
        return flow, composition, N, sort

    def get_display_units_and_notation(self, **kwargs):
        display_units = self.display_units
        display_notation = self.display_notation
        units_dct = {}
        notation_dct = {}
        for name, value in kwargs.items():
            units, notation = UofM.parse_units_notation(value)
            units_dct[name] = getattr(
                display_units, name) if units is None else units
            notation_dct[name] = getattr(
                display_notation, name) if notation is None else notation
        return units_dct, notation_dct

    def _info_str(self, units, notation, composition, N_max, all_IDs, indexer, factor):
        basic_info = self._basic_info()
        basic_info += self._info_phaseTP(self.phase, units, notation)
        flow_units = units['flow']
        flow_notation = notation['flow']
        if N_max == 0:
            return basic_info[:-1]
        N_IDs = len(all_IDs)
        if N_IDs == 0:
            return basic_info + 'flow: 0'

        # Remaining lines (all flow rates)
        flow_array = factor * indexer[all_IDs]
        if composition:
            total_flow = flow_array.sum()
            beginning = "composition (%): "
            new_line = '\n' + len(beginning) * ' '
            flow_array = 100 * flow_array/total_flow
        else:
            beginning = f'flow ({flow_units}): '
            new_line = '\n' + len(beginning) * ' '
        flow_rates = ''
        too_many_chemicals = N_IDs > N_max
        if not too_many_chemicals:
            N_max = N_IDs
        lengths = [len(i) for i in all_IDs[:N_max]]
        maxlen = max(lengths) + 2
        for i in range(N_max):
            spaces = ' ' * (maxlen - lengths[i])
            if i:
                flow_rates += new_line
            flow_rates += all_IDs[i] + spaces + \
                f'{flow_array[i]:{flow_notation}}'
        if too_many_chemicals:
            spaces = ' ' * (maxlen - 3)
            flow_rates += new_line + '...' + spaces + \
                f'{flow_array[N_max:].sum():{flow_notation}}'
        if composition:
            dashes = '-' * (maxlen - 2)
            flow_rates += f"{new_line}{dashes}  {
                total_flow:{flow_notation}} {flow_units}"
        return (basic_info
                + beginning
                + flow_rates)

    def _info_df(self, units, notation, composition, N_max, all_IDs, indexer, factor):
        if not all_IDs:
            return pd.DataFrame([0], columns=[self.ID.replace('_', ' ')], index=['Flow'])
        T_units = units['T']
        P_units = units['P']
        flow_units = units['flow']
        T_notation = notation['T']
        P_notation = notation['P']
        flow_notation = notation['flow']
        T = UofM.convert(self.T, 'K', T_units)
        P = UofM.convert(self.P, 'Pa', P_units)
        data = []
        index = []
        index.append((f"Temperature [{T_units}]", ''))
        data.append(f"{T:{T_notation}}")
        index.append((f"Pressure [{P_units}]", ''))
        data.append(f"{P:{P_notation}}")
        for phase in self.phases:
            if indexer.data.ndim == 2:
                flow_array = factor * indexer[phase, all_IDs]
            else:
                flow_array = factor * indexer[all_IDs]
            phase = valid_phases[phase]
            if phase.islower():
                phase = phase.capitalize()
            if composition:
                total_flow = flow_array.sum()
                index.append((f"{phase} [{flow_units}]", ''))
                data.append(f"{total_flow:{flow_notation}}")
                if total_flow == 0:
                    comp_array = flow_array
                else:
                    comp_array = 100 * flow_array / total_flow
                for i, (ID, comp) in enumerate(zip(all_IDs, comp_array)):
                    if not comp:
                        continue
                    if i >= N_max:
                        index.append(("Composition [%]", '(remainder)'))
                        data.append(
                            f"{comp_array[N_max:].sum():{flow_notation}}")
                        break
                    else:
                        index.append(("Composition [%]", ID))
                        data.append(f"{comp:{flow_notation}}")
            else:
                for i, (ID, flow) in enumerate(zip(all_IDs, flow_array)):
                    if not flow:
                        continue
                    if i >= N_max:
                        index.append(
                            (f"{phase} [{flow_units}]", '(remainder)'))
                        data.append(
                            f"{flow_array[N_max:].sum():{flow_notation}}")
                        break
                    else:
                        index.append((f"{phase} [{flow_units}]", ID))
                        data.append(f"{flow:{flow_notation}}")
        return pd.DataFrame(data, columns=[self.ID.replace('_', ' ')],
                            index=pd.MultiIndex.from_tuples(index))

    def _info(self, layout, T, P, flow, composition, N, IDs, sort=None, df=False):
        """Return string with all specifications."""
        units, notation = self.get_display_units_and_notation(
            T=T, P=P, flow=flow)
        units['flow'], composition, N, sort = self._translate_layout(
            layout, units['flow'], composition, N, sort)
        display_units = self.display_units
        N_max = display_units.N if N is None else N
        composition = display_units.composition if composition is None else composition
        sort = display_units.sort if sort is None else sort
        name, factor = self._get_flow_name_and_factor(units['flow'])
        indexer = getattr(self, 'i' + name)
        if not IDs:
            IDs = self.chemicals.IDs
            data = getattr(self, name)
        else:
            data = indexer[IDs]
        IDs, data = nonzeros(IDs, data)
        if sort:
            index = sorted(range(len(data)),
                           key=lambda x: data[x], reverse=True)
            IDs = [IDs[i] for i in index]
        IDs = tuple(IDs)
        return (self._info_df if df else self._info_str)(
            units, notation, composition, N_max, IDs, indexer, factor,
        )

    def _get_tooltip_string(self, format, full):
        if format not in ('html', 'svg'):
            return ''
        if self.isempty():
            tooltip = '(empty)'
        elif format == 'html' and full:
            df = self._info(None, None, None, None, None,
                            None, None, None, df=True)
            tooltip = (
                " " +  # makes sure graphviz does not try to parse the string as HTML
                # unset makes sure that table header style can be overwritten in CSS
                df.to_html(justify='unset').
                # makes sure tippy.js does not add any whitespaces
                replace("\n", "").replace("  ", "")
            )
        else:
            newline = '<br>' if format == 'html' else '\n'
            display_units = self.display_units
            T_units = display_units.T
            P_units = display_units.P
            flow_units = display_units.flow
            T = UofM.convert(self.T, 'K', T_units)
            P = UofM.convert(self.P, 'Pa', P_units)
            display_notation = self.display_notation
            T_notation = display_notation.T
            P_notation = display_notation.P
            flow_notation = display_notation.flow
            tooltip = (
                f"Temperature: {T:{T_notation}} {T_units}{newline}"
                f"Pressure: {P:{P_notation}} {P_units}"
            )
            for phase in self.phases:
                stream = self[phase] if self.imol.data.ndim == 2 else self
                flow = stream.get_total_flow(flow_units)
                phase = valid_phases[phase]
                if phase.islower():
                    phase = phase.capitalize()
                tooltip += f"{newline}{phase} flow: {flow:{flow_notation}} {flow_units}"
            if format == 'html':
                tooltip = " " + tooltip
        return tooltip

    def show(self,
             layout: Optional[str] = None,
             T: Optional[str] = None,
             P: Optional[str] = None,
             flow: Optional[str] = None,
             composition: Optional[bool] = None,
             N: Optional[int] = None,
             IDs: Optional[Sequence[str]] = None,
             sort: Optional[bool] = None,
             df: Optional[bool] = None):
        """
        Print all specifications.

        Parameters
        ----------
        layout : 
            Convenience paramater for passing `flow`, `composition`, and `N`. 
            Must have the form {'%' or ''}{'wt', 'mol' or 'vol'}{# or ''}.
            For example: '%wt100' corresponds to compostion=True, flow='kg/hr', 
            and N=100.
        T : 
            Temperature units.
        P : 
            Pressure units.
        flow : 
            Flow rate units.
        composition : 
            Whether to show composition.
        N : 
            Number of compounds to display.
        IDs : 
            IDs of compounds to display. Defaults to all chemicals.
        sort :
            Whether to sort flows in descending order.
        df :
            Whether to return a pandas DataFrame.

        Examples
        --------
        Show a stream's composition by weight for only the top 2 chemicals
        with the highest mass fractions:

        >>> import biosteam as bst
        >>> bst.settings.set_thermo(['Water', 'Ethanol', 'Methanol', 'Propanol'])
        >>> stream = bst.Stream('stream', Water=0.5, Ethanol=1.5, Methanol=0.2, Propanol=0.3, units='kg/hr')
        >>> stream.show('cwt2s') # Alternatively: stream.show(composition=True, flow='kg/hr', N=2, sort=True)
        Stream: stream
        phase: 'l', T: 298.15 K, P: 101325 Pa
        composition (%): Ethanol  60
                         Water    20
                         ...      20
                         -------  2.5 kg/hr

        """
        print(self._info(layout, T, P, flow, composition, N, IDs, sort, df))
    _ipython_display_ = show

    def print(self, units: Optional[str] = None):
        """
        Print in a format that you can use recreate the stream.

        Parameters
        ----------
        units : 
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

    # Convinience math methods for scripting

    def __add__(self, other):
        return Stream.sum([self, other])

    def __radd__(self, other):
        return Stream.sum([self, other])

    def __iadd__(self, other):
        self.mix_from([self, other])
        return self

    def __isub__(self, other):
        self.separate_out(other)
        return self

    def __neg__(self):
        new = self.copy()
        new._imol.data *= -1
        return new

    def __mul__(self, other):
        new = self.copy()
        new._imol.data *= other
        return new

    def __rmul__(self, other):
        new = self.copy()
        new._imol.data *= other
        return new

    def __truediv__(self, other):
        new = self.copy()
        new._imol.data /= other
        return new

    def __imul__(self, other):
        self._imol.data *= other
        return self

    def __itruediv__(self, other):
        self._imol.data /= other
        return self
