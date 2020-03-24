# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:54:15 2019

@author: yoelr
"""
from ._stream import Stream
from ._thermal_condition import ThermalCondition
from .indexer import MolarFlowIndexer
from ._settings import settings
from .equilibrium import VLE
from .utils import Cache, assert_same_chemicals
import numpy as np

__all__ = ('MultiStream', )

class MultiStream(Stream):
    """
    Create a MultiStream object that defines material flow rates for multiple
    phases along with its thermodynamic state. Thermodynamic and transport
    properties of a stream are available as properties, while thermodynamic
    equilbrium (e.g. VLE, and bubble and dew points) are available as methods. 

    Parameters
    ----------
    ID='' : str
        A unique identification. If ID is None, stream will not be registered.
        If no ID is given, stream will be registered with a unique ID.
    flow=() : 2d array
        All flow rates corresponding to `phases` by row and chemical IDs by column.
    thermo=None : Thermo
        Thermodynamic equilibrium package. Defaults to `thermosteam.settings.get_thermo()`.
    units='kmol/hr' : str
        Flow rate units of measure (only mass, molar, and
        volumetric flow rates are valid).
    phases : tuple['g', 'l', 's', 'G', 'L', 'S']
        Tuple denoting the phases present. Defaults to ('g', 'l').
    T=298.15 : float
        Temperature [K].
    P=101325 : float
        Pressure [Pa].
    price=0 : float
        Price per unit mass [USD/kg].
    **phase_flow : tuple[str, float]
        phase-(ID, flow) pairs.
    
    Examples
    --------
    Before creating streams, first set the chemicals:
        
    >>> import thermosteam as tmo
    >>> tmo.settings.set_thermo(['Water', 'Ethanol'])
    
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
    (298.15, 101325)
    
    Unlike Stream objects, the `mol` attribute does not store data, it simply
    returns the total flow rate of each chemical. Setting an element of the
    array raises an error to prevent the wrong assumption that the data is
    linked:
    
    >>> s1.mol
    array([1.11 , 0.217])
    >>> # s1.mol[0] = 1
    >>> # -> ValueError: assignment destination is read-only
    
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
    >>> # -> IndexError: multiple phases present; must include phase key to
    set chemical data
    
    Chemical flows must be set by phase:

    >>> s1.imol['l', ('Ethanol', 'Water')] = [1, 0]
    
    The most convinient way to get and set flow rates is through the 
    `get_flow` and `set_flow` methods:
    
    >>> # Set flow
    >>> key = ('l', 'Water')
    >>> s1.set_flow(1, 'gpm', key)
    >>> s1.get_flow('gpm', key)
    0.9999999999999999
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
        
    >>> # s1.set_flow([10, 20], 'kg/hr', ('Ethanol', 'Water'))
    >>> # -> IndexError: multiple phases present; must include phase key to set chemical data
    
    Note that for both Stream and MultiStream objects, `mol`, `imol`, 
    and `get_flow` return chemical flows across all phases when given only 
    chemical IDs.
    
    Vapor-liquid equilibrium can be performed by setting 2 degrees of freedom
    from the following list:
    * T [Temperature; in K]
    * P [Pressure; in K]
    * V [Vapor fraction]
    * H [Enthalpy; in kJ/hr]:
        
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
    
    >>> # s1['g'].phase = 'l'
    >>> # -> AttributeError: phase is locked
    
    """
    __slots__ = ()
    def __init__(self, ID="", flow=(), T=298.15, P=101325.,
                 phases=('g', 'l'), units=None,
                 thermo=None, price=0, **phase_flows):
        self._TP = ThermalCondition(T, P)
        thermo = self._load_thermo(thermo)
        self._init_indexer(flow, phases, thermo.chemicals, phase_flows)
        self.price = price
        if units:
            name, factor = self._get_flow_name_and_factor(units)
            flow = getattr(self, 'i' + name)
            flow.data[:] = self._imol._data * factor
        self._sink = self._source = None
        self._init_cache()
        self._register(ID)
        
    def _init_indexer(self, flow, phases, chemicals, phase_flows):
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
        
    def _init_cache(self):
        super()._init_cache()
        self._streams = {}
        self._vle_cache = Cache(VLE, self._imol, self._TP, thermo=self._thermo,
                                bubble_point_cache=self._bubble_point_cache,
                                dew_point_cache=self._dew_point_cache)
        
    def __getitem__(self, phase):
        streams = self._streams
        if phase in streams:
            stream = streams[phase]
        else:
            stream = Stream.__new__(Stream)
            stream._sink = stream._source = None
            stream._imol = self._imol.get_phase(phase)
            stream._ID = None
            stream._TP = self._TP
            stream._thermo = self._thermo
            streams[phase] = stream
        return stream
    
    def __iter__(self):
        for i in self._imol._phases: yield self[i]
    
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
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'])
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
        >>> tmo.settings.set_thermo(['Water', 'Ethanol'])
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
        phases = sorted(phases)
        if phases != self.phases:
            self._imol = self._imol.to_material_array(phases)
            self._vle_cache = Cache(VLE, self._imol, self._TP, thermo=self._thermo)
    
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
        return self.mixture.xH_at_TP(self._imol.iter_data(), self._TP)
    @H.setter
    def H(self, H):
        self.T = self.mixture.xsolve_T(self._imol.iter_data(), H, self.T, self.P)

    @property
    def S(self):
        """[float] Entropy flow rate in kJ/hr."""
        return self.mixture.xS_at_TP(self._imol.iter_data(), self._TP)
    @property
    def C(self):
        """[float] Heat capacity flow rate in kJ/hr."""
        return self.mixture.xCn_at_TP(self._imol.iter_data(), self._TP)
    @property
    def F_vol(self):
        """[float] Total volumetric flow rate in m3/hr."""
        return 1000. * self.mixture.xV_at_TP(self._imol.iter_data(), self._TP)
    @F_vol.setter
    def F_vol(self, value):
        F_vol = self.F_vol
        if not F_vol: raise AttributeError("undefined composition; cannot set flow rate")
        self.ivol._data[:] *= value / F_vol / 1000.
    
    @property
    def Hvap(self):
        """[float] Enthalpy of vaporization flow rate in kJ/hr."""
        return self.mixture.Hvap_at_TP(self._imol['l'], self._TP)
    
    ### Composition properties ###
    
    @property
    def V(self):
        """[float] Molar volume [m^3/mol]."""
        return self.mixture.xV_at_TP(self._imol.iter_composition(), self._TP)
    @property
    def kappa(self):
        """[float] Thermal conductivity [W/m/k]."""
        return self.mixture.xkappa_at_TP(self._imol.iter_composition(), self._TP)        
    @property
    def Cn(self):
        """[float] Molar heat capacity [J/mol/K]."""
        return self.mixture.xCn_at_TP(self._imol.iter_composition(), self._TP)
    @property
    def mu(self):
        """[float] Hydrolic viscosity [Pa*s]."""
        return self.mixture.xmu_at_TP(self._imol.iter_composition(), self._TP)

    @property
    def sigma(self):
        """[float] Surface tension [N/m]."""
        mol = self._imol['l']
        return self.mixture.sigma_at_TP(mol / mol.sum(), self._TP)
    @property
    def epsilon(self):
        """[float] Relative permittivity [-]."""
        mol = self._imol['l']
        return self.mixture.epsilon_at_TP(mol / mol.sum(), self._TP)
        
    ### Methods ###
        
    def copy_flow(self, stream, phase=..., IDs=..., *, remove=False, exclude=False):
        """Copy flow rates of stream to self.
        
        Parameters
        ----------
        stream : Stream
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
        
        """
        chemicals = self.chemicals
        if exclude:
            not_index = chemicals.get_index(IDs)
            index = np.ones(chemicals.size, dtype=bool)
            index[not_index] = False
        else:
            index = chemicals.get_index(IDs)
        if isinstance(stream, MultiStream):
            stream_phase_mol = stream.imol[phase]
            self_phase_mol = self.imol[phase]
            self_phase_mol[index] = stream_phase_mol[index]
            if remove: 
                stream_phase_mol[index] = 0
        else:
            if phase is Ellipsis:
                phase = stream.phase
                self_imol = self.imol
                for i in self.phases:
                    if i != phase: self_imol[i] = 0.
            if stream.phase == phase:
                stream_mol = stream.mol
                self_phase_mol = self.imol[phase]
                self_phase_mol[index] = stream_mol[index]
                if remove: 
                    stream_mol[index] = 0
    
    def get_normalized_mol(self, IDs):
        z = self.imol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_vol(self, IDs):
        z = self.ivol[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_normalized_mass(self, IDs):
        z = self.imass[..., IDs].sum(0)
        z /= z.sum()
        return z
    
    def get_molar_composition(self, IDs):
        return self.imol[..., IDs].sum(0)/self.F_mol
    
    def get_mass_composition(self, IDs):
        return self.imass[..., IDs].sum(0)/self.F_mass
    
    def get_volumetric_composition(self, IDs):
        return self.ivol[..., IDs].sum(0)/self.F_vol
    
    def get_concentration(self, phase, IDs):
        return self.imol[phase, IDs]/self.F_vol
    
    def mix_from(self, others):
        if settings._debug: assert_same_chemicals(self, others)
        multi = []; single = []; isa = isinstance
        for i in others:
            if i: (multi if isa(i, MultiStream) else single).append(i)
        self.empty()
        for i in single:
            self._imol[i.phase] += i.mol    
        self._imol._data[:] += sum([i._imol._data for i in multi])
        T = others[0].T
        if all([T==i.T for i in others[1:]]):
            self.T = T
        else:
            self.H = sum([i.H for i in others])
        
    def link_with(self, other):
        if settings._debug:
            assert isinstance(other, self.__class__), "other must be of same type to link with"
        self._TP = other._TP
        self._imol._data = other._imol._data
        self._streams = other._streams
        self._vle_cache = other._vle_cache
        self._dew_point_cache = other._dew_point_cache
        self._bubble_point_cache = other._bubble_point_cache
        self._imol._data_cache = other._imol._data_cache
    
    ### Equilibrium ###
    
    @property
    def vle(self):
        """[VLE] An object that can perform vapor-liquid equilibrium on the stream."""
        return self._vle_cache.retrieve()
    
    ### Casting ###
    
    @property
    def phase(self):
        return "".join(self._imol._phases)
    @phase.setter
    def phase(self, phase):
        assert len(phase) == 1, f'invalid phase {repr(phase)}'
        self.__class__ = Stream
        self._imol = self._imol.to_chemical_indexer(phase)
        
    ### Representation ###
    
    def _info(self, T, P, flow, composition, N):
        """Return string with all specifications."""
        from .indexer import nonzeros
        IDs = self.chemicals.IDs
        basic_info = self._basic_info()
        all_IDs, _ = nonzeros(self.chemicals.IDs, self.mol)
        all_IDs = tuple(all_IDs)
        display_units = self.display_units
        T_units = T or display_units.T
        P_units = P or display_units.P
        flow_units = flow or display_units.flow
        N_max = N or display_units.N
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
                flow_rates += f'{IDs[i]} ' + spaces + f' {data[i]:.4g}'
            if too_many_chemicals:
                flow_rates += new_line + '...'
            if composition:
                dashes = '-' * maxlen
                flow_rates += f"{new_line}{dashes}  {total_flow:.3g} {flow_units}"
            # Put it together
            phases_flow_rates_info += beginning + flow_rates + '\n'
            
        return basic_info + phases_flow_rates_info.rstrip('\n')
    
    def print(self):
        from .utils import repr_kwarg, repr_couples
        IDs = self.chemicals.IDs
        phase_data = []
        for phase, data in self.imol.iter_data():
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
    