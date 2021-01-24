# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import thermosteam as tmo
from warnings import warn
from flexsolve import IQ_interpolation
from chemicals.identifiers import pubchem_db
from chemicals.vapor_pressure import vapor_pressure_handle
from chemicals.phase_change import (Tb as normal_boiling_point_temperature,
                                    Tm as normal_melting_point_temperature,
                                    Hfus as heat_of_fusion,
                                    heat_of_vaporization_handle)
from chemicals.critical import (Ihmels,
                                Tc as critical_point_temperature,
                                Pc as critical_point_pressure,
                                Vc as critical_point_volume)
from chemicals.acentric import (omega as acentric_factor,
                                omega_definition as acentric_factor_definition,
                                LK_omega as acentric_factor_LK,
                                Stiel_polar_factor as compute_Stiel_Polar)
from chemicals.triple import (Tt as triple_point_temperature,
                              Pt as triple_point_pressure)
from chemicals.combustion import combustion_data, combustion_stoichiometry
from chemicals.volume import volume_handle
from chemicals.heat_capacity import heat_capacity_handle
from chemicals.reaction import (
    Hf as heat_of_formation,
    S0 as absolute_entropy_of_formation
)
from chemicals.elements import (
    similarity_variable as compute_similarity_variable, 
    molecular_weight as compute_molecular_weight,
    get_atoms,
)
from chemicals.viscosity import viscosity_handle
from chemicals.thermal_conductivity import thermal_conductivity_handle
from chemicals.permittivity import permitivity_handle
from chemicals.interface import surface_tension_handle
from chemicals.dipole import dipole_moment
from .free_energy import (
    Enthalpy, Entropy,
    EnthalpyRefSolid, EnthalpyRefLiquid, EnthalpyRefGas,
    EntropyRefSolid, EntropyRefLiquid, EntropyRefGas,
    ExcessEnthalpyRefSolid, ExcessEnthalpyRefLiquid, ExcessEnthalpyRefGas,
    ExcessEntropyRefSolid, ExcessEntropyRefLiquid, ExcessEntropyRefGas
)
from .equilibrium.unifac import (
    DDBST_UNIFAC_assignments, 
    DDBST_MODIFIED_UNIFAC_assignments,
    DDBST_PSRK_assignments,
    UNIFACGroupCounts, 
    DortmundGroupCounts, 
    PSRKGroupCounts,
    NISTGroupCounts,
)
from .base import (PhaseHandle, PhaseTHandle, PhaseTPHandle,
                   ThermoModelHandle, TDependentModelHandle,
                   TPDependentModelHandle, display_asfunctor)
from .units_of_measure import chemical_units_of_measure
from .eos import GCEOS_DUMMY, PR
from .utils import copy_maybe, check_valid_ID
from . import functional as fn 
from ._phase import check_phase
from . import units_of_measure as thermo_units

# from .solubility import SolubilityParameter
# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

__all__ = ('Chemical',)

# %% Filling missing properties

def get_chemical_data(chemical):
    getfield = getattr
    return {i:getfield(chemical, i, None) for i in chemical.__slots__}

def unpickle_chemical(chemical_data):
    chemical = object.__new__(Chemical)
    setfield = setattr
    for field, value in chemical_data.items():
        setfield(chemical, field, value)
    return chemical


# %% Representation
    
def chemical_identity(chemical, pretty=False):
    typeheader = f"{type(chemical).__name__}:"
    full_ID = f"{typeheader} {chemical.ID} (phase_ref={repr(chemical.phase_ref)})"
    phase = chemical.locked_state
    state = ' at ' + f"phase={repr(phase)}" if phase else ""
    return full_ID + state


# %% Initialize EOS
                         
def create_eos(eos, Tc, Pc, omega):
    try: return eos(T=298.15, P=101325., Tc=Tc, Pc=Pc, omega=omega)
    except: return GCEOS_DUMMY(T=298.15, P=101325.)

# %% Resetting data

def reset_constant(chemical, var, value):
    getfield = getattr
    setattr(chemical, '_'+var, value)
    for handle in _model_and_phase_handles: 
        getfield(chemical, handle).set_value(var, value)

def reset_energy_constant(chemical, var, value):
    getfield = getattr
    setattr(chemical, '_'+var, value)
    for handle in _energy_handles: 
        getfield(chemical, handle).set_value(var, value)

def raise_helpful_handle_error(handle):
    var = handle.var
    if isinstance(handle, PhaseHandle):
        raise AttributeError(
            f"cannot set '{var}'; use the `add_model` "
            f"and the `set_model_priority` methods "
            f"to modify the thermodynamic properties for "
            f"each phase (e.g. {var}.l.add_model(...))")
    elif isinstance(handle, ThermoModelHandle):
        raise AttributeError(
            f"cannot set '{var}'; use the `{var}.add_model` "
            f"and the `{var}.set_model_priority` methods to "
            f"modify the thermodynamic property instead")
    else:
        raise Exception(
            'expected a either a PhaseHandle or a ThermoModelHandle object; '
           f'got a {type(handle).__name__} object instead')

def raise_helpful_energy_functor_error(energy_functor):
    var = energy_functor.var
    raise AttributeError(f"cannot set '{var}'; use the `<Chemical>.reset_free_energies` "
                     "method to reset the free energy models with updated "
                     "chemical data")

# %%  Utilities

def as_chemical(chemical):
    return chemical if isinstance(chemical, Chemical) else Chemical(chemical)

# %% Chemical fields

_names = ('_CAS',
          '_InChI', 
          '_InChI_key',
          '_common_name',
          '_iupac_name',
          '_pubchemid', 
          '_smiles', 
          '_formula')

_groups = ('_Dortmund',
           '_UNIFAC', 
           '_PSRK',
           '_NIST')

_model_handles = ('_Psat', '_Hvap', '_sigma', '_epsilon')
    
_phase_handles = ('_kappa', '_V', '_Cn', '_mu')

_energy_handles = ('_S_excess', '_H_excess', '_S', '_H')

_model_and_phase_handles = _model_handles + _phase_handles

_model_and_phase_properties = ('Psat', 'Hvap', 'sigma', 'epsilon',
                               'kappa', 'V', 'Cn', 'mu')

_handles = _model_and_phase_handles + _energy_handles

_data = ('_MW', '_Tm', '_Tb', '_Tt', '_Tc', '_Pt', '_Pc', '_Vc',
         '_Hf', '_S0', '_LHV', '_HHV', '_Hfus', '_Sfus', '_omega', '_dipole',
         '_similarity_variable', '_iscyclic_aliphatic', '_combustion')

_functor_data = {'MW', 'Tm', 'Tb', 'Tt', 'Pt', 'Tc', 'Pc', 'Vc',
                 'Hfus', 'Sfus', 'omega', 'dipole',
                 'similarity_variable', 'iscyclic_aliphatic'}

_checked_properties = ('phase_ref', 'eos', 'eos_1atm',
                  'S_excess', 'H_excess', 'mu', 'kappa', 'V', 'S', 
                  'H', 'Cn', 'Psat', 'Hvap', 'sigma', 'epsilon', 
                  'Dortmund', 'UNIFAC', 'PSRK', 'Hf', 'LHV', 'HHV',
                  'combustion', *_functor_data)

_chemical_fields = {'\n[Names]  ': _names,
                    '\n[Groups] ': _groups,
                    '\n[Data]   ': _data}


# %% Chemical

class Chemical:
    """
    Creates a Chemical object which contains constant chemical properties,
    as well as thermodynamic and transport properties as a function of 
    temperature and pressure.
    
    Parameters
    ----------
    ID : str
        One of the following [-]:
            * Name, in IUPAC form or common form or a synonym registered in PubChem
            * InChI name, prefixed by 'InChI=1S/' or 'InChI=1/'
            * InChI key, prefixed by 'InChIKey='
            * PubChem CID, prefixed by 'PubChem='
            * SMILES (prefix with 'SMILES=' to ensure smiles parsing)
            * CAS number
    cache : optional
        Whether or not to use cached chemicals and cache new chemicals.
    
    Other Parameters
    ----------------
    search_ID : str, optional
        ID to search through database. Pass this key-word argument
        when you'd like to give a custom name to the chemical, but
        cannot find the chemical with that name.
    eos : GCEOS subclass, optional
        Equation of state class for solving thermodynamic properties.
        Defaults to Peng-Robinson.
    phase_ref : {'s', 'l', 'g'}, optional
        Reference phase of chemical.
    CAS : str, optional
        CAS number of chemical.
    phase: {'s', 'l' or 'g'}, optional
        Phase to set state of chemical.
    search_db=True: bool, optional
        Whether to search the data base for chemical.
    Cn : float or function(T), optional
        Molar heat capacity model [J/mol] as a function of temperature [K].
    sigma : float or function(T), optional
        Surface tension model [N/m] as a function of temperature [K].
    epsilon : float or function(T), optional
        Relative permitivity model [-] as a function of temperature [K].
    Psat : float or function(T), optional
        Vapor pressure model [N/m] as a function of temperature [K].
    Hvap : float or function(T), optional
        Heat of vaporization model [J/mol] as a function of temperature [K].
    V : float or function(T, P), optional
        Molar volume model [m3/mol] as a function of temperature [K] and pressure [Pa].
    mu : float or function(T, P), optional
        Dynamic viscosity model [Pa*s] as a function of temperature [K] and pressure [Pa].
    kappa : float or function(T, P), optional
        Thermal conductivity model [W/m/K] as a function of temperature [K] and pressure [Pa].
    Cp : float, optional
        Constant heat capacity model [J/g].
    rho : float, optional
        Constant density model [kg/m3].
    default=False : bool, optional
        Whether to default any missing chemical properties such as molar volume,
        heat capacity, surface tension, thermal conductivity, and molecular weight
        to that of water (on a weight basis).
    **data : float or str
        User data (e.g. Tb, formula, etc.).
    
    Examples
    --------
    Chemical objects contain pure component properties:
    
    >>> import thermosteam as tmo
    >>> # Initialize with an identifier
    >>> # (e.g. by name, CAS, InChI...)
    >>> Water = tmo.Chemical('Water') 
    >>> Water.show()
    Chemical: Water (phase_ref='l')
    [Names]  CAS: 7732-18-5
             InChI: H2O/h1H2
             InChI_key: XLYOFNOQVPJJNP-U...
             common_name: water
             iupac_name: ('oxidane',)
             pubchemid: 962
             smiles: O
             formula: H2O
    [Groups] Dortmund: <1H2O>
             UNIFAC: <1H2O>
             PSRK: <1H2O>
             NIST: <Empty>
    [Data]   MW: 18.015 g/mol
             Tm: 273.15 K
             Tb: 373.12 K
             Tt: 273.15 K
             Tc: 647.14 K
             Pt: 610.88 Pa
             Pc: 2.2048e+07 Pa
             Vc: 5.6e-05 m^3/mol
             Hf: -2.8582e+05 J/mol
             S0: 70 J/K/mol
             LHV: 44011 J/mol
             HHV: 0 J/mol
             Hfus: 6010 J/mol
             Sfus: None
             omega: 0.344
             dipole: 1.85 Debye
             similarity_variable: 0.16653
             iscyclic_aliphatic: 0
             combustion: {'H2O': 1.0}

    All fields shown are accessible:
    
    >>> Water.CAS
    '7732-18-5'

    Functional group identifiers (e.g. `Dortmund`, `UNIFAC`, `PSRK`) allow for the estimation of activity coefficients through group contribution methods. In other words, these attributes define the functional groups for thermodynamic equilibrium calculations:
        
    >>> Water.Dortmund
    <DortmundGroupCounts: 1H2O>
    
    Temperature (in Kelvin) and pressure (in Pascal) dependent properties can be computed:
        
    >>> # Vapor pressure (Pa)
    >>> Water.Psat(T=373.15)
    101284.55
    >>> # Surface tension (N/m)
    >>> Water.sigma(T=298.15)
    0.0719722
    >>> # Molar volume (m^3/mol)
    >>> Water.V(phase='l', T=298.15, P=101325)
    1.80692...e-05
    
    Note that the reference state of all chemicals is 25 degC and 1 atm:
    
    >>> (Water.T_ref, Water.P_ref)
    (298.15, 101325.0)
    >>> # Enthalpy at reference conditions (J/mol; without excess energies)
    >>> Water.H(T=298.15, phase='l')
    0.0
    
    Constant pure component properties are also available:
    
    >>> # Molecular weight (g/mol)
    >>> Water.MW
    18.01528
    >>> # Boiling point (K)
    >>> Water.Tb
    373.124
    
    Temperature dependent properties are managed by model handles:
    
    >>> Water.Psat.show()
    TDependentModelHandle(T, P=None) -> Psat [Pa]
    [0] Wagner original
    [1] Antoine
    [2] EQ101
    [3] Wagner
    [4] boiling critical relation
    [5] Lee Kesler
    [6] Ambrose Walton
    [7] Sanjari
    [8] Edalat

    Phase dependent properties have attributes with model handles for each phase:

    >>> Water.V
    <PhaseTPHandle(phase, T, P) -> V [m^3/mol]>
    >>> (Water.V.l, Water.V.g)
    (<TPDependentModelHandle(T, P) -> V.l [m^3/mol]>, <TPDependentModelHandle(T, P) -> V.g [m^3/mol]>)

    A model handle contains a series of models applicable to a certain domain:
    
    >>> Water.Psat[0].show()
    TDependentModel(T, P=None) -> Psat [Pa]
     name: Wagner original
     Tmin: 275 K
     Tmax: 647.35 K

    When called, the model handle searches through each model until it finds one with an applicable domain. If none are applicable, a value error is raised:
        
    >>> Water.Psat(373.15)
    101284.55
    >>> # Water.Psat(1000.0) ->
    >>> # ValueError: Water has no valid saturated vapor pressure model at T=1000.00 K
    
    Model handles as well as the models themselves have tabulation and plotting methods to help visualize how properties depend on temperature and pressure.
    
    >>> # import matplotlib.pyplot as plt
    >>> # Water.Psat.plot_vs_T([Water.Tm, Water.Tb], 'degC', 'atm', label="Water")
    >>> # plt.show()
    
    .. figure:: ./images/Water_Psat_vs_T.png
    
    >>> # Plot all models
    
    >>> # Water.Psat.plot_models_vs_T([Water.Tm, Water.Tb], 'degC', 'atm')
    >>> # plt.show()
    
    .. figure:: ./images/Water_all_models_Psat_vs_T.png
    
    Each model may contain either a function or a functor (a function with stored data) to compute the property:
        
    >>> functor = Water.Psat[0].evaluate
    >>> functor.show()
    Functor: Wagner_original(T, P=None) -> Psat [Pa]
     Tc: 647.35 K
     Pc: 2.2122e+07 Pa
     a: -7.7645
     b: 1.4584
     c: -2.7758
     d: -1.233
    
    .. Note::
       All chemical property functors are available in the thermosteam.properties subpackage. You can also use help(<functor>) for further information on the math and equations used in the functor.
    
    A new model can be added easily to a model handle through the `add_model` method, for example:
        
    >>> # Set top_priority=True to place model in postion [0]
    >>> @Water.Psat.add_model(Tmin=273.20, Tmax=473.20, top_priority=True)
    ... def User_antoine_model(T):
    ...     return 10.0**(10.116 -  1687.537 / (T - 42.98))
    >>> Water.Psat.show()
    TDependentModelHandle(T, P=None) -> Psat [Pa]
    [0] User antoine model
    [1] Wagner original
    [2] Antoine
    [3] EQ101
    [4] Wagner
    [5] boiling critical relation
    [6] Lee Kesler
    [7] Ambrose Walton
    [8] Sanjari
    [9] Edalat

    The `add_model` method is a high level interface that even lets you create a constant model:
        
    >>> value = Water.V.l.add_model(1.687e-05, name='User constant')
    ... # Model is appended at the end by default
    >>> added_model = Water.V.l[-1]
    >>> added_model.show()
    ConstantThermoModel(T=None, P=None) -> V.l [m^3/mol]
     name: User constant
     value: 1.687e-05
     Tmin: 0 K
     Tmax: inf K
     Pmin: 0 Pa
     Pmax: inf Pa

    .. Note::
       Because no bounds were given, the model assumes it is valid across all temperatures and pressures.

    Manage the model order with the `set_model_priority` and `move_up_model_priority` methods:
    
    >>> # Note: In this case, we pass the model name, but its
    >>> # also possible to pass the current index, or the model itself.
    >>> Water.Psat.move_up_model_priority('Wagner original')
    >>> Water.Psat.show()
    TDependentModelHandle(T, P=None) -> Psat [Pa]
    [0] Wagner original
    [1] Antoine
    [2] EQ101
    [3] Wagner
    [4] boiling critical relation
    [5] Lee Kesler
    [6] Ambrose Walton
    [7] Sanjari
    [8] Edalat
    [9] User antoine model
    
    >>> Water.Psat.set_model_priority('Antoine')
    >>> Water.Psat.show()
    TDependentModelHandle(T, P=None) -> Psat [Pa]
    [0] Antoine
    [1] Wagner original
    [2] EQ101
    [3] Wagner
    [4] boiling critical relation
    [5] Lee Kesler
    [6] Ambrose Walton
    [7] Sanjari
    [8] Edalat
    [9] User antoine model
    
    The default priority is `0` (or top priority), but you can choose any priority:
    
    >>> Water.Psat.set_model_priority('Antoine', 1)
    >>> Water.Psat.show()
    TDependentModelHandle(T, P=None) -> Psat [Pa]
    [0] Wagner original
    [1] Antoine
    [2] EQ101
    [3] Wagner
    [4] boiling critical relation
    [5] Lee Kesler
    [6] Ambrose Walton
    [7] Sanjari
    [8] Edalat
    [9] User antoine model
    
    .. note::
    
        All models are stored as a :py:class:`deque` in the `models` attribute (e.g. Water.Psat.models).
    
    Attributes
    ----------
    mu(phase, T, P) : 
        Dynamic viscosity [Pa*s].
    kappa(phase, T, P): 
        Thermal conductivity [W/m/K].
    V(phase, T, P): 
        Molar volume [m^3/mol].
    Cn(phase, T) : 
        Molar heat capacity [J/mol/K].
    Psat(T) : 
        Vapor pressure [Pa].
    Hvap(T) : 
        Heat of vaporization [J/mol]
    sigma(T) : 
        Surface tension [N/m].
    epsilon(T) : 
        Relative permitivity [-]
    S(phase, T, P) : 
        Entropy [J/mol].
    H(phase, T) : 
        Enthalpy [J/mol].
    S_excess(T, P) : 
        Excess entropy [J/mol].
    H_excess(T, P) : 
        Excess enthalpy [J/mol].
    
    """
    __slots__ = ('_ID', '_locked_state', 
                 '_phase_ref', '_eos', '_eos_1atm',
                 *_names, *_groups, 
                 *_handles, *_data,
                 '_N_solutes')
    
    #: [float] Reference temperature in Kelvin
    T_ref = 298.15
    #: [float] Reference pressure in Pascal
    P_ref = 101325.
    #: [float] Reference enthalpy in J/mol
    H_ref = 0.
    #: dict[str, Chemical] Cached chemicals
    chemical_cache = {}
    #: [bool] Wheather or not to search cache by default
    cache = False
    
    ### Creators ###
    
    def __new__(cls, ID, cache=None, *, search_ID=None,
                eos=None, phase_ref=None, CAS=None,
                default=False, phase=None, search_db=True, 
                V=None, Cn=None, mu=None, Cp=None, rho=None,
                sigma=None, kappa=None, epsilon=None, Psat=None,
                Hvap=None, **data):
        chemical_cache = cls.chemical_cache
        if (cache or cls.cache) and ID in chemical_cache:
            if any([search_ID, eos, phase_ref, CAS, default, phase, 
                    V, Cn, mu, Cp, rho, sigma, kappa, epsilon, Psat, Hvap,
                    data]):
                warn('cached chemical returned; additional parameters disregarded')
            return chemical_cache[ID]
        search_ID = search_ID or ID
        if not eos: eos = PR
        if search_db:
            metadata = pubchem_db.search(search_ID)
            data['metadata'] = metadata
            self = cls.new(ID, metadata.CASs, eos, phase_ref, phase,
                           **data)
        else:
            self = cls.blank(ID, CAS, phase_ref, phase=phase, **data)
        if phase:
            if mu: self.mu.add_model(mu, top_priority=True)
            if Cn: self.Cn.add_model(Cn, top_priority=True)
            if kappa: self.kappa.add_model(kappa, top_priority=True)
            if Cp: self.Cn.add_model(Cp * self.MW, top_priority=True)
            if rho: self.V.add_model(fn.rho_to_V(rho, self.MW), top_priority=True)
            if V: self.V.add_model(V, top_priority=True)
        else:
            multi_phase_items = (('mu', mu),
                                 ('Cn', Cn),
                                 ('kappa', kappa), 
                                 ('Cp', Cp),
                                 ('rho', rho),
                                 ('V', V))
            for i,j in multi_phase_items:
                if j: raise ValueError(f'must specify phase to set {i} model')
        if sigma: self.sigma.add_model(sigma, top_priority=True)
        if epsilon: self.epsilon.add_model(epsilon, top_priority=True)
        if Psat: self.Psat.add_model(Psat, top_priority=True)
        if Hvap: self.Hvap.add_model(Hvap, top_priority=True)
        if default: self.default()
        if cache:
            chemical_cache[ID] = self
            if len(chemical_cache) > 100:
                for i in chemical_cache:
                    del chemical_cache[i]
                    break
        return self

    @classmethod
    def new(cls, ID, CAS, eos=PR, phase_ref=None, phase=None, **data):
        """Create a new chemical from data without searching through
        the data base, and load all possible models from given data."""
        self = super().__new__(cls)
        self._ID = ID
        self.reset(CAS, eos, phase_ref, phase=phase, **data)
        return self

    @classmethod
    def blank(cls, ID, CAS=None, phase_ref=None, phase=None,
              formula=None, **data):
        """
        Return a new Chemical object without any thermodynamic models or data 
        (unless provided).

        Parameters
        ----------
        ID : str
            Chemical identifier.
        CAS : str, optional
            CAS number. If none provided, it defaults to the `ID`.
        phase_ref : str, optional
            Phase at the reference state (T=298.15, P=101325).
        phase : str, optional
            Phase to set state as a single phase chemical.
        **data : 
            Any data to fill chemical with.
        
        Examples
        --------
        >>> from thermosteam import Chemical
        >>> Substance = Chemical.blank('Substance')
        >>> Substance.show()
        Chemical: Substance (phase_ref=None)
        [Names]  CAS: Substance
                 InChI: None
                 InChI_key: None
                 common_name: None
                 iupac_name: None
                 pubchemid: None
                 smiles: None
                 formula: None
        [Groups] Dortmund: <Empty>
                 UNIFAC: <Empty>
                 PSRK: <Empty>
                 NIST: <Empty>
        [Data]   MW: None
                 Tm: None
                 Tb: None
                 Tt: None
                 Tc: None
                 Pt: None
                 Pc: None
                 Vc: None
                 Hf: None
                 S0: None
                 LHV: None
                 HHV: None
                 Hfus: None
                 Sfus: None
                 omega: None
                 dipole: None
                 similarity_variable: None
                 iscyclic_aliphatic: None
                 combustion: None
        
        """
        self = super().__new__(cls)
        self._eos = self._eos_1atm = None
        self._UNIFAC = UNIFACGroupCounts()
        self._Dortmund = DortmundGroupCounts()
        self._PSRK = PSRKGroupCounts()
        self._NIST = NISTGroupCounts()
        setfield = setattr
        for i in _names: setfield(self, i, None)
        for i in _data: setfield(self, i, None)
        for i in _energy_handles: setfield(self, i, None)
        if phase:
            phase = phase[0]
            check_phase(phase)
            for i in ('kappa', 'mu', 'V'):
                setfield(self, '_' + i, TPDependentModelHandle(i))
            self._Cn = TDependentModelHandle('Cn')
        else:
            for i in ('kappa', 'mu', 'V'):
                setfield(self, '_' + i, PhaseTPHandle(i))
            self._Cn = Cn = PhaseTHandle('Cn')
            Cn.s._chemical = Cn.l._chemical = Cn.g._chemical = self
        for i in ('sigma', 'epsilon', 'Psat', 'Hvap'):
            setfield(self, '_' + i, TDependentModelHandle(i))
        self._locked_state = phase
        check_valid_ID(ID)
        self._ID = ID
        self._phase_ref = phase_ref or phase
        self._CAS = CAS or ID
        for i,j in data.items(): setfield(self, '_' + i , j)
        self._eos = create_eos(PR, self._Tc, self._Pc, self._omega)
        self._eos_1atm = self._eos.to_TP(298.15, 101325)
        self._label_handles()
        if formula: self.formula = formula
        else: self._formula = formula
        return self

    def copy(self, ID, CAS=None, **data):
        """
        Return a copy of the chemical with a new ID.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Glucose = tmo.Chemical('Glucose')
        >>> Glucose.copy('Mannose')
        Chemical: Mannose (phase_ref='l')
        [Names]  CAS: Mannose
                 InChI: C6H12O6/c7-1-3(9)5(1...
                 InChI_key: GZCGUPFRVQAUEE-S...
                 common_name: D-Glucose
                 iupac_name: ('(2R,3S,4R,5R)...
                 pubchemid: 1.0753e+05
                 smiles: O=C[C@H](O)[C@@H](O...
                 formula: C6H12O6
        [Groups] Dortmund: <1CH2, 4CH, 1OH(P...
                 UNIFAC: <1CH2, 4CH, 5OH, 1C...
                 PSRK: <1CH2, 4CH, 5OH, 1CHO...
                 NIST: <Empty>
        [Data]   MW: 180.16 g/mol
                 Tm: None
                 Tb: 616.29 K
                 Tt: None
                 Tc: 755 K
                 Pt: None
                 Pc: 4.82e+06 Pa
                 Vc: 0.000414 m^3/mol
                 Hf: -1.2711e+06 J/mol
                 S0: 0 J/K/mol
                 LHV: -2.5406e+06 J/mol
                 HHV: -2.8047e+06 J/mol
                 Hfus: 0 J/mol
                 Sfus: None
                 omega: 2.387
                 dipole: None
                 similarity_variable: 0.13322
                 iscyclic_aliphatic: 0
                 combustion: {'CO2': 6, 'O2'...
        
        """
        new = super().__new__(self.__class__)
        getfield = getattr
        setfield = setattr
        for field in self.__slots__: 
            value = getfield(self, field, None)
            setfield(new, field, copy_maybe(value))
        new._ID = ID
        new._CAS = CAS or ID
        new._locked_state = new._locked_state
        new._init_energies(new.Cn, new.Hvap, new.Psat, new.Hfus, new.Sfus, new.Tm,
                           new.Tb, new.eos, new.eos_1atm, new.phase_ref)
        new._label_handles()
        for i,j in data.items(): setfield(new, i , j)
        return new
    __copy__ = copy

    def _label_handles(self):
        handles = (self._Psat, self._Hvap, self._sigma, self._epsilon,
                   self._V, self._Cn, self._mu, self._kappa)
        isa = isinstance
        for handle in handles:
            if isa(handle, PhaseHandle):
                handle.s._chemical = \
                handle.l._chemical = \
                handle.g._chemical = self
            else:
                handle._chemical = self

    def __reduce__(self):
        return unpickle_chemical, (get_chemical_data(self),)
    
    ### Helpful functionality ###
    def rho(self, *args, **kwargs):
        """
        Return density given thermal condition [kg/m^3].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.rho('l', 298.15, 101325)
        997.015
        
        """
        return fn.V_to_rho(self.V(*args, **kwargs), self.MW)
    
    def Cp(self, *args, **kwargs):
        """
        Return heat capacity given thermal condition [J/g/K].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.Cp('l', 298.15, 101325)
        4.180
        
        """
        return self.Cn(*args, **kwargs) / self.MW
    
    def alpha(self, *args, **kwargs):
        """
        Return thermal diffusivity  given thermal condition [m^2/s].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.alpha('l', 298.15, 101325)
        1.455...e-07
        
        """
        return fn.alpha(self.kappa(*args, **kwargs), 
                        self.rho(*args, **kwargs), 
                        self.Cp(*args, **kwargs) * 1000.)
    
    def nu(self, *args, **kwargs):
        """
        Return kinematic viscosity given thermal condition [m^2/s].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.nu('l', 298.15, 101325)
        8.958...e-07
        
        """
        return fn.mu_to_nu(self.mu(*args, **kwargs), 
                           self.rho(*args, **kwargs))
    
    def Pr(self, *args, **kwargs):
        """
        Return the Prandtl number given thermal condition [-].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.Pr('l', 298.15, 101325)
        6.156
        
        """
        return fn.Pr(self.Cp(*args, **kwargs) * 1000,
                     self.kappa(*args, **kwargs), 
                     self.mu(*args, **kwargs))
    
    def get_property(self, name, units, *args, **kwargs):
        """
        Return property in requested units.

        Parameters
        ----------
        name : str
            Name of stream property.
        units : str
            Units of measure.
        *args, **kwargs :
            Thermal condition.

        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.get_property('sigma', 'N/m', 300.) # Surface tension
        0.071685

        >>> Water.get_property('rho', 'g/cm3', 'l', 300., 101325) # Density
        0.996267

        """
        value = getattr(self, name)(*args, **kwargs)
        units_dct = thermo_units.chemical_units_of_measure
        if name in units_dct:
            original_units = units_dct[name]
        else:
            raise ValueError(f"'{name}' is not thermodynamic property")
        return original_units.convert(value, units)
    
    @property
    def phase_ref(self):
        """{'s', 'l', 'g'} Phase at 298 K and 101325 Pa."""
        return self._phase_ref
    @phase_ref.setter
    def phase_ref(self, phase):
        if phase not in ('s', 'l', 'g'):
            raise ValueError("phase must be either 's', 'l', or 'g'")
        self._phase_ref = phase
        self.reset_free_energies()

    ### Unchangeable properties ###

    @property
    def ID(self):
        """[str] Identification of chemical."""
        return self._ID
    @property
    def CAS(self):
        """[str] CAS number of chemical."""
        return self._CAS
    
    ### Names ###
    
    @property
    def InChI(self):
        """[str] IUPAC International Chemical Identifier."""
        return self._InChI
    @InChI.setter
    def InChI(self, InChI):
        self._InChI = str(InChI)
    
    @property
    def InChI_key(self):
        """[str] IUPAC International Chemical Identifier shorthand."""
        return self._InChI_key
    @InChI_key.setter
    def InChI_key(self, InChI_key):
        self._InChI_key = str(InChI_key)
    
    @property
    def common_name(self):
        """[str] Common name identifier."""
        return self._common_name
    @common_name.setter
    def common_name(self, common_name):
        self._common_name = str(common_name)
    
    @property
    def iupac_name(self):
        """[str] Standard name as defined by IUPAC."""
        return self._iupac_name
    @iupac_name.setter
    def iupac_name(self, iupac_name):
        self._iupac_name = str(iupac_name)
    
    @property
    def pubchemid(self):
        """[str] Chemical identifier as defined by PubMed."""
        return self._pubchemid
    @pubchemid.setter
    def pubchemid(self, pubchemid):
        self._pubchemid = str(pubchemid)
    
    @property
    def smiles(self):
        """[str] Chemical SMILES formula."""
        return self._smiles
    @smiles.setter
    def smiles(self, smiles):
        self._smiles = str(smiles)
    
    @property
    def formula(self):
        """[str] Chemical atomic formula."""
        return self._formula
    @formula.setter
    def formula(self, formula):
        if self._formula: raise AttributeError('cannot set formula')
        self._formula = str(formula)
        if self._Hf is None:
            self.MW = compute_molecular_weight(self.atoms)
        else:
            self.reset_combustion_data()
    
    ### Functional groups ###
    
    @property
    def Dortmund(self):
        """[DortmundGroupCounts] Dictionary-like object with functional group 
        numerical identifiers as keys and the number of groups as values."""
        return self._Dortmund
    @property
    def UNIFAC(self):
        """[UNIFACGroupCounts] Dictionary-like object with functional group 
        numerical identifiers as keys and the number of groups as values."""
        return self._UNIFAC
    @property
    def PSRK(self): 
        """[PSRKGroupCounts] Dictionary-like object with functional group 
        numerical identifiers as keys and the number of groups as values."""
        return self._PSRK
    @property
    def NIST(self): 
        """[NISTGroupCounts] Dictionary-like object with functional group 
        numerical identifiers as keys and the number of groups as values."""
        return self._NIST
    
    ### Equations of state ###
    
    @property
    def eos(self):
        """[object] Instance for solving equations of state."""
        return self._eos
    @property
    def eos_1atm(self): 
        """[object] Instance for solving equations of state at 1 atm."""
        return self._eos_1atm
    
    ### Phase/model handles ###
    
    @property
    def mu(self): 
        """Dynamic viscosity [Pa*s]."""
        return self._mu
    @mu.setter
    def mu(self, value): 
        raise_helpful_handle_error(self._mu)
    
    @property
    def kappa(self): 
        """Thermal conductivity [W/m/K]."""
        return self._kappa
    @kappa.setter
    def kappa(self, value): 
        raise_helpful_handle_error(self._kappa)
        
    @property
    def V(self): 
        """Molar volume [m^3/mol]."""
        return self._V
    @V.setter
    def V(self, value): 
        raise_helpful_handle_error(self._V)
        
    @property
    def Cn(self): 
        """Molar heat capacity [J/mol/K]."""
        return self._Cn
    @Cn.setter
    def Cn(self, value): 
        raise_helpful_handle_error(self._Cn)
        
    @property
    def Psat(self):
        """Vapor pressure [Pa]."""
        return self._Psat
    @Psat.setter
    def Psat(self, value): 
        raise_helpful_handle_error(self._Psat)
        
    @property
    def Hvap(self): 
        """Heat of vaporization [J/mol]."""
        return self._Hvap
    @Hvap.setter
    def Hvap(self, value): 
        raise_helpful_handle_error(self._Hvap)
        
    @property
    def sigma(self): 
        """Surface tension [N/m]."""
        return self._sigma
    @sigma.setter
    def sigma(self, value): 
        raise_helpful_handle_error(self._sigma)
        
    @property
    def epsilon(self): 
        """Relative permitivity [-]."""
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        raise_helpful_handle_error(self._epsilon)
    
    @property
    def S(self): 
        """Entropy [J/mol]."""
        return self._S
    @S.setter
    def S(self, value): 
        raise_helpful_energy_functor_error(self._S)
    
    @property
    def H(self): 
        """Enthalpy [J/mol]."""
        return self._H
    @H.setter
    def H(self, value): 
        raise_helpful_energy_functor_error(self._H)
    
    @property
    def S_excess(self): 
        """Excess entropy [J/mol]."""
        return self._S_excess
    @S_excess.setter
    def S_excess(self, value): 
        raise_helpful_energy_functor_error(self._S_excess)
    
    @property
    def H_excess(self): 
        """Excess enthalpy [J/mol]."""
        return self._H_excess
    @H_excess.setter
    def H_excess(self, value): 
        raise_helpful_energy_functor_error(self._H_excess)
    
    ### Data ###
    
    @property
    def MW(self):
        """Molecular weight [g/mol]."""
        return self._MW
    @MW.setter
    def MW(self, MW):
        if self._MW: raise AttributeError('cannot set molecular weight')
        reset_constant(self, 'MW', float(MW))
    
    @property
    def Tm(self):
        """Normal melting temperature [K]."""
        return self._Tm
    @Tm.setter
    def Tm(self, Tm):
        reset_constant(self, 'Tm', float(Tm))
        self.reset_free_energies()
    
    @property
    def Tb(self):
        """Normal boiling temperature [K]."""
        return self._Tb
    @Tb.setter
    def Tb(self, Tb):
        reset_constant(self, 'Tb', float(Tb))
        self.reset_free_energies()
    
    @property
    def Pt(self):
        """Triple point pressure [Pa]."""
        return self._Pt
    @Pt.setter
    def Pt(self, Pt):
        self._Pt = float(Pt)
    
    @property
    def Tt(self):
        """Triple point temperature [K]."""
        return self._Tt
    @Tt.setter
    def Tt(self, Tt):
        self._Tt = float(Tt)
    
    @property
    def Tc(self):
        """Critical point temperature [K]."""
        return self._Tc
    @Tc.setter
    def Tc(self, Tc):
        reset_constant(self, 'Tc',float(Tc))
    
    @property
    def Pc(self):
        """Critical point pressure [Pa]."""
        return self._Pc
    @Pc.setter
    def Pc(self, Pc):
        reset_constant(self, 'Pc', float(Pc))
    
    @property
    def Vc(self):
        """Critical point molar volume [m^3/mol]."""
        return self._Vc
    @Vc.setter
    def Vc(self, Vc):
        reset_constant(self, 'Vc', float(Vc))
    
    @property
    def Hfus(self):
        """Heat of fusion [J/mol]."""
        return self._Hfus
    @Hfus.setter
    def Hfus(self, Hfus):
        reset_energy_constant(self, 'Hfus', float(Hfus))
    
    @property
    def Sfus(self):
        """Entropy of fusion [J/mol]."""
        return self._Sfus
    @Sfus.setter
    def Sfus(self, Sfus):
        reset_energy_constant(self, 'Sfus', float(Sfus))
    
    
    @property
    def omega(self):
        """Accentric factor [-]."""
        return self._omega
    @omega.setter
    def omega(self, omega):
        reset_constant(self, 'omega', float(omega))
    
    @property
    def dipole(self):
        """Dipole moment [Debye]."""
        return self._dipole
    @dipole.setter
    def dipole(self, dipole):
        reset_constant(self, 'dipole', float(dipole))
    
    @property
    def similarity_variable(self):
        """Similarity variable [-]."""
        return self._similarity_variable
    @similarity_variable.setter
    def similarity_variable(self, similarity_variable):
        reset_constant(self, 'similarity_variable', float(similarity_variable))
    
    @property
    def iscyclic_aliphatic(self):
        """Whether the chemical is cyclic-aliphatic."""
        return self._iscyclic_aliphatic
    @iscyclic_aliphatic.setter
    def iscyclic_aliphatic(self, iscyclic_aliphatic):
        reset_constant(self, 'iscyclic_aliphatic', bool(iscyclic_aliphatic))
    
    
    ### Reaction data ###
    
    @property
    def Hf(self):
        """Heat of formation at reference phase and temperature [J/mol]."""
        return self._Hf
    @Hf.setter
    def Hf(self, Hf):
        self._Hf = float(Hf)
        if self._formula: self.reset_combustion_data()
    
    @property
    def S0(self):
        return self._S0
    @S0.setter
    def S0(self, S0):
        reset_energy_constant(self, 'S0', float(S0))
    
    @property
    def LHV(self):
        """Lower heating value [J/mol]."""
        return self._LHV
    @LHV.setter
    def LHV(self, LHV):
        self._LHV = float(LHV)
    
    @property
    def HHV(self):
        """Higher heating value [J/mol]."""
        return self._HHV
    @HHV.setter
    def HHV(self, HHV):
        self._HHV = float(HHV)
    
    @property
    def combustion(self):
        """dict[str, int] Combustion reaction."""
        return self._combustion
    @combustion.setter
    def combustion(self, combustion):
        self._combustion = dict(combustion)

    ### Computed data ###
        
    @property
    def Stiel_Polar(self):
        """[float] Stiel Polar factor for computing surface tension."""
        Psat = self._Psat
        omega = self._omega
        Tc = self._Tc
        Pc = self._Pc
        if all([Psat, omega, Tc, Pc]):
            P_at_Tr_sixtenths = Psat.try_out(0.6 * Tc)
            if P_at_Tr_sixtenths:
                Stiel_Polar = compute_Stiel_Polar(P_at_Tr_sixtenths, Pc, omega)
            else:
                Stiel_Polar = None
        else:
            Stiel_Polar = None
        return Stiel_Polar
    
    @property
    def Zc(self):
        """[float] Compressibility factor."""
        critical_point = (self._Tc, self._Pc, self._Vc)
        return fn.Z(*critical_point) if all(critical_point) else None
    
    @property
    def has_hydroxyl(self):
        """[bool] Whether or not chemical contains a hydroxyl functional group,
        as determined by the Dortmund/UNIFAC/PSRK functional groups."""
        for dct in (self._Dortmund, self._UNIFAC, self._PSRK):
            for n in (14, 15, 16, 81):
                if n in dct: return True
        return False

    @property
    def atoms(self):
        """dict[str: int] Atom-count pairs based on formula."""
        formula = self._formula
        return get_atoms(formula) if formula else {}
    
    def get_combustion_reaction(self, chemicals=None, conversion=1.0):
        """Return a Reaction object defining the combustion of this chemical.
        If no combustion stoichiometry is available, return None."""
        combustion = self._combustion
        if not combustion: return None
        if len(combustion) == 1: return None
        ID = self._ID
        combustion = combustion.copy()
        combustion[ID] = -1
        return tmo.reaction.Reaction(combustion, ID, conversion, chemicals)
    
    def get_phase(self, T=298.15, P=101325.):
        """Return phase of chemical at given thermal condition.
        
        Examples
        --------
        >>> from thermosteam import Chemical
        >>> Water = Chemical('Water', cache=True)
        >>> Water.get_phase(T=400, P=101325)
        'g'
        """
        if self._locked_state: return self._locked_state
        if self._Tm and T <= self._Tm: return 's'
        if self._Psat and P <= self._Psat(T): return 'g'
        else: return 'l'
    
    ### Data solvers###

    def Tsat(self, P, Tguess=None, Tmin=None, Tmax=None):
        """Return the saturated temperature (in Kelvin) given the pressure (in Pascal)."""
        Tb = self._Tb
        Psat = self._Psat
        if not Psat: return None
        if not Tmin: Tmin = Psat.Tmin + 1.
        if not Tmax: Tmax = Psat.Tmax - 1.
        if Tb:
            if P == 101325: return Tb
            else: Tguess = Tb
        elif not Tguess:
            Tguess = (Tmin + Tmax)/2.0
        try:
            T = IQ_interpolation(lambda T: Psat(T) - P,
                                 Tmin, Tmax, Psat(Tmin) - P, Psat(Tmax) - P,
                                 Tguess, 1e-3, 1e-1, checkroot=False)
        except: return None
        return T

    ### Reinitializers ###
    
    def reset(self, CAS, eos=PR, phase_ref=None,
              smiles=None, InChI=None, InChI_key=None,
              pubchemid=None, iupac_name=None, common_name=None,
              formula=None, MW=None, Tm=None,
              Tb=None, Tc=None, Pc=None, Vc=None, omega=None,
              Tt=None, Pt=None, Hf=None, S0=None, LHV=None, combustion=None,
              HHV=None, Hfus=None, dipole=None,
              similarity_variable=None, iscyclic_aliphatic=None,
              *, metadata=None, phase=None):
        """
        Reset all chemical properties.

        Parameters
        ----------
        CAS : str
            CAS number of chemical to load.
        eos : optional
            Equation of state. The default is Peng Robinson.
        phase_ref : str, optional
            Phase reference. Defaults to the phase at 298.15 K and 101325 Pa.

        """
        try:
            info = metadata or pubchem_db.search_CAS(CAS)
        except:
            pass            
        else:
            if formula and info.formula and get_atoms(formula) != get_atoms(info.formula):
                raise RuntimeError(
                    f'{self.ID} (CAS: {CAS}) formula from database, '
                    f'{info.formula}, does not match with user '
                    f'specification, {formula}')
            if MW and info.MW and MW != info.MW:
                raise RuntimeError(
                    f'{self.ID} (CAS: {CAS}) molecular weight from database, '
                    f'{info.MW:.2f} g/mol, does not match with user '
                    f'specification, {MW:.2f} g/mol')
            smiles = smiles or info.smiles
            InChI = InChI or info.InChI
            InChI_key = InChI_key or info.InChI_key
            pubchemid = pubchemid or info.pubchemid
            iupac_name = iupac_name or info.iupac_name, 
            common_name = common_name or info.common_name
            formula = formula or info.formula
            MW = MW or info.MW
        if formula and not MW:
            MW = compute_molecular_weight(formula)
        self._init_names(CAS, smiles, InChI, InChI_key, 
                         pubchemid, iupac_name, common_name,
                         formula)
        self._init_groups(InChI_key)
        if CAS == '56-81-5': # TODO: Make this part of data
            self._Dortmund = DortmundGroupCounts({2: 2, 3: 1, 14: 2, 81: 1})
        atoms = self.atoms
        self._init_data(CAS, MW, Tm, Tb, Tc, Pc, Vc, omega, Pt, Tt, Hfus,
                        dipole, atoms, similarity_variable, iscyclic_aliphatic)
        self._init_eos(eos, self._Tc, self._Pc, self._omega)
        self._init_handles(CAS, self._MW, self._Tm, self._Tb, self._Tc,
                           self._Pc, self.Zc, self._Vc,
                           self._omega, self._dipole, self._similarity_variable,
                           self._iscyclic_aliphatic, self._eos, self.has_hydroxyl)
        self._locked_state = None
        if phase: self.at_state(phase)
        self._estimate_missing_properties()
        self._init_energies(self._Cn, self._Hvap, self._Psat, self._Hfus, self._Sfus,
                            self._Tm, self._Tb, self._eos, self._eos_1atm,
                            phase_ref)
        self._init_reactions(Hf, S0, LHV, HHV, combustion, atoms)
        if self._formula and self._Hf is not None: self.reset_combustion_data()

    def reset_combustion_data(self, method="Stoichiometry"):
        """Reset combustion data (LHV, HHV, and combution attributes)
        based on the molecular formula and the heat of formation."""
        cd = combustion_data(self.atoms, Hf=self._Hf, MW=self._MW, method=method)
        if not self._MW: self._MW = cd.MW
        self._LHV = cd.LHV
        self._HHV = cd.HHV
        self._combustion = cd.stoichiometry
    
    def reset_free_energies(self):
        """Reset the `H`, `S`, `H_excess`, and `S_excess` functors."""
        if not self._eos:
            self._eos = GCEOS_DUMMY(T=298.15, P=101325.)
            self._eos_1atm = self._eos.to_TP(298.15, 101325)
        self._init_energies(self._Cn, self._Hvap, self._Psat, self._Hfus, self._Sfus,
                            self._Tm, self._Tb, self._eos, self._eos_1atm, self._phase_ref)

    ### Initializers ###
    
    def _init_names(self, CAS, smiles, InChI, InChI_key,
                    pubchemid, iupac_name, common_name, formula):
        self._CAS = CAS
        self._smiles = smiles
        self._InChI = InChI
        self._InChI_key = InChI_key
        self._pubchemid = pubchemid
        self._iupac_name = iupac_name
        self._common_name = common_name
        self._formula = formula
        
    def _init_groups(self, InChI_key):
        if InChI_key in DDBST_UNIFAC_assignments:
            self._UNIFAC = DDBST_UNIFAC_assignments[InChI_key]
        else:
            self._UNIFAC = UNIFACGroupCounts()
        if InChI_key in DDBST_MODIFIED_UNIFAC_assignments:
            self._Dortmund = DDBST_MODIFIED_UNIFAC_assignments[InChI_key]
        else:
            self._Dortmund = DortmundGroupCounts()
        if InChI_key in DDBST_PSRK_assignments:
            self._PSRK = DDBST_PSRK_assignments[InChI_key]
        else:
            self._PSRK = PSRKGroupCounts()
        # TODO: Download data for NIST UNIFAC group assignments
        self._NIST = NISTGroupCounts()

    def _init_data(self, CAS, MW, Tm, Tb, Tc, Pc, Vc, omega, Pt, Tt, Hfus,
                   dipole, atoms, similarity_variable, iscyclic_aliphatic):
        self._MW = MW
        self._Tm = Tm or normal_melting_point_temperature(CAS)
        self._Tb = Tb = Tb or normal_boiling_point_temperature(CAS)

        # Critical Point
        self._Tc = Tc = Tc or critical_point_temperature(CAS)
        self._Pc = Pc = Pc or critical_point_pressure(CAS)
        self._Vc = Vc or critical_point_volume(CAS)
        data = (Tb, Tc, Pc)
        self._omega = omega or acentric_factor(CAS) or (acentric_factor_LK(*data) if all(data) else None)

        # Triple point
        self._Pt = Pt or triple_point_pressure(CAS)
        self._Tt = Tt or triple_point_temperature(CAS)

        # Energy
        self._Hfus = heat_of_fusion(CAS) or 0. if Hfus is None else Hfus
        self._Sfus = None if Hfus is None or Tm is None else Hfus / Tm 
        
        # Other
        self._dipole = dipole or dipole_moment(CAS)
        atoms = atoms or self.atoms
        if atoms and not similarity_variable:
            similarity_variable = compute_similarity_variable(atoms, MW)
        self._similarity_variable = similarity_variable
        self._iscyclic_aliphatic = iscyclic_aliphatic or False

    def _init_reactions(self, Hf, S0, LHV, HHV, combustion, atoms):
        Hvap_298K = None
        if Hf is None:
            Hvap_298K = self.Hvap.try_out(298.15)
            Hf = heat_of_formation(self._CAS, self._phase_ref,
                                   Hvap_298K, self.Hfus) 
        if S0 is None:
            Hvap_298K = self.Hvap.try_out(298.15) if Hvap_298K is None else Hvap_298K
            Svap_298K = None if Hvap_298K is None else Hvap_298K / 298.15
            S0 = absolute_entropy_of_formation(self._CAS, self._phase_ref,
                                               Svap_298K, self.Sfus) or 0.
        self._Hf = Hf
        self.S0 = S0
        atoms = atoms or self.atoms
        if not all([LHV, HHV, combustion]) and atoms and Hf:
            cd = combustion_data(atoms, Hf=self._Hf, MW=self._MW, missing_handling='Ash')
            LHV = cd.LHV
            HHV = cd.HHV
            combustion = cd.stoichiometry
        self._LHV = LHV
        self._HHV = HHV
        self._combustion = combustion

    def _init_eos(self, eos, Tc, Pc, omega):
        self._eos = create_eos(eos, Tc, Pc, omega)
        self._eos_1atm = self._eos.to_TP(298.15, 101325)

    def _init_handles(self, CAS, MW, Tm, Tb, Tc, Pc, Zc, Vc, omega,
                      dipole, similarity_variable, iscyclic_aliphatic, eos,
                      has_hydroxyl):
        # Vapor pressure
        data = (CAS, Tb, Tc, Pc, omega)
        self._Psat = Psat = vapor_pressure_handle(data)
        
        # Volume
        sdata = (CAS,)
        ldata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, Psat, eos, dipole, has_hydroxyl)
        gdata = (CAS, Tc, Pc, omega, eos)
        self._V = V = volume_handle(sdata, ldata, gdata)
        
        # Heat capacity
        Cn = PhaseTHandle('Cn')
        sdata = (CAS, similarity_variable, MW)
        ldata = (CAS, Tb, Tc, omega, MW, similarity_variable, Cn.g)
        gdata = (CAS, MW, similarity_variable, iscyclic_aliphatic)
        self._Cn = Cn = heat_capacity_handle(sdata, ldata, gdata, Cn)
        
        # Heat of vaporization
        data = (CAS, Tb, Tc, Pc, omega, similarity_variable, Psat, V)
        self._Hvap = heat_of_vaporization_handle(data)
        
        # Viscosity
        ldata = (CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, V.l)
        gdata = (CAS, MW, Tc, Pc, Zc, dipole)
        self._mu = mu = viscosity_handle(None, ldata, gdata)
        
        # Conductivity
        ldata = (CAS, MW, Tm, Tb, Tc, Pc, omega, V.l)
        gdata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, V.g, Cn.g, mu.g)
        self._kappa = thermal_conductivity_handle(None, ldata, gdata)
        
        # Surface tension
        data = (CAS, MW, Tb, Tc, Pc, Vc, Zc,
                omega, self.Stiel_Polar)
        self._sigma = surface_tension_handle(data)
        
        # Other
        self._epsilon = permitivity_handle((CAS, V.l,))
        self._label_handles()
        
        # self.delta = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def _estimate_missing_properties(self):
        # Melting temperature is a week function of pressure,
        # so assume the triple point temperature is the 
        # melting temperature
        Tm = self._Tm
        Tt = self._Tt
        Pt = self._Pt
        Tc = self._Tc
        Pc = self._Pc
        Vc = self._Vc
        if not Tm and Tt:
            self._Tm = Tm = Tt
        if not Tt and Tm:
            self._Tt = Tt = Tm
        if not Pt and Tt:
            self._Pt = Pt = self._Psat.try_out(Tt)
        
        # Find missing critical property, if only two are given.
        critical_point = (Tc, Pc, Vc)
        N_values = sum([bool(i) for i in critical_point])
        if N_values == 2:
            third_property = Ihmels(*critical_point)
            if not Tc: self._Tc = third_property
            elif not Pc: self._Pc = third_property
            elif not Vc: self._Vc = third_property
        Tb = self._Tb
        if not Tb:
            self._Tb = Tb = self.Tsat(101325)
            
        omega = self._omega
        if not omega and Pc and Tc:
            P_at_Tr_seventenths = self._Psat.try_out(0.7 * Tc)
            if P_at_Tr_seventenths:
                omega = acentric_factor_definition(P_at_Tr_seventenths, Pc)
            if not omega and Tb:
                omega = acentric_factor_LK(Tb, Tc, Pc)
            self._omega = omega

    def _init_energies(self, Cn, Hvap, Psat, Hfus, Sfus, Tm, Tb, eos, eos_1atm,
                       phase_ref=None):        
        # Reference
        P_ref = self.P_ref
        T_ref = self.T_ref
        H_ref = self.H_ref
        S0 = 0. # Replaced later in _init_reactions method
        single_phase = self._locked_state
        if isinstance(Cn, PhaseHandle):
            Cn_s = Cn.s
            Cn_l = Cn.l
            Cn_g = Cn.g
            has_Cns = bool(Cn_s)
            has_Cnl = bool(Cn_l)
            has_Cng = bool(Cn_g)
        elif Cn and single_phase:
            self._phase_ref = single_phase
            self._H = Enthalpy.functor(Cn, T_ref, H_ref)
            self._S = Entropy.functor(Cn, T_ref, S0)
            Cn_s = Cn_l = Cn_g = Cn
            has_Cns = has_Cnl = has_Cng = True
        else:
            has_Cns = has_Cnl = has_Cng = False
        if phase_ref: phase_ref = phase_ref[0]
        self._phase_ref = phase_ref
        if any((has_Cns, has_Cnl, has_Cng)):
            if not phase_ref:
                if Tm and T_ref <= Tm:
                    self._phase_ref = phase_ref = 's'
                elif Tb and T_ref >= Tb:
                    self._phase_ref = phase_ref = 'g'
                else:
                    self._phase_ref = phase_ref = 'l'
            if Hvap:
                Hvap_Tb = Hvap.try_out(Tb) if Tb else None
                Svap_Tb = Hvap_Tb / Tb if Hvap_Tb else None
            else:
                Hvap_Tb = Svap_Tb = None
            
            # Enthalpy and entropy integrals
            if phase_ref != 'l' and has_Cnl and (Tm and Tb):
                H_int_Tm_to_Tb_l = Cn_l.integrate_by_T(Tm, Tb)
                S_int_Tm_to_Tb_l = Cn_l.integrate_by_T_over_T(Tm, Tb)
            else:
                H_int_Tm_to_Tb_l = S_int_Tm_to_Tb_l = None
            if phase_ref == 's' and has_Cns and Tm:
                H_int_T_ref_to_Tm_s = Cn_s.integrate_by_T(T_ref, Tm)
                S_int_T_ref_to_Tm_s = Cn_s.integrate_by_T_over_T(T_ref, Tm)
            else:
                H_int_T_ref_to_Tm_s = S_int_T_ref_to_Tm_s = None
            if phase_ref == 'g' and has_Cng and Tb:
                H_int_Tb_to_T_ref_g = Cn_g.integrate_by_T(Tb, T_ref)
                S_int_Tb_to_T_ref_g = Cn_g.integrate_by_T_over_T(Tb, T_ref)
            else:
                H_int_Tb_to_T_ref_g = S_int_Tb_to_T_ref_g = None
            if phase_ref == 'l':
                if has_Cnl:
                    if Tb:
                        H_int_T_ref_to_Tb_l = Cn_l.integrate_by_T(T_ref, Tb)
                        S_int_T_ref_to_Tb_l = Cn_l.integrate_by_T_over_T(T_ref, Tb)
                    else:
                        H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
                    if Tm:
                        H_int_Tm_to_T_ref_l = Cn_l.integrate_by_T(Tm, T_ref)
                        S_int_Tm_to_T_ref_l = Cn_l.integrate_by_T_over_T(Tm, T_ref)
                    else:
                        H_int_Tm_to_T_ref_l = S_int_Tm_to_T_ref_l = None
                else:
                    H_int_Tm_to_T_ref_l = S_int_Tm_to_T_ref_l = \
                    H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
    
            # Excess data
            if isinstance(eos, GCEOS_DUMMY):
                H_dep_ref_g = S_dep_ref_g = H_dep_ref_l = S_dep_ref_l = \
                H_dep_T_ref_Pb = S_dep_T_ref_Pb = H_dep_Tb_Pb_g = H_dep_Tb_P_ref_g = \
                S_dep_Tb_P_ref_g = S_dep_Tb_Pb_g = 0
            else:
                if phase_ref == 'g':
                    eos_phase_ref = eos.to_TP(T_ref, P_ref)
                    H_dep_ref_g = eos_phase_ref.H_dep_g
                    S_dep_ref_g = eos_phase_ref.S_dep_g
                elif phase_ref == 'l':
                    eos_phase_ref = eos.to_TP(T_ref, P_ref)
                    H_dep_ref_l = eos_phase_ref.H_dep_l
                    S_dep_ref_l = eos_phase_ref.S_dep_l
                    eos_T_ref_Pb = eos.to_TP(T_ref, 101325)
                    H_dep_T_ref_Pb = eos_T_ref_Pb.H_dep_l
                    S_dep_T_ref_Pb = eos_T_ref_Pb.S_dep_l
                if Tb:
                    try:
                        eos_Tb = eos.to_TP(Tb, 101325)
                        eos_Tb_P_ref = eos.to_TP(Tb, P_ref)
                        H_dep_Tb_Pb_g = eos_Tb.H_dep_g
                        H_dep_Tb_P_ref_g = eos_Tb_P_ref.H_dep_g
                        S_dep_Tb_P_ref_g = eos_Tb_P_ref.S_dep_g
                        S_dep_Tb_Pb_g = eos_Tb.S_dep_g
                    except:
                        S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = H_dep_Tb_P_ref_g = \
                        H_dep_Tb_Pb_g = 0.
                else:
                    S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = H_dep_Tb_P_ref_g = \
                    H_dep_Tb_Pb_g = 0.
            
            # Enthalpy and Entropy
            if not single_phase:
                if phase_ref == 's':
                    sdata = (Cn_s, T_ref, H_ref)
                    ldata = (Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref)
                    gdata = (Cn_g, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref)
                    self._H = EnthalpyRefSolid(sdata, ldata, gdata)
                    sdata = (Cn_s, T_ref, S0)
                    ldata = (Cn_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S0)
                    gdata = (Cn_g, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S0)
                    self._S = EntropyRefSolid(sdata, ldata, gdata)
                elif phase_ref == 'l':
                    sdata = (Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref)
                    ldata = (Cn_l, T_ref, H_ref)
                    gdata = (Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, Tb, H_ref)
                    self._H = EnthalpyRefLiquid(sdata, ldata, gdata)
                    sdata = (Cn_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S0)
                    ldata = (Cn_l, T_ref, S0)
                    gdata = (Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb, Tb, P_ref, S0)
                    self._S = EntropyRefLiquid(sdata, ldata, gdata)
                elif phase_ref == 'g':
                    sdata = (Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref)
                    ldata = (Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref)
                    gdata = (Cn_g, T_ref, H_ref)
                    self._H = EnthalpyRefGas(sdata, ldata, gdata)
                    sdata = (Cn_s, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S0)
                    ldata = (Cn_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S0)
                    gdata = (Cn_g, T_ref, P_ref, S0)
                    self._S = EntropyRefGas(sdata, ldata, gdata)
            
            # Excess energies
            if phase_ref == 's':
                self._H_excess = ExcessEnthalpyRefSolid((), (), ())
                self._S_excess = ExcessEntropyRefSolid((), (), ())
            elif phase_ref == 'l':
                gdata = (eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g)
                self._H_excess = ExcessEnthalpyRefLiquid((), (), gdata)
                gdata = (eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g)
                self._S_excess = ExcessEntropyRefLiquid((), (), gdata)
            elif phase_ref == 'g':
                ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_1atm)
                gdata = (eos, H_dep_ref_g)
                self._H_excess = ExcessEnthalpyRefGas((), ldata, gdata)
                ldata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_1atm)
                gdata = (eos, S_dep_ref_g)
                self._S_excess = ExcessEntropyRefGas((), ldata, gdata)
                
            if single_phase:
                getfield = getattr
                self._H_excess = getfield(self._H_excess, single_phase)
                self._S_excess = getfield(self._S_excess, single_phase)
        else:
            self._H = self._S = self._S_excess = self._H_excess = None

    ### Filling missing values ###

    def get_key_property_names(self):
        """Return the attribute names of key properties required to model a process."""
        if not self._locked_state and self._phase_ref != 's':
            return ('V', 'S', 'H', 'Cn', 'Psat', 'Tb', 'Hvap')
        else:
            return ('V', 'S', 'H', 'Cn')

    def default(self, properties=None):
        """
        Default all `properties` with the chemical properties of water. If no
        `properties` given, all essential chemical properties that are missing
        are defaulted. `properties` which are still missing are returned as set.
        
        Parameters
        ----------
        properties : Iterable[str], optional
            Names of chemical properties to default.
        
        Returns
        -------
        missing_properties : set[str]
            Names of chemical properties that are still missing.
        
        Examples
        --------
        >>> from thermosteam import Chemical
        >>> Substance = Chemical.blank('Substance')
        >>> missing_properties = Substance.default()
        >>> sorted(missing_properties)
        ['Dortmund', 'Hfus', 'Hvap', 'PSRK', 'Pc', 'Psat', 'Pt', 'Sfus', 'Tb', 'Tc', 'Tm', 'Tt', 'UNIFAC', 'V', 'Vc', 'dipole', 'iscyclic_aliphatic', 'omega', 'similarity_variable']
        
        Note that missing properties does not include essential properties volume, heat capacity, and conductivity.
        
        """
        if not properties:
            properties = self.get_missing_properties(properties)   
        hasfield = hasattr
        # Default to Water property values
        if 'MW' in properties:
            self._MW = MW = 1
        else:
            MW = self._MW
        if 'sigma' in properties:
            self._sigma.add_model(0.072055)
        if 'mu' in properties:
            mu = self._mu
            if hasfield(mu, 'l'):
                mu.l.add_model(0.00091272)
            elif not mu:
                mu.add_model(0.00091272)
        if 'V' in properties:
            V = self._V
            V_default = fn.rho_to_V(1050, MW)
            if hasfield(V, 'l'):
                V.l.add_model(V_default)
            elif not V:
                V.add_model(V_default)
        if 'kappa' in properties:
            kappa = self._kappa
            if hasfield(kappa, 'l'):
                kappa.l.add_model(0.5942)
            if not kappa:
                kappa.add_model(0.5942)
        if self._formula:
            if any([i in properties for i in ('HHV', 'LHV', 'combustion', 'Hf')]):
                if 'Hf' in properties:
                    method = 'Dulong'
                else:
                    method = 'Stoichiometry'
                stoichiometry = combustion_stoichiometry(self.atoms, MW=MW, missing_handling='Ash')
                try:
                    cd = combustion_data(self.atoms, Hf=self._Hf, MW=MW, stoichiometry=stoichiometry, method=method)
                except:
                    if 'Hf' in properties:
                        self._Hf = 0.
                    if 'HHV' in properties:
                        self._HHV = 0.
                    if 'LHV' in properties:
                        self._LHV = 0.
                    if 'combustion' in properties:
                        self._combustion = stoichiometry
                else:
                    if 'Hf' in properties:
                        self._Hf = cd.Hf
                    if 'HHV' in properties:
                        self._HHV = cd.HHV
                    if 'LHV' in properties:
                        self._LHV = cd.LHV
                    if 'combustion' in properties:
                        self._combustion = stoichiometry
        else:
            if 'LHV' in properties:
                self._LHV = 0
            if 'HHV' in properties:
                self._HHV = 0
            if 'combustion' in properties:
                self._combustion = {}
        if 'Hf' in properties:
            self._Hf = 0
        if 'epsilon' in properties:
            self._epsilon.add_model(0)
        if 'phase_ref' in properties:
            self._phase_ref = 'l'
        if 'eos' in properties:
            self._eos = GCEOS_DUMMY(T=298.15, P=101325.)
            self._eos_1atm = self._eos.to_TP(298.15, 101325)
        if 'Cn' in properties:
            MW = self._MW
            Cn = self._Cn
            phase_ref = self._phase_ref
            getfield = getattr
            single_phase = self._locked_state
            if single_phase:
                Cn.add_model(4.18*MW)
                Cn_phase = Cn
            else:
                Cn_phase = getfield(Cn, phase_ref)
                Cn_phase.add_model(4.18*MW)
            self.reset_free_energies()
        if not self._H:
            self.reset_free_energies()
        missing = set(properties)
        missing.difference_update({'MW', 'CAS', 'Cn', 'Hf', 'sigma',
                                   'mu', 'kappa', 'LHV', 'HHV', 'epsilon', 'H',
                                   'S', 'H_excess', 'S_excess', 'phase_ref', 
                                   'combustion'})
        return missing
    
    def get_missing_properties(self, properties=None):
        """
        Return a list all missing thermodynamic properties.
        
        Examples
        --------
        >>> from thermosteam import Chemical
        >>> Substance = Chemical.blank('Substance', phase_ref='l')
        >>> sorted(Substance.get_missing_properties())
        ['Cn', 'Dortmund', 'H', 'HHV', 'H_excess', 'Hf', 'Hfus', 'Hvap', 'LHV', 'MW', 'PSRK', 'Pc', 'Psat', 'Pt', 'S', 'S_excess', 'Sfus', 'Tb', 'Tc', 'Tm', 'Tt', 'UNIFAC', 'V', 'Vc', 'combustion', 'dipole', 'epsilon', 'iscyclic_aliphatic', 'kappa', 'mu', 'omega', 'sigma', 'similarity_variable']
        
        """
        getfield = getattr
        return [i for i in (properties or _checked_properties) if not getfield(self, i)]
    
    def copy_models_from(self, other, names=None):
        """Copy models from other."""
        if names:
            for i in names:
                if i not in _model_and_phase_properties:
                    raise ValueError(f"{i} is not a valid model name; "
                                      "names must be a subset of "
                                     f"{_model_and_phase_properties}")
        else:
            missing_handles = self.get_missing_properties(_model_and_phase_handles)
            other_missing_handles = other.get_missing_properties(_model_and_phase_handles)
            names = set(missing_handles).difference(other_missing_handles)
        getfield = getattr
        isa = isinstance
        phase = self._locked_state
        other_phase = other._locked_state
        for key in names:
            handle = getfield(self, key)
            other_handle = getfield(other, key)
            if isa(handle, ThermoModelHandle):
                if isa(other_handle, ThermoModelHandle):
                    models = other_handle._models
                elif isa(other_handle, PhaseHandle):
                    models = getfield(other_handle, phase)._models
                handle._models = models.copy()
            elif isa(handle, PhaseHandle):
                if isa(other_handle, ThermoModelHandle):
                    handle = getfield(handle, other_phase)
                    handle._models = other_handle._models.copy()
                elif isa(other_handle, PhaseHandle):
                    for i, model_handle in handle:
                        models = getfield(other_handle, i)._models.copy()
                        model_handle._models = models
        if {'Cn', 'Hvap'}.intersection(names): self.reset_free_energies()
    
    @property
    def locked_state(self):
        """[str] Constant phase of chemical."""
        return self._locked_state
    
    @property
    def N_solutes(self):
        """[int] Number of molecules formed when solvated."""
        return getattr(self, '_N_solutes', None)
    @N_solutes.setter
    def N_solutes(self, N_solutes):
        self._N_solutes = int(N_solutes)
    
    def at_state(self, phase, copy=False):
        """
        Set the state of chemical.
        
        Examples
        --------
        >>> from thermosteam import Chemical
        >>> N2 = Chemical('N2')
        >>> N2.at_state(phase='g')
        >>> N2.H(298.15) # No longer a function of phase
        0.0
        """
        if copy:
            new = self.copy(self._ID)
            new.at_state(phase)
            return new
        locked_state = self._locked_state
        if locked_state:
            if locked_state != phase:
                raise TypeError(f"{self}'s state is already locked")   
            else:
                return         
        elif phase:
            lock_phase(self, phase)
        else:
            raise ValueError(f"invalid phase {repr(phase)}")
    
    def show(self):
        """Print all specifications"""
        info = chemical_identity(self, pretty=True)
        for header, fields in _chemical_fields.items():
            section = []
            for field in fields:
                value = getattr(self, field)
                field = field.lstrip('_')
                if value is None:
                    line = f"{field}: None"
                if callable(value):
                    line = f"{display_asfunctor(value, name=field, var=field, show_var=False)}"
                else:
                    if isinstance(value, (int, float)):
                        line = f"{field}: {value:.5g}"
                        units = chemical_units_of_measure.get(field, "")
                        if units: line += f' {units}'
                    else:
                        value = str(value)
                        line = f"{field}: {value}"
                        if len(line) > 27: line = line[:27] + '...'
                section.append(line)
            if section:
                info += header + ("\n" + 9*" ").join(section)
        print(info)
        
    _ipython_display_ = show
    
    def __str__(self):
        return self._ID
    
    def __repr__(self):
        return f"Chemical('{self}')"
    
def lock_phase(chemical, phase):
    phase = phase[0]
    check_phase(phase)
    getfield = getattr
    setfield = object.__setattr__
    hasfield = hasattr
    for field in _phase_handles:
        phase_property = getfield(chemical, field)
        if hasfield(phase_property, phase):
            model_handle = getfield(phase_property, phase)
            setfield(chemical, field, model_handle)
    for field in _energy_handles:
        try: phase_property = getfield(chemical, field)
        except: continue
        if hasfield(phase_property, phase):
            functor = getfield(phase_property, phase)
            setfield(chemical, field, functor)
    Cn = chemical.Cn
    chemical._phase_ref = phase
    chemical._H = Enthalpy.functor(Cn, chemical.T_ref, chemical.H_ref)
    S0 = chemical._S0 if hasfield(chemical, '_S0') else 0.
    chemical._S = Entropy.functor(Cn, chemical.T_ref, S0)
    chemical._locked_state = phase

# # Fire Safety Limits
# self.Tflash = Tflash(CAS)
# self.Tautoignition = Tautoignition(CAS)
# self.LFL = LFL(CASRN=CAS, atoms=self.atoms, Hc=self.Hc)
# self.UFL = UFL(CASRN=CAS, atoms=self.atoms, Hc=self.Hc)

# # Chemical Exposure Limits
# self.TWA = TWA(CASRN=CAS)
# self.STEL = STEL(CASRN=CAS)
# self.Ceiling = Ceiling(CASRN=CAS)
# self.Skin = Skin(CASRN=CAS)
# self.Carcinogen = Carcinogen(CASRN=CAS)

# # Misc
# self.dipole = dipole(CASRN=CAS) # Units of Debye
# self.Stockmayer = Stockmayer(CASRN=CAS, Tm=self.Tm, Tb=self.Tb, Tc=self.Tc,
#                               Zc=self.Zc, omega=self.omega)

# # Environmental
# self.GWP = GWP(CASRN=CAS)
# self.ODP = ODP(CASRN=CAS)
# self.logP = logP(CASRN=CAS)

# # Analytical
# self.RI, self.RIT = refractive_index(CASRN=CAS)
# self.conductivity, self.conductivityT = conductivity(CASRN=CAS)
    