# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
try:
    import CoolProp
except:    
    add_ID = True
else:
    add_ID = False
import thermosteam as tmo
from warnings import warn
from flexsolve import IQ_interpolation, aitken_secant
from chemicals.identifiers import pubchem_db
from chemicals.phase_change import (Tb as normal_boiling_point_temperature,
                                    Tm as normal_melting_point_temperature,
                                    Hfus as heat_of_fusion)
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
from chemicals.reaction import (
    Hf as heat_of_formation,
    S0 as absolute_entropy_of_formation
)
from chemicals.elements import (
    similarity_variable as compute_similarity_variable, 
    molecular_weight as compute_molecular_weight,
    charge_from_formula,
    atoms_to_Hill,
    get_atoms,
)
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
                   display_asfunctor)
from .units_of_measure import chemical_units_of_measure
from .utils import copy_maybe, check_valid_ID
from . import functional as fn 
from ._phase import check_phase, valid_phases
from . import units_of_measure as thermo_units
from chemicals.utils import Z
from thermo.eos import IG, PR
from thermo import (
    TDependentProperty, TPDependentProperty,
    VaporPressure, 
    EnthalpyVaporization,
    SurfaceTension,
    SublimationPressure, EnthalpySublimation,
    VolumeSolid, VolumeLiquid, VolumeGas, VolumeSupercriticalLiquid,
    HeatCapacitySolid, HeatCapacityLiquid, HeatCapacityGas,
    ThermalConductivityLiquid, ThermalConductivityGas,
    ViscosityLiquid, ViscosityGas,
    PermittivityLiquid,
)

# from .solubility import SolubilityParameter
# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

__all__ = ('Chemical',)

# %% Temporarilly here

# TODO: Make a data table for new additions (not just for common sugars).
# Source: PubChem
sugar_solid_densities = {
    '50-99-7': 0.0001125975, 
    '57-48-7': 0.0001063495,
    '3458-28-4': 0.00011698,
    '25990-60-7': 9.84459e-05,
    '59-23-4': 0.000120104,   
}

class CompressibilityFactor:
    __slots__ = ('V',)
    
    def __init__(self, V):
        self.V = V
    
    def Z(self, T, P):
        return Z(T, P, self.V(T, P))
    
    def __repr__(self):
        return f"{type(self).__name__}(self.V)"

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
    except: return IG(T=298.15, P=101325.)

# %% Resetting data

def reset_constant(chemical, var, value):
    getfield = getattr
    hasfield = hasattr
    setfield = setattr
    setfield(chemical, '_'+var, value)
    isa = isinstance
    for handle in _model_and_phase_handles: 
        handle = getfield(chemical, handle, None)
        if handle is None: continue
        if isa(handle, PhaseHandle):
            for phase, obj in handle:
                if hasfield(obj, var): setfield(obj, var, value)
        elif hasfield(handle, var): setfield(handle, var, value)

def reset_energy_constant(chemical, var, value):
    getfield = getattr
    hasfield = hasattr
    setfield = object.__setattr__
    setfield(chemical, '_'+var, value)
    isa = isinstance
    for handle in _energy_handles: 
        handle = getfield(chemical, handle, None)
        if handle is None: continue
        if isa(handle, PhaseHandle):
            for phase, obj in handle:
                if hasfield(obj, var): setfield(obj, var, value)
        elif hasfield(handle, var):
            setfield(handle, var, value)

def raise_helpful_handle_error(var, handle):
    if isinstance(handle, PhaseHandle):
        raise AttributeError(
            f"cannot set '{var}'; use `add_method` "
            f"to modify the thermodynamic properties for "
            f"each phase (e.g. {var}.l.add_method(...))")
    elif isinstance(handle, TDependentProperty):
        raise AttributeError(
            f"cannot set '{var}'; use `{var}.add_method` to "
            f"modify the thermodynamic property instead")
    else:
        raise Exception(
            'expected either a PhaseHandle or a TDependentProperty object; '
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

_model_handles = ('_Psat', '_Hvap', '_sigma', '_epsilon', '_Hsub', '_Psub')
    
_phase_handles = ('_kappa', '_V', '_Cn', '_mu')

_energy_handles = ('_S_excess', '_H_excess', '_S', '_H')

_model_and_phase_handles = _model_handles + _phase_handles

_model_and_phase_properties = ('Psat', 'Hvap', 'sigma', 'epsilon',
                               'kappa', 'V', 'Cn', 'mu')

_handles = _model_and_phase_handles + _energy_handles

_data = ('_MW', '_Tm', '_Tb', '_Tt', '_Tc', '_Pt', '_Pc', '_Vc',
         '_Hf', '_S0', '_LHV', '_HHV', '_Hfus', '_Sfus', '_omega', '_dipole',
         '_similarity_variable', '_iscyclic_aliphatic', '_combustion',)

_functor_data = {'MW', 'Tm', 'Tb', 'Tt', 'Pt', 'Tc', 'Pc', 'Vc',
                 'Hfus', 'Sfus', 'omega', 'dipole',
                 'similarity_variable', 'iscyclic_aliphatic'}

_checked_properties = (
    'phase_ref', 'eos', 
    'S_excess', 'H_excess', 'mu', 'kappa', 'V', 'S', 
    'H', 'Cn', 'Psat', 'Hvap', 'sigma', 'epsilon', 
    'Dortmund', 'UNIFAC', 'PSRK', 'Hf', 'LHV', 'HHV',
    'combustion', *_functor_data
)

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
            * Name, in IUPAC form or common form or an alias registered in PubChem
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
    Chemical objects contain the same thermodynamic models and data as
    `thermo <https://thermo.readthedocs.io/>`_'s Chemical objects, but present
    an API more suitable for `BioSTEAM <https://biosteam.readthedocs.io/en/latest/>`_'s needs.
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
             Pt: 610 Pa
             Pc: 2.2048e+07 Pa
             Vc: 5.6e-05 m^3/mol
             Hf: -2.8582e+05 J/mol
             S0: 70 J/K/mol
             LHV: -44011 J/mol
             HHV: -0 J/mol
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
    101417.99665995422
    >>> # Surface tension (N/m)
    >>> Water.sigma(T=298.15)
    0.07197220523022962
    >>> # Molar volume (m^3/mol)
    >>> Water.V(phase='l', T=298.15, P=101325)
    1.806...e-05
    
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
    
    Temperature dependent properties are managed by objects:
    
    >>> Water.Psat
    VaporPressure(CASRN="7732-18-5 (Water)", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="IAPWS")

    Phase dependent properties have attributes with model handles for each phase:

    >>> Water.V
    <PhaseTPHandle(phase, T, P) -> V [m^3/mol]>
    >>> # Water.V.l
    >>> # VolumeLiquid(CASRN="7732-18-5 (Water)", MW=18.01528, Tb=373.124, Tc=647.14, Pc=22048320.0, Vc=5.6-05, Zc=0.2294727, omega=0.344, dipole=1.85, Psat=VaporPressure(CASRN="7732-18-5 (Water)", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="WAGNER_MCGARRY"), eos=[PR(Tc=647.14, Pc=22048320.0, omega=0.344, T=298.15, P=101325.0)], extrapolation="constant", method="VDI_PPDS", method_P="COSTALD_COMPRESSED", tabular_extrapolation_permitted=True)
    
    A new model can be added easily using `add_method`, for example:
        
    >>> def User_antoine_model(T):
    ...     return 10.0**(10.116 -  1687.537 / (T - 42.98))
    >>> Water.Psat.add_method(f=User_antoine_model, Tmin=273.20, Tmax=473.20)
    >>> Water.Psat
    VaporPressure(CASRN="7732-18-5 (Water)", Tb=373.124, Tc=647.14, Pc=22048320.0, omega=0.344, extrapolation="AntoineAB|DIPPR101_ABC", method="USER_METHOD")

    The `add_method` method is a high level interface that even lets you create a constant model:
        
    >>> Water.Cn.l.add_method(75.31) 
    >>> Water.Cn('l', 350)
    75.31

    .. Note::
       Because no bounds were given, the model assumes it is valid across all temperatures and pressures.

    Choose what model to use through the `method` attribute:
    
    >>> list(sorted(Water.Cn.l.all_methods))
    ['CRCSTD',
     'DADGOSTAR_SHAW',
     'POLING_CONST',
     'ROWLINSON_BONDI',
     'ROWLINSON_POLING',
     'USER_METHOD',
     'WEBBOOK_SHOMATE',
     'ZABRANSKY_SPLINE_C']
    >>> Water.Cn.l.method = 'ZABRANSKY_SPLINE_C'
    
    .. note::
    
        For details on the available methods and the capabilities of objects
        like VaporPressure and VolumeLiquid, visit 
        `thermo's documentation <https://thermo.readthedocs.io/>`_.
    
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
                 '_phase_ref', '_eos', 
                 '_aliases', *_names, *_groups, 
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
        ID = ID.strip()
        if not phase and len(ID) > 2 and ID[-2] == ',':
            phase = ID[-1]
            if phase not in valid_phases:
                raise ValueError(f'invalid phase {repr(phase)} encountered while parsing ID')
            ID = ID[:-2]
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
            if mu: self._mu.add_method(mu)
            if Cn: self._Cn.add_method(Cn)
            if kappa: self._kappa.add_method(kappa)
            if V: self._V.add_method(V)
        else:
            multi_phase_items = (('mu', mu),
                                 ('Cn', Cn),
                                 ('kappa', kappa), 
                                 ('Cp', Cp),
                                 ('rho', rho),
                                 ('V', V))
            for i,j in multi_phase_items:
                if j: raise ValueError(f'must specify phase to set {i} model')
        if sigma: self._sigma.add_method(sigma)
        if epsilon: self._epsilon.add_method(epsilon)
        if Psat: self._Psat.add_method(Psat)
        if Hvap: self._Hvap.add_method(Hvap)
        if default: self.default()
        if phase:
            if Cp: self._Cn.add_method(Cp * self._MW)
            if rho: self._V.add_method(fn.rho_to_V(rho, self._MW))
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
        the database, and load all possible models from given data."""
        self = super().__new__(cls)
        self._ID = ID
        self.reset(CAS, eos, phase_ref, phase=phase, **data)
        return self

    @classmethod
    def blank(cls, ID, CAS=None, phase_ref=None, phase=None,
              formula=None, aliases=None, synonyms=None, **data):
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
                 S0: 0 J/K/mol
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
        self._ID = ID
        self._eos = None
        self._UNIFAC = UNIFACGroupCounts()
        self._Dortmund = DortmundGroupCounts()
        self._PSRK = PSRKGroupCounts()
        self._NIST = NISTGroupCounts()
        self.aliases = aliases or ()
        if synonyms: 
            if isinstance(synonyms, str):
                self.aliases.add(synonyms)
            else:
                self.aliases.update(synonyms)
        setfield = setattr
        for i in _names: setfield(self, i, None)
        for i in _data: setfield(self, i, None)
        for i in _energy_handles: setfield(self, i, None)
        check_valid_ID(ID)
        self._phase_ref = phase_ref or phase
        self._CAS = CAS or ID
        for i,j in data.items(): setfield(self, '_' + i , j)
        if formula: self.formula = formula
        self._eos = create_eos(PR, self._Tc, self._Pc, self._omega)
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
        self._estimate_missing_properties(None)
        self._init_handles(CAS, self._MW, self._Tm, self._Tb, self._Pt, self._Tt, 
                           self._Tc, self._Pc, self.Zc, self._Vc,
                           self._omega, self._dipole, self._similarity_variable,
                           self._iscyclic_aliphatic, self._eos, self.has_hydroxyl,
                           self._Hfus)
        self._locked_state = None
        if phase: self.at_state(phase)
        self._estimate_missing_properties(self.Psat)
        self._init_energies(self._Cn, self._Hvap, self._Psat, self._Hfus, self._Sfus,
                            self._Tm, self._Tb, self._eos, self._phase_ref)
        self._init_reactions(self._Hf, self._S0, self._LHV, self._HHV, 
                             self._combustion, self.atoms)
        if self._formula and self._Hf is not None: self.reset_combustion_data()
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True
        return self

    @property
    def charge(self):
        """Charge of chemical as described in the chemcial formula."""
        return charge_from_formula(self.formula)

    def copy(self, ID, CAS=None, **data):
        """
        Return a copy of the chemical with a new ID.
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Glucose = tmo.Chemical('Glucose')
        >>> Mannose = Glucose.copy('Mannose')
        >>> assert Mannose.MW == Glucose.MW
        
        """
        new = super().__new__(self.__class__)
        getfield = getattr
        setfield = setattr
        for field in self.__slots__: 
            value = getfield(self, field, None)
            setfield(new, field, copy_maybe(value))
        for field in _names[:-1]:
            setfield(new, field, None)
        new._ID = ID
        new._CAS = CAS or ID
        new._aliases = set()
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
        new._init_energies(new.Cn, new.Hvap, new.Psat, new.Hfus, new.Sfus, new.Tm,
                           new.Tb, new.eos, new.phase_ref)
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True
        new._label_handles()
        for i,j in data.items(): setfield(new, i , j)
        return new
    __copy__ = copy

    def _label_handles(self):
        handles = (self._Psat, self._Hvap, self._sigma, self._epsilon,
                   self._V, self._Cn, self._mu, self._kappa)
        isa = isinstance
        if add_ID and self.CAS != self.ID:
            label = f"{self.CAS} ({self.ID})"
        else:
            label = self.CAS
        for handle in handles:
            if isa(handle, PhaseHandle):
                for i, j in handle:
                    j.CASRN = label
            else:
                handle.CASRN = label

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
        997.024
        
        """
        return fn.V_to_rho(self._V(*args, **kwargs), self._MW)
    
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
        return self._Cn(*args, **kwargs) / self._MW
    
    def alpha(self, *args, **kwargs):
        """
        Return thermal diffusivity  given thermal condition [m^2/s].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.alpha('l', 298.15, 101325)
        1.425...e-07
        
        """
        return fn.alpha(self._kappa(*args, **kwargs), 
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
        9.154...e-07
        
        """
        return fn.mu_to_nu(self._mu(*args, **kwargs), 
                           self.rho(*args, **kwargs))
    
    def Pr(self, *args, **kwargs):
        """
        Return the Prandtl number given thermal condition [-].
        
        Examples
        --------
        >>> import thermosteam as tmo
        >>> Water = tmo.Chemical('Water', cache=True)
        >>> Water.Pr('l', 298.15, 101325)
        6.421
        
        """
        return fn.Pr(self.Cp(*args, **kwargs) * 1000,
                     self._kappa(*args, **kwargs), 
                     self._mu(*args, **kwargs))
    
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
        0.07168596252716256

        >>> Water.get_property('rho', 'g/cm3', 'l', 300., 101325) # Density
        0.9962

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
    def aliases(self):
        """set[str] User-defined aliases."""
        return self._aliases
    @aliases.setter
    def aliases(self, aliases):
        self._aliases = set([aliases]) if isinstance(aliases, str) else set(aliases)
    
    synonyms = aliases
    
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
            self._MW = compute_molecular_weight(self.atoms)
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
    
    ### Phase/model handles ###
    
    @property
    def mu(self): 
        """Dynamic viscosity [Pa*s]."""
        return self._mu
    @mu.setter
    def mu(self, value): 
        raise_helpful_handle_error('mu', self._mu)
    
    @property
    def kappa(self): 
        """Thermal conductivity [W/m/K]."""
        return self._kappa
    @kappa.setter
    def kappa(self, value): 
        raise_helpful_handle_error('kappa', self._kappa)
        
    @property
    def V(self): 
        """Molar volume [m^3/mol]."""
        return self._V
    @V.setter
    def V(self, value): 
        raise_helpful_handle_error('V', self._V)
        
    @property
    def Cn(self): 
        """Molar heat capacity [J/mol/K]."""
        return self._Cn
    @Cn.setter
    def Cn(self, value): 
        raise_helpful_handle_error('Cn', self._Cn)
        
    @property
    def Psat(self):
        """Vapor pressure [Pa]."""
        return self._Psat
    @Psat.setter
    def Psat(self, value): 
        raise_helpful_handle_error('Psat', self._Psat)
        
    @property
    def Hvap(self): 
        """Heat of vaporization [J/mol]."""
        return self._Hvap
    @Hvap.setter
    def Hvap(self, value): 
        raise_helpful_handle_error('Hvap', self._Hvap)
    
    @property
    def Hsub(self):
        """Heat of sublimation [J/mol]."""
        return self._Hsub
    @Hsub.setter
    def Hsub(self, value):
        raise_helpful_handle_error('Hsub', self._Hsub)
        
    @property
    def Psub(self):
        """Sublimation pressure [Pa]."""
        return self._Psub
    @Psub.setter
    def Psub(self, value):
        raise_helpful_handle_error('Psub', self._Psub)
        
    @property
    def sigma(self): 
        """Surface tension [N/m]."""
        return self._sigma
    @sigma.setter
    def sigma(self, value): 
        raise_helpful_handle_error('sigma', self._sigma)
        
    @property
    def epsilon(self): 
        """Relative permitivity [-]."""
        return self._epsilon
    @epsilon.setter
    def epsilon(self, value):
        raise_helpful_handle_error('epsilon', self._epsilon)
    
    @property
    def S(self): 
        """Entropy [J/mol]."""
        return self._S
    @S.setter
    def S(self, value): 
        raise_helpful_energy_functor_error('S', self._S)
    
    @property
    def H(self): 
        """Enthalpy [J/mol]."""
        return self._H
    @H.setter
    def H(self, value): 
        raise_helpful_energy_functor_error('H', self._H)
    
    @property
    def S_excess(self): 
        """Excess entropy [J/mol]."""
        return self._S_excess
    @S_excess.setter
    def S_excess(self, value): 
        raise_helpful_energy_functor_error('S_excess', self._S_excess)
    
    @property
    def H_excess(self): 
        """Excess enthalpy [J/mol]."""
        return self._H_excess
    @H_excess.setter
    def H_excess(self, value): 
        raise_helpful_energy_functor_error('H_excess', self._H_excess)
    
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
        reset_constant(self, 'Tc', float(Tc))
    
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
            try:
                P_at_Tr_sixtenths = Psat(0.6 * Tc)
            except:
                P_at_Tr_sixtenths = None
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
        return get_atoms(formula).copy() if formula else {}
    @atoms.setter
    def atoms(self, atoms):
        """dict[str: int] Atom-count pairs based on formula."""
        self.formula = atoms_to_Hill(atoms)
    
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
    
    ### Data solvers ###

    def Tsat(self, P, Tguess=None, Tmin=None, Tmax=None, *, check_validity=True):
        """Return the saturated temperature (in Kelvin) given the pressure (in Pascal)."""
        Tb = self._Tb
        Psat = self._Psat
        if check_validity and P >= self._Pc:
            raise ValueError(
                'pressure, P, cannot be above the supercritical pressure, Pc'
            )
        if not Psat: return None
        if not Tmin: Tmin = Psat.Tmin + 1.
        if not Tmax: Tmax = Psat.Tmax - 1.
        if Tb:
            if P == 101325: return Tb
            else: Tguess = Tb
        elif not Tguess:
            Tguess = (Tmin + Tmax)/2.0
        y0 = Psat(Tmin) - P
        y1 = Psat(Tmax) - P
        if y0 < 0. < y1:
            T = IQ_interpolation(lambda T: Psat(T) - P,
                                 Tmin, Tmax, y0, y1,
                                 Tguess, 1e-6, 1e-2, checkroot=False)
        else:
            T = aitken_secant(lambda T: Psat(T) - P,
                              Tmin, Tmax, 1e-6, 1e-2, checkroot=False)
        return T

    ### Reinitializers ###
    
    def reset(self, CAS, eos=PR, phase_ref=None,
              smiles=None, InChI=None, InChI_key=None,
              pubchemid=None, iupac_name=None, common_name=None,
              formula=None, MW=None, Tm=None,
              Tb=None, Tc=None, Pc=None, Vc=None, omega=None,
              Tt=None, Pt=None, Hf=None, S0=None, LHV=None, combustion=None,
              HHV=None, Hfus=None, dipole=None,
              similarity_variable=None, iscyclic_aliphatic=None, aliases=None,
              synonyms=None, *, metadata=None, phase=None):
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
                         formula, aliases, synonyms)
        self._init_groups(InChI_key)
        if CAS == '56-81-5': # TODO: Make this part of data
            self._Dortmund = DortmundGroupCounts({2: 2, 3: 1, 14: 2, 81: 1})
        atoms = self.atoms
        self._init_data(CAS, MW, Tm, Tb, Tc, Pc, Vc, omega, Pt, Tt, Hfus,
                        dipole, atoms, similarity_variable, iscyclic_aliphatic)
        self._init_eos(eos, self._Tc, self._Pc, self._omega)
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
        self._estimate_missing_properties(None)
        self._init_handles(CAS, self._MW, self._Tm, self._Tb, self._Pt, self._Tt, 
                           self._Tc, self._Pc, self.Zc, self._Vc,
                           self._omega, self._dipole, self._similarity_variable,
                           self._iscyclic_aliphatic, self._eos, self.has_hydroxyl,
                           self._Hfus)
        self._locked_state = None
        if phase: self.at_state(phase)
        self._estimate_missing_properties(self._Psat)
        self._init_energies(self._Cn, self._Hvap, self._Psat, self._Hfus, self._Sfus,
                            self._Tm, self._Tb, self._eos, phase_ref)
        self._init_reactions(Hf, S0, LHV, HHV, combustion, atoms)
        if self._formula and self._Hf is not None: self.reset_combustion_data()
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True

    def reset_combustion_data(self, method="Stoichiometry"):
        """Reset combustion data (LHV, HHV, and combution attributes)
        based on the molecular formula and the heat of formation."""
        cd = combustion_data(self.atoms, Hf=self._Hf, MW=self._MW, method=method)
        if not self._MW: self._MW = cd.MW
        self._LHV = - cd.LHV
        self._HHV = - cd.HHV
        self._combustion = cd.stoichiometry
    
    def reset_free_energies(self):
        """Reset the `H`, `S`, `H_excess`, and `S_excess` functors."""
        if not self._eos:
            self._eos = create_eos(PR, self._Tc, self._Pc, self._omega)
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = False
        self._init_energies(self._Cn, self._Hvap, self._Psat, self._Hfus, self._Sfus,
                            self._Tm, self._Tb, self._eos, self._phase_ref)
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True

    ### Initializers ###
    
    def _init_names(self, CAS, smiles, InChI, InChI_key,
                    pubchemid, iupac_name, common_name, formula, aliases, synonyms):
        self._CAS = CAS
        self._smiles = smiles
        self._InChI = InChI
        self._InChI_key = InChI_key
        self._pubchemid = pubchemid
        self._iupac_name = iupac_name
        self._common_name = common_name
        self.aliases = aliases or ()
        if synonyms: 
            if isinstance(synonyms, str):
                self.aliases.add(synonyms)
            else:
                self.aliases.update(synonyms)
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
            Hvap_298K = self.Hvap(298.15)
            Hf = heat_of_formation(self._CAS, self._phase_ref,
                                   Hvap_298K, self.Hfus) 
        if S0 is None:
            Hvap_298K = self.Hvap(298.15) if Hvap_298K is None else Hvap_298K
            Svap_298K = None if Hvap_298K is None else Hvap_298K / 298.15
            S0 = absolute_entropy_of_formation(self._CAS, self._phase_ref,
                                               Svap_298K, self.Sfus) or 0.
        self._Hf = Hf
        self.S0 = S0
        atoms = atoms or self.atoms
        if not all([LHV, HHV, combustion]) and atoms and Hf:
            cd = combustion_data(atoms, Hf=self._Hf, MW=self._MW, missing_handling='Ash')
            LHV = - cd.LHV
            HHV = - cd.HHV
            combustion = cd.stoichiometry
        self._LHV = LHV
        self._HHV = HHV
        self._combustion = combustion

    def _init_eos(self, eos, Tc, Pc, omega):
        self._eos = create_eos(eos, Tc, Pc, omega)

    def _init_handles(self, CAS, MW, Tm, Tb, Pt, Tt, Tc, Pc, Zc, Vc, omega,
                      dipole, similarity_variable, iscyclic_aliphatic, eos,
                      has_hydroxyl, Hfus):
        # Vapor pressure
        self._Psat = Psat = VaporPressure(CASRN=CAS, Tb=Tb, Tc=Tc, Pc=Pc, omega=omega)
        
        # Volume
        Vl = VolumeLiquid(MW=MW, Tb=Tb, Tc=Tc, Pc=Pc, Vc=Vc, 
                          Zc=Zc, omega=omega, dipole=dipole, Psat=Psat, 
                          CASRN=CAS, eos=[eos], has_hydroxyl=has_hydroxyl)
        Vml_Tm = Vl(Tm, 101325.) if Tm else None
        Vs = VolumeSolid(CAS, MW, Tt, Vml_Tm)
        Vg = VolumeGas(MW=MW, Tc=Tc, Pc=Pc, omega=omega, dipole=dipole,
                       eos=[eos], CASRN=CAS)
        self._V = V = PhaseTPHandle('V', Vs, Vl, Vg)
        if CAS in sugar_solid_densities: Vs.add_method(sugar_solid_densities[CAS])
        
        # Sensible heats
        Cns = HeatCapacitySolid(CASRN=CAS, similarity_variable=similarity_variable, MW=MW)
        Cng = HeatCapacityGas(CASRN=CAS, MW=MW, similarity_variable=similarity_variable, 
                              iscyclic_aliphatic=iscyclic_aliphatic)
        Cnl = HeatCapacityLiquid(CASRN=CAS, Tc=Tc, omega=omega, MW=MW, 
                                 similarity_variable=similarity_variable, Cpgm=Cng)
        self._Cn = Cn = PhaseTHandle('Cn', Cns, Cnl, Cng)
        
        # Latent heats
        Zl = CompressibilityFactor(Vl)
        Zg = CompressibilityFactor(Vg)
        self._Hvap = Hvap = EnthalpyVaporization(
            CASRN=CAS, Tb=Tb, Tc=Tc, Pc=Pc, omega=omega,
            Zl=Zl, Zg=Zg, similarity_variable=similarity_variable, Psat=Psat
        )
        self._Hsub = Hsub = EnthalpySublimation(
            CASRN=CAS, Tm=Tm, Tt=Tt, Cpg=Cng, Cps=Cns, Hvap=Hvap
        )
        self._Psub = SublimationPressure(
            CASRN=CAS, Tt=Tt, Pt=Pt, Hsub_t=Hsub(Tm) if Hsub and Tm else None,
        )
        
        # Viscosity
        mul = ViscosityLiquid(CASRN=CAS, MW=MW, Tm=Tm, Tc=Tc, Pc=Pc, Vc=Vc, 
                              omega=omega, Psat=Psat, Vml=V.l)
        mug = ViscosityGas(CASRN=CAS, MW=MW, Tc=Tc, Pc=Pc, Zc=Zc, dipole=dipole,
                           Vmg=Vg)
        self._mu = mu = PhaseTPHandle('mu', mul, mul, mug)
        
        # Conductivity
        kappal = ThermalConductivityLiquid(CASRN=CAS, MW=MW, Tm=Tm, Tb=Tb, 
                                           Tc=Tc, Pc=Pc, omega=omega, Hfus=Hfus)
        kappag = ThermalConductivityGas(CASRN=CAS, MW=MW, Tb=Tb, Tc=Tc, Pc=Pc, 
                                        Vc=Vc, Zc=Zc, omega=omega, dipole=dipole, 
                                        Vmg=V.g, Cpgm=Cn.g, mug=mu.g)
        self._kappa = PhaseTPHandle('kappa', kappal, kappal, kappag)
        
        # Surface tension
        Hvap_Tb = self._Hvap(Tb) if Tb else None
        self._sigma = SurfaceTension(CASRN=CAS, MW=MW, Tb=Tb, Tc=Tc, Pc=Pc, 
                                     Vc=Vc, Zc=Zc, omega=omega, 
                                     Stiel_Polar=self.Stiel_Polar,
                                     Hvap_Tb=Hvap_Tb, Vml=Vl, Cpl=Cnl, 
                                     extrapolation='constant')
        
        # Other
        self._epsilon = PermittivityLiquid(CAS)
        self._label_handles()
        

    def _estimate_missing_properties(self, Psat):
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
        if not Pt and Tt and Psat:
            self._Pt = Pt = Psat(Tt)
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
            try:
                self._Tb = Tb = self.Tsat(101325, check_validity=False)
            except:
                self._Tb = Tb = None
            
        omega = self._omega
        if not omega and Pc and Tc:
            if Psat: 
                P_at_Tr_seventenths = Psat(0.7 * Tc)
                if P_at_Tr_seventenths: omega = acentric_factor_definition(P_at_Tr_seventenths, Pc)
            elif not omega and Tb:
                omega = acentric_factor_LK(Tb, Tc, Pc)
            self._omega = omega

    def _init_energies(self, Cn, Hvap, Psat, Hfus, Sfus, Tm, Tb, eos, 
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
            has_Cns = has_Cnl = has_Cng = False
            if single_phase == 'l':
                has_Cnl = True
            elif single_phase == 's':
                has_Cns = True
            elif single_phase == 'g':
                has_Cng = True    
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
                Hvap_Tb = Hvap(Tb) if Tb else None
                Svap_Tb = Hvap_Tb / Tb if Hvap_Tb else None
            else:
                Hvap_Tb = Svap_Tb = None
            
            # Enthalpy and entropy integrals
            if phase_ref != 'l' and has_Cnl and (Tm and Tb):
                H_int_Tm_to_Tb_l = Cn_l.T_dependent_property_integral(Tm, Tb)
                S_int_Tm_to_Tb_l = Cn_l.T_dependent_property_integral_over_T(Tm, Tb)
            else:
                H_int_Tm_to_Tb_l = S_int_Tm_to_Tb_l = None
            if phase_ref == 's' and has_Cns and Tm:
                H_int_T_ref_to_Tm_s = Cn_s.T_dependent_property_integral(T_ref, Tm)
                S_int_T_ref_to_Tm_s = Cn_s.T_dependent_property_integral_over_T(T_ref, Tm)
            else:
                H_int_T_ref_to_Tm_s = S_int_T_ref_to_Tm_s = None
            if phase_ref == 'g' and has_Cng and Tb:
                H_int_Tb_to_T_ref_g = Cn_g.T_dependent_property_integral(Tb, T_ref)
                S_int_Tb_to_T_ref_g = Cn_g.T_dependent_property_integral_over_T(Tb, T_ref)
            else:
                H_int_Tb_to_T_ref_g = S_int_Tb_to_T_ref_g = None
            if phase_ref == 'l':
                if has_Cnl:
                    if Tb:
                        H_int_T_ref_to_Tb_l = Cn_l.T_dependent_property_integral(T_ref, Tb)
                        S_int_T_ref_to_Tb_l = Cn_l.T_dependent_property_integral_over_T(T_ref, Tb)
                    else:
                        H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
                    if Tm:
                        try:
                            H_int_Tm_to_T_ref_l = Cn_l.T_dependent_property_integral(Tm, T_ref)
                            S_int_Tm_to_T_ref_l = Cn_l.T_dependent_property_integral_over_T(Tm, T_ref)
                        except:
                            H_int_Tm_to_T_ref_l = S_int_Tm_to_T_ref_l = None
                    else:
                        H_int_Tm_to_T_ref_l = S_int_Tm_to_T_ref_l = None
                else:
                    H_int_Tm_to_T_ref_l = S_int_Tm_to_T_ref_l = \
                    H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
    
            # Excess data
            if isinstance(eos, IG):
                H_dep_ref_g = S_dep_ref_g = H_dep_ref_l = S_dep_ref_l = \
                H_dep_T_ref_Pb = S_dep_T_ref_Pb = H_dep_Tb_Pb_g = H_dep_Tb_P_ref_g = \
                S_dep_Tb_P_ref_g = S_dep_Tb_Pb_g = 0
            else:
                if phase_ref == 'g':
                    try:
                        eos_phase_ref = eos.to_TP(T_ref, P_ref)
                        H_dep_ref_g = eos_phase_ref.H_dep_g
                        S_dep_ref_g = eos_phase_ref.S_dep_g
                    except:
                        H_dep_ref_g = S_dep_ref_g = 0.
                elif phase_ref == 'l':
                    try:
                        eos_T_ref_Pb = eos.to_TP(T_ref, P_ref)
                        H_dep_ref_l = H_dep_T_ref_Pb = eos_T_ref_Pb.H_dep_l
                        S_dep_ref_l = S_dep_T_ref_Pb = eos_T_ref_Pb.S_dep_l
                    except:
                        H_dep_ref_l = H_dep_T_ref_Pb = 0.
                        S_dep_ref_l = S_dep_T_ref_Pb = 0.
                if Tb:
                    try:
                        eos_Tb_P_ref = eos.to_TP(Tb, P_ref)
                        H_dep_Tb_Pb_g = H_dep_Tb_P_ref_g = eos_Tb_P_ref.H_dep_g
                        S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = eos_Tb_P_ref.S_dep_g
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
                ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g)
                gdata = (eos, H_dep_ref_g)
                self._H_excess = ExcessEnthalpyRefGas((), ldata, gdata)
                ldata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g)
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
        ['Dortmund',
         'Hfus',
         'Hvap',
         'PSRK',
         'Pc',
         'Psat',
         'Pt',
         'Sfus',
         'Tb',
         'Tc',
         'Tm',
         'Tt',
         'UNIFAC',
         'Vc',
         'dipole',
         'iscyclic_aliphatic',
         'omega',
         'similarity_variable']
        
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
            self._sigma.add_method(0.072055)
        if 'mu' in properties:
            mu = self._mu
            if hasfield(mu, 'l'):
                mu.l.add_method(0.00091272)
            elif not mu:
                mu.add_method(0.00091272)
        if 'V' in properties:
            V = self._V
            V_default = fn.rho_to_V(1000, MW)
            if hasfield(V, 'l'):
                V.l.add_method(V_default)
            elif not V:
                V.add_method(V_default)
        if 'kappa' in properties:
            kappa = self._kappa
            if hasfield(kappa, 'l'):
                kappa.l.add_method(0.5942)
            if not kappa:
                kappa.add_method(0.5942)
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
                        self._HHV = - cd.HHV
                    if 'LHV' in properties:
                        self._LHV = - cd.LHV
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
            self._epsilon.add_method(0)
        if 'phase_ref' in properties:
            self._phase_ref = 'l'
        if 'eos' in properties:
            self._eos = create_eos(PR, self._Tc, self._Pc, self._omega)
        if 'Cn' in properties:
            MW = self._MW
            Cn = self._Cn
            phase_ref = self._phase_ref
            getfield = getattr
            single_phase = self._locked_state
            if single_phase:
                Cn.add_method(4.18*MW)
                Cn_phase = Cn
            else:
                Cn_phase = getfield(Cn, phase_ref)
                Cn_phase.add_method(4.18*MW)
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
        ['Cn',
         'Dortmund',
         'H',
         'HHV',
         'H_excess',
         'Hf',
         'Hfus',
         'Hvap',
         'LHV',
         'MW',
         'PSRK',
         'Pc',
         'Psat',
         'Pt',
         'S',
         'S_excess',
         'Sfus',
         'Tb',
         'Tc',
         'Tm',
         'Tt',
         'UNIFAC',
         'Vc',
         'combustion',
         'dipole',
         'epsilon',
         'iscyclic_aliphatic',
         'kappa',
         'mu',
         'omega',
         'sigma',
         'similarity_variable']
        
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
            names = ['_' + i for i in names]
        else:
            missing_handles = self.get_missing_properties(_model_and_phase_handles)
            other_missing_handles = other.get_missing_properties(_model_and_phase_handles)
            names = set(missing_handles).difference(other_missing_handles)
        getfield = getattr
        setfield = object.__setattr__
        isa = isinstance
        phase = self._locked_state
        other_phase = other._locked_state
        for key in names:
            handle = getfield(self, key)
            other_handle = getfield(other, key)
            if isa(handle, TDependentProperty):
                if isa(other_handle, TDependentProperty):
                    pass
                elif isa(other_handle, PhaseHandle):
                    other_handle = getfield(other_handle, phase)
                else:
                    raise RuntimeError(f"unexpected type '{type(other_handle).__name__}"
                                       f"for attribute '{key}'")
                new_handle = other_handle.copy()
                new_handle.CASRN = handle.CASRN
                setfield(self, key, new_handle)
            elif isa(handle, PhaseHandle):
                if isa(other_handle, TDependentProperty):
                    old_handle = getfield(handle, other_phase)
                    new_handle = other_handle.copy()
                    new_handle.CASRN = old_handle.CASRN
                    setfield(handle, other_phase, new_handle)
                elif isa(other_handle, PhaseHandle):
                    for i, obj in handle:
                        new_handle = getfield(other_handle, i).copy()
                        new_handle.CASRN = obj.CASRN
                        setfield(handle, i, new_handle)
        if {'_Cn', '_Hvap'}.intersection(names): self.reset_free_energies()
    
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
    
