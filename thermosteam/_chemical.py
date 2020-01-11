# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, 2017 Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

__all__ = ('Chemical',)

from scipy.optimize import newton
from .utils import copy_maybe, cucumber
from .functors.identifiers import CAS_from_any, pubchem_db
from .functors.vapor_pressure import VaporPressure
from .functors.phase_change import Tb, Tm, Hfus, Hsub, EnthalpyVaporization
from .functors.critical import Tc, Pc, Vc
from .functors.acentric import omega, StielPolar
from .functors.triple import Tt, Pt
from .functors.volume import Volume
from .functors.heat_capacity import HeatCapacity
from .functors.reaction import Hf
from .functors.combustion import Hcombustion
from .functors.elements import similarity_variable, simple_formula_parser
from .functors.eos import GCEOS_DUMMY, PR
from .functors.viscosity import Viscosity
from .functors.thermal_conductivity import ThermalConductivity
from .functors.free_energy import \
    EnthalpyRefSolid, EnthalpyRefLiquid, EnthalpyRefGas, \
    EntropyRefSolid, EntropyRefLiquid, EntropyRefGas, \
    ExcessEnthalpyRefSolid, ExcessEnthalpyRefLiquid, ExcessEnthalpyRefGas, \
    ExcessEntropyRefSolid, ExcessEntropyRefLiquid, ExcessEntropyRefGas
from .base import PhaseProperty, ChemicalPhaseTProperty, ChemicalPhaseTPProperty, \
                  display_asfunctor, TDependentModelHandle
from .base.units_of_measure import chemical_units_of_measure
from .functors.dipole import dipole_moment as dipole
from .functional import Z, rho_to_V #, isentropic_exponent, Joule_Thomson, B_from_Z, isobaric_expansion
from .functors.permittivity import Permittivity
from .functors.interface import SurfaceTension
from .equilibrium.unifac_data import \
    DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments
# from .solubility import SolubilityParameter
# from .safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen
# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

# %% Utilities
                                      
class LockedState:
    __slots__ = ('_phase', '_T', '_P')
    def __init__(self, phase=None, T=None, P=None):
        self._phase = phase
        self._T = T
        self._P = P
    
    @property
    def phase(self):
        return self._phase    
    @property
    def T(self):
        return self._T
    @property
    def P(self):
        return self._P
    
    def __bool__(self):
        return any(self)
    
    def __iter__(self):
        yield self._phase
        yield self._T
        yield self._P
    
    def __repr__(self):
        return f"{type(self).__name__}(phase={self.phase}, T={self.T}, P={self.P})"


# %% Filling missing properties

def fill_from_dict(chemical, dict, slots=None):
    if not slots: slots = chemical.missing(slots)
    missing = []
    setfield = setattr
    for i in slots :
        if i in dict:
            setfield(chemical, i, dict[i])
        else:
            missing.append(i)
    return missing

def fill_from_obj(chemical, obj, slots=None):
    if not slots: slots = chemical.missing(slots)
    missing = []
    setfield = setattr
    getfield = getattr
    for i in slots :
        attr = getfield(obj, i)
        if attr:
            setfield(obj, i, attr)
        else:
            missing.append(i)
    return missing

def fill(chemical, other, slots=None):
    fill = fill_from_dict if isinstance(other, dict) else fill_from_obj
    return fill(chemical, other, slots)


# %% Representation
    
def chemical_identity(chemical, pretty=False):
    typeheader = f"{type(chemical).__name__}:"
    full_ID = f"{typeheader} {chemical.ID} (phase_ref={repr(chemical.phase_ref)})"
    phase, T, P = chemical._locked_state
    state = []
    if phase:
        state.append(f"phase={repr(phase)}")
    if T:
        state.append(f"T={T} K")
    if P:
        state.append(f"P={P} Pa")
    state = ' at ' + ', '.join(state) if state else ""
    if pretty and (T and P):
        full_ID += "\n" + len(typeheader) * " " 
    return full_ID + state


# %% Initialize EOS
                         
def create_eos(eos, Tc, Pc, omega):
    try: return eos(T=298.15, P=101325., Tc=Tc, Pc=Pc, omega=omega)
    except: return GCEOS_DUMMY(T=298.15, P=101325.)


# %% Chemical fields

_names = ('CAS', 'InChI', 'InChI_key',
         'common_name', 'iupac_name',
         'pubchemid', 'smiles')

_groups = ('Dortmund', 'UNIFAC', 'PSRK')

_thermo = ('S_excess', 'H_excess', 'mu', 'kappa', 'V', 'S', 'H', 'Cn',
           'Psat', 'Hvap', 'sigma', 'epsilon')
    
_phase_properties = ('kappa', 'V', 'Cn', 'mu')

_free_energies = ('S_excess', 'H_excess', 'S', 'H')

_liquid_only_properties = ('sigma', 'epsilon')

_equilibrium_properties = ('Psat', 'Hvap')

_data = ('MW', 'Tm', 'Tb', 'Tt', 'Tc', 'Pt', 'Pc', 'Vc', 'Zc',
         'Hf', 'Hc', 'Hfus', 'Hsub', 'omega', 'dipole',
         'StielPolar', 'similarity_variable', 'iscyclic_aliphatic')

_chemical_fields = {'\n[Names]  ': _names,
                    '\n[Groups] ': _groups,
                    '\n[Thermo] ': _thermo,
                    '\n[Data]   ': _data}


# %% Chemical
@cucumber # Just means you can pickle it
class Chemical:
    """Creates a Chemical object which contains basic information such as 
    molecular weight and the structure of the species, as well as thermodynamic
    and transport properties as a function of temperature and pressure.
    
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
    
    """
    __slots__ = ('_ID', 'eos', 'eos_1atm', '_locked_state', '_phase_ref',
                 *_names, *_groups, *_thermo, *_data)
    T_ref = 298.15; P_ref = 101325.; H_ref = 0.; S_ref = 0.
    _cache = {}
    
    def __new__(cls, ID, *, eos=PR):
        cache = cls._cache
        CAS = CAS_from_any(ID)
        if CAS in cache:
            self = cache[CAS]
        else:
            info = pubchem_db.search_CAS(CAS)
            self = super().__new__(cls)
            self._ID = ID
            self._locked_state = LockedState()
            self._init_names(CAS, info.smiles, info.InChI, info.InChI_key, 
                             info.pubchemid, info.iupac_name, info.common_name)
            self._init_groups(info.InChI_key)
            if CAS == '56-81-5': # TODO: Make this part of data
                self.Dortmund = {2: 2, 3: 1, 14: 2, 81: 1}
            self._init_data(CAS, info.MW, atoms=simple_formula_parser(info.formula))
            self._init_eos(eos, self.Tc, self.Pc, self.omega)
            self._init_properties(CAS, self.MW, self.Tm, self.Tb, self.Tc,
                                  self.Pc, self.Zc, self.Vc, self.Hfus,
                                  self.omega, self.dipole, self.similarity_variable,
                                  self.iscyclic_aliphatic, self.eos)
            self._init_energies(self.Cn, self.Hvap, self.Psat, self.Hfus,
                                self.Tm, self.Tb, self.eos, self.eos_1atm)
            cache[CAS] = self
        return self

    @property
    def ID(self):
        return self._ID

    def Tsat(self, P, T_min=200):
        return newton(lambda T: P - self.Psat(T if T > T_min else T_min), self.Tb)

    def copy(self, ID):
        new = super().__new__(self.__class__)
        getfield = getattr
        setfield = setattr
        for field in self.__slots__: 
            value = getfield(self, field)
            setfield(new, field, copy_maybe(value))
        new._init_energies(new.Cn, new.Hvap, new.Psat, new.Hfus, new.Tm,
                           new.Tb, new.eos, new.eos_1atm, new.phase_ref,
                           new._locked_state.phase)
        return new
    __copy__ = copy
    
    def _init_names(self, CAS, smiles, InChI, InChI_key,
                    pubchemid, iupac_name, common_name):
        self.CAS = CAS
        self.smiles = smiles
        self.InChI = InChI
        self.InChI_key = InChI_key
        self.pubchemid = pubchemid
        self.iupac_name = iupac_name
        self.common_name = common_name
        
    def _init_groups(self, InChI_key):
        if InChI_key in DDBST_UNIFAC_assignments:
            self.UNIFAC = DDBST_UNIFAC_assignments[InChI_key]
        else:
            self.UNIFAC = None
        if InChI_key in DDBST_MODIFIED_UNIFAC_assignments:
            self.Dortmund = DDBST_MODIFIED_UNIFAC_assignments[InChI_key]
        else:
            self.Dortmund = None
        if InChI_key in DDBST_PSRK_assignments:
            self.PSRK = DDBST_PSRK_assignments[InChI_key]
        else:
            self.PSRK = None

    def _init_data(self, CAS, MW, atoms):
        self.MW = MW
        self.Tm = Tm(CAS)
        self.Tb = Tb(CAS)

        # Critical Point
        self.Tc = Tc(CAS)
        self.Pc = Pc(CAS)
        self.Vc = Vc(CAS)
        self.omega = omega(CAS)
        self.Zc = Z(self.Tc, self.Pc, self.Vc) if all((self.Tc, self.Pc, self.Vc)) else None

        # Triple point
        self.Pt = Pt(CAS)
        self.Tt = Tt(CAS)

        # Energy
        self.Hfus = Hfus(CASRN=CAS, MW=MW)
        self.Hsub = Hsub(CASRN=CAS)

        # Chemistry
        self.Hf = Hf(CAS) or 0.
        self.Hc = Hcombustion(atoms=atoms, Hf=self.Hf) or 0.
        
        # Critical Point
        self.StielPolar = StielPolar(CAS, Tc, Pc, omega)
        
        # Other
        self.dipole = dipole(CAS)
        self.similarity_variable = similarity_variable(atoms, MW)
        self.iscyclic_aliphatic = False

    def _init_eos(self, eos, Tc, Pc, omega):
        self.eos = create_eos(eos, Tc, Pc, omega)
        self.eos_1atm = self.eos.to_TP(298.15, 101325)

    def _init_properties(self, CAS, MW, Tm, Tb, Tc, Pc, Zc, Vc, Hfus, omega,
                         dipole, similarity_variable, iscyclic_aliphatic, eos):
        # Vapor pressure
        self.Psat = Psat = VaporPressure((CAS, Tb, Tc, Pc, omega))
        
        # Volume
        sdata = (CAS,)
        ldata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, Psat, eos)
        gdata = (CAS, Tc, Pc, omega, eos)
        self.V = V = Volume(sdata, ldata, gdata)
        
        # Heat capacity
        Cn = ChemicalPhaseTProperty(var='Cn')
        sdata = (CAS, similarity_variable, MW)
        ldata = (CAS, Tb, Tc, omega, MW, similarity_variable, Cn)
        gdata = (CAS, MW, similarity_variable, iscyclic_aliphatic)
        self.Cn = Cn = HeatCapacity(sdata, ldata, gdata, Cn)
        
        # Heat of vaporization
        data = (CAS, Tb, Tc, Pc, omega, similarity_variable, Psat, V)
        self.Hvap = Hvap = EnthalpyVaporization(data)
        
        # Viscosity
        ldata = (CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, V.l)
        gdata = (CAS, MW, Tc, Pc, Zc, dipole)
        self.mu = mu = Viscosity(None, ldata, gdata)
        
        # Conductivity
        ldata = (CAS, MW, Tm, Tb, Tc, Pc, omega, Hfus)
        gdata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, V.g, Cn.g, mu.g)
        self.kappa = ThermalConductivity(None, ldata, gdata)
        
        # Surface tension
        Hvap_Tb = Hvap(Tb) if Tb else None
        Cnl_Tb = Cn.l(Tb) if (Cn.l and Tb) else None
        rhol = 1/V.l(Tb, 101325.) * MW if (V.l and MW) else None
        data = (CAS, MW, Tb, Tc, Pc, Vc, Zc,
                omega, StielPolar, Hvap_Tb, rhol, Cnl_Tb)
        self.sigma = SurfaceTension(data)
        
        # Other
        self.epsilon = Permittivity((CAS, V.l,))
        # self.delta = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def _init_energies(self, Cn, Hvap, Psat, Hfus, Tm, Tb, eos, eos_1atm,
                       phase_ref=None, single_phase=False):        
        # Reference
        P_ref = self.P_ref
        T_ref = self.T_ref
        H_ref = self.H_ref
        S_ref = self.S_ref
        Sfus = Hfus / Tm if Hfus and Tm else None
        
        if isinstance(Cn, PhaseProperty):
            Cn_s = Cn.s
            Cn_l = Cn.l
            Cn_g = Cn.g
            has_Cns = bool(Cn_s)
            has_Cnl = bool(Cn_l)
            has_Cng = bool(Cn_g)
        elif Cn and single_phase:
            has_Cns = single_phase == 's'
            has_Cnl = single_phase == 'l'
            has_Cng = single_phase == 'g'
            Cn_s = Cn_l = Cn_g = Cn
        else:
            has_Cns = has_Cnl = has_Cng = False
        
        if any((has_Cns, has_Cnl, has_Cng)):
            if not phase_ref:
                if Tm and T_ref <= Tm:
                    self._phase_ref = phase_ref = 's'
                elif Tb and T_ref >= Tb:
                    self._phase_ref = phase_ref = 'g'
                else:
                    self._phase_ref = phase_ref = 'l'

            Hvap_Tb = Hvap(Tb) if Tb else None
            Svap_Tb = Hvap_Tb / Tb if Tb else None
            
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
                    eos_Tb = eos.to_TP(Tb, 101325)
                    eos_Tb_P_ref = eos.to_TP(Tb, P_ref)
                    H_dep_Tb_Pb_g = eos_Tb.H_dep_g
                    H_dep_Tb_P_ref_g = eos_Tb_P_ref.H_dep_g
                    S_dep_Tb_P_ref_g = eos_Tb_P_ref.S_dep_g
                    S_dep_Tb_Pb_g = eos_Tb.S_dep_g
                else:
                    S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = H_dep_Tb_P_ref_g = \
                    H_dep_Tb_Pb_g = 0.
            
            # Enthalpy and Entropy
            if phase_ref == 's':
                sdata = (Cn_s, T_ref, H_ref)
                ldata = (Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref)
                gdata = (Cn_g, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref)
                self.H = EnthalpyRefSolid(sdata, ldata, gdata)
                sdata = (Cn_s, T_ref, S_ref)
                ldata = (Cn_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S_ref)
                gdata = (Cn_g, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S_ref)
                self.S = EntropyRefSolid(sdata, ldata, gdata)
            elif phase_ref == 'l':
                sdata = (Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref)
                ldata = (Cn_l, T_ref, H_ref)
                gdata = (Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, T_ref, H_ref)
                self.H = EnthalpyRefLiquid(sdata, ldata, gdata)
                sdata = (Cn_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S_ref)
                ldata = (Cn_l, T_ref, S_ref)
                gdata = (Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb, T_ref, P_ref, S_ref)
                self.S = EntropyRefLiquid(sdata, ldata, gdata)
            elif phase_ref == 'g':
                sdata = (Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref)
                ldata = (Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref)
                gdata = (Cn_g, T_ref, H_ref)
                self.H = EnthalpyRefGas(sdata, ldata, gdata)
                sdata = (Cn_s, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S_ref)
                ldata = (Cn_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S_ref)
                gdata = (Cn_g, T_ref, P_ref, S_ref)
                self.S = EntropyRefGas(sdata, ldata, gdata)
            
            # Excess energies
            if phase_ref == 's':
                self.H_excess = ExcessEnthalpyRefSolid((), (), ())
                self.S_excess = ExcessEntropyRefSolid((), (), ())
            elif phase_ref == 'l':
                gdata = (eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g)
                self.H_excess = ExcessEnthalpyRefLiquid((), (), gdata)
                gdata = (eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g)
                self.S_excess = ExcessEntropyRefLiquid((), (), gdata)
            elif phase_ref == 'g':
                ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_1atm)
                gdata = (eos, H_dep_ref_g)
                self.H_excess = ExcessEnthalpyRefGas((), ldata, gdata)
                ldata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_1atm)
                gdata = (eos, S_dep_ref_g)
                self.S_excess = ExcessEntropyRefGas((), ldata, gdata)
        else:
            self.H = self.S = self.S_excess = self.H_excess = None
            
        if single_phase:
            getfield = getattr
            self.H = getfield(self.H, single_phase)
            self.S = getfield(self.S, single_phase)
            self.H_excess = getfield(self.H_excess, single_phase)
            self.S_excess = getfield(self.S_excess, single_phase)

    def default(self, slots=None):
        if not slots:
            slots = self.missing(slots)   
        hasfield = hasattr
        # Default to Water property values
        if 'CAS' in slots:
            self.CAS = self.ID
        if 'MW' in slots:
            self.MW = MW = 1
        else:
            MW = self.MW
        if 'sigma' in slots:
            self.sigma.model(0.072055)
        if 'mu' in slots:
            mu = self.mu
            if hasfield(mu, 'l'):
                mu.l.model(0.00091272)
            elif not mu:
                mu.model(0.00091272)
        if 'V' in slots:
            V = self.V
            V_default = rho_to_V(1050, MW)
            if hasfield(V, 'l'):
                V.l.model(V_default)
            elif not V:
                V.model(V_default)
        if 'kappa' in slots:
            kappa = self.kappa
            if hasfield(kappa, 'l'):
                kappa.l.model(0.5942)
            if not kappa:
                kappa.model(0.5942)
        if 'Hc' in slots:
            self.Hc = 0
        if 'Hf' in slots:
            self.Hf = 0
        if 'epsilon' in slots:
            self.epsilon.model(0)
        if '_phase_ref' in slots:
            self._phase_ref = 'l'
        if 'eos' in slots:
            self.eos = GCEOS_DUMMY(T=298.15, P=101325.)
            self.eos_1atm = self.eos.to_TP(298.15, 101325)
        if 'Cn' in slots:
            Cn = self.Cn
            phase_ref = self.phase_ref
            getfield = getattr
            single_phase = isinstance(Cn, TDependentModelHandle)
            if single_phase:
                Cn.model(4.18*MW, var='Cn')
                Cn_phase = Cn
            else:
                Cn_phase = getfield(Cn, phase_ref)
                Cn_phase.model(4.18*MW, var='Cn')
            self._init_energies(Cn, self.Hvap, self.Psat, self.Hfus, self.Tm,
                                self.Tb, self.eos, self.eos_1atm, self.phase_ref,
                                single_phase and phase_ref)
        missing = set(slots)
        missing.difference_update({'MW', 'CAS', 'Cn', 'Hf', 'sigma',
                                   'mu', 'kappa', 'Hc', 'epsilon', 'H',
                                   'S', 'H_excess', 'S_excess', '_phase_ref'})
        return missing
    
    def missing(self, slots=None):
        getfield = getattr
        return [i for i in (slots or self.__slots__) if not getfield(self, i)]
    
    def fill(self, *sources, slots=None, default=True):
        missing = slots if slots else self.missing(slots)
        for source in sources:
            missing = fill(source, missing)
        if default:
            missing = self.default(missing)
        return missing
    
    @classmethod
    def blank(cls, ID):
        self = super().__new__(cls)
        setfield = setattr
        self.eos = self.eos_1atm = self._phase_ref = None
        for i in _names: setfield(self, i, None)
        for i in _groups: setfield(self, i, None)
        for i in _data: setfield(self, i, None)
        for i in _free_energies: setfield(self, i, None)
        for i in ('kappa', 'mu', 'V'):
            setfield(self, i, ChemicalPhaseTPProperty())
        self.Cn = ChemicalPhaseTProperty()
        for i in ('sigma', 'epsilon', 'Psat', 'Hvap'):
            setfield(self, i, TDependentModelHandle())
        self._locked_state = LockedState()
        self._ID = ID
        return self
    
    @classmethod
    def new(cls, ID, *sources, slots=None, default=True):
        try: self = cls(ID)
        except: self = cls.blank(ID)
        self.fill(*sources, slots=slots, default=default)
        return self
    
    def phase(self, T=298.15, P=101325.):
        if self._locked_state.phase: return self._locked_state.phase
        if self.Tm and T <= self.Tm: return 's'
        if self.Psat and P <= self.Psat(T): return 'g'
        else: return 'l'
    
    @property
    def phase_ref(self):
        return self._phase_ref
    
    @property
    def locked_state(self):
        return self._locked_state
    
    def at_state(self, ID, phase=None, T=None, P=None):
        new = self.copy(ID)
        if new.locked_state:
            raise TypeError(f"{self}'s state is already locked")    
        elif T and P:
            if phase:
                lock_locked_state(new, phase, T, P)
            else:
                lock_TP(new, T, P)                
        elif phase:
            lock_phase(new, phase)
        else:
            raise ValueError("must pass a either a phase, T and P, or both to lock state")
        new._locked_state.__init__(phase, T, P)
        return new
    
    def show(self):
        getfield = getattr
        info = chemical_identity(self, pretty=True)
        for header, fields in _chemical_fields.items():
            section = []
            for field in fields:
                value = getfield(self, field)
                if value is None: continue
                else:
                    if callable(value):
                        line = f"{display_asfunctor(value, name=field, var=field, show_var=False)}"
                    else:
                        try:
                            line = f"{field}: {value:.5g}"
                        except:
                            value = str(value)
                            line = f"{field}: {value}"
                            if len(line) > 27: line = line[:27] + '...'
                        else:
                            units = chemical_units_of_measure.get(field, "")
                            if units: line += f' {units}'
                section.append(line)
            if section:
                info += header + ("\n" + 9*" ").join(section)
        print(info)
        
    _ipython_display_ = show
    
    def __str__(self):
        return self.ID
    
    def __repr__(self):
        return f"Chemical('{self}')"
    
    
def lock_TP(chemical, T, P):
    getfield = getattr
    setfield = setattr
    phases = ('s', 'l', 'g')
    for field in _phase_properties:
        phase_property = getfield(chemical, field)
        for phase in phases:
            model_handle = getfield(phase_property, phase)
            try: value = model_handle.lock_TP(T, P)
            except: value = None
            setfield(phase_property, phase, value)
    for field in _liquid_only_properties:
        model_handle = getfield(chemical, field) 
        value = model_handle.lock_TP(T, P)
        setfield(chemical, field, value)

def lock_phase(chemical, phase):
    getfield = getattr
    setfield = setattr
    for field in _phase_properties:
        phase_property = getfield(chemical, field)
        model_handle = getfield(phase_property, phase)
        setfield(chemical, field, model_handle)
    for field in _free_energies:
        phase_property = getfield(chemical, field)
        functor = getfield(phase_property, phase)
        setfield(chemical, field, functor)

def lock_locked_state(chemical, phase, T, P):
    getfield = getattr
    setfield = setattr
    for field in _phase_properties:
        phase_property = getfield(chemical, field)
        model_handle = getfield(phase_property, phase)
        try: value = model_handle.lock_TP(T, P)
        except: value = None
        setfield(chemical, field, value)
    for field in _liquid_only_properties:
        model_handle = getfield(chemical, field) 
        try: value = model_handle.lock_TP(T, P)
        except: value = None
        setfield(chemical, field, value) 
    for field in _free_energies:
        phase_property = getfield(chemical, field)
        functor = getfield(phase_property, phase)
        setfield(chemical, field, functor)

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
    