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

from .identifiers import CAS_from_any, pubchem_db
from .vapor_pressure import VaporPressure
from .phase_change import Tb, Tm, Hfus, Hsub, EnthalpyVaporization
from .critical import Tc, Pc, Vc
from .acentric import omega, StielPolar
from .triple import Tt, Pt
from .volume import Volume
from .heat_capacity import HeatCapacity
from .reaction import Hf
from .combustion import Hcombustion
from .elements import similarity_variable, simple_formula_parser
from .eos import GCEOS_DUMMY, PR
from .viscosity import Viscosity
from .thermal_conductivity import ThermalConductivity
from .free_energy import EnthalpyRefSolid, EnthalpyRefLiquid, EnthalpyRefGas, \
                         EntropyRefSolid, EntropyRefLiquid, EntropyRefGas, \
                         ExcessEnthalpyRefSolid, ExcessEnthalpyRefLiquid, ExcessEnthalpyRefGas, \
                         ExcessEntropyRefSolid, ExcessEntropyRefLiquid, ExcessEntropyRefGas
from ..base import PhaseProperty, ChemicalPhaseTProperty, ChemicalPhaseTPProperty, \
                   display_asfunctor, thermo_model, ThermoModelHandle, \
                   ConstantThermoModel, TDependentModelHandle
from ..base.units_of_measure import units_of_measure
from .dipole import dipole_moment as dipole
from ..functional import Z #, isentropic_exponent, Joule_Thomson, B_from_Z, isobaric_expansion
from .permittivity import Permittivity
from .interface import SurfaceTension
from ..equilibrium.unifac_data import DDBST_UNIFAC_assignments, \
                                      DDBST_MODIFIED_UNIFAC_assignments, \
                                      DDBST_PSRK_assignments
# from .solubility import SolubilityParameter
# from .safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen
# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

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
    fullID = f"{typeheader} {chemical.ID} (phase_ref={repr(chemical.phase_ref)})"
    phase, T, P = chemical._phaseTP
    state = []
    if phase:
        state.append(f"phase={repr(phase)}")
    if T:
        state.append(f"T={T} K")
    if P:
        state.append(f"P={P} Pa")
    state = ' at ' + ', '.join(state) if state else ""
    if pretty and (T and P):
        fullID += "\n" + len(typeheader) * " " 
    return fullID + state


# %% Initialize EOS
                         
def create_eos(eos, Tc, Pc, omega):
    try: return eos(T=298.15, P=101325., Tc=Tc, Pc=Pc, omega=omega)
    except: return GCEOS_DUMMY(T=298.15, P=101325.)


# %% Chemical fields

_names = ('CAS', 'InChI', 'InChI_key',
         'common_name', 'iupac_name',
         'pubchemid', 'smiles')

_groups = ('Dortmund', 'UNIFAC', 'PSRK')

_thermo = ('S_excess', 'H_excess', 'mu', 'k', 'V', 'S', 'H', 'Cp',
           'Psat', 'Hvap', 'sigma', 'epsilon')
    
_phase_properties = ('k', 'V', 'Cp', 'mu')

_free_energies = ('S_excess', 'H_excess', 'S', 'H')

_liquid_only_properties = ('sigma', 'epsilon')

_equilibrium_properties = ('Psat', 'Hvap')

_data = ('MW', 'Tm', 'Tb', 'Tt', 'Tc', 'Pt', 'Pc', 'Vc', 'Zc',
         'Hf', 'Hc', 'Hfus', 'Hsub', 'rhoc', 'omega', 'dipole',
         'StielPolar', 'similarity_variable', 'iscyclic_aliphatic')

_chemical_fields = {'\n[Names]  ': _names,
                    '\n[Groups] ': _groups,
                    '\n[Thermo] ': _thermo,
                    '\n[Data]   ': _data}


# %% Chemical

class Chemical:
    __slots__ = ('ID', 'eos', 'eos_1atm', '_phaseTP', '_phase_ref') \
                + _names + _groups + _thermo + _data
    T_ref = 298.15; P_ref = 101325.; H_ref = 0.; S_ref = 0.
    
    def __init__(self, ID, *, eos=PR, CAS=None, cache=True):
        CAS = CAS or CAS_from_any(ID) or ID
        info = pubchem_db.search_CAS(CAS)
        self.ID = ID
        self._phaseTP = (None, None, None)
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
        self._init_energies(self.Cp, self.Hvap, self.Psat, self.Hfus,
                            self.Tm, self.Tb, self.eos, self.eos_1atm)

    def copy(self):
        new = self.__new__(self.__class__)
        getfield = getattr
        setfield = setattr
        for field in self.__slots__: setfield(new, field, getfield(self, field))
        return new
    
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
        self.rhoc = 1./self.Vc if self.Vc else None

        # Triple point
        self.Pt = Pt(CAS)
        self.Tt = Tt(CAS)

        # Energy
        self.Hfus = Hfus(CASRN=CAS, MW=MW)
        self.Hsub = Hsub(CASRN=CAS)

        # Chemistry
        self.Hf = Hf(CAS)
        self.Hc = Hcombustion(atoms=atoms, Hf=self.Hf)
        
        # Critical Point
        self.StielPolar = StielPolar(CAS, Tc, Pc, omega)
        
        # Other
        self.dipole = dipole(CAS)
        self.similarity_variable = similarity_variable(atoms, MW)
        self.iscyclic_aliphatic = None

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
        Cp = ChemicalPhaseTProperty()
        sdata = (CAS, similarity_variable, MW)
        ldata = (CAS, Tb, Tc, omega, MW, similarity_variable, Cp)
        gdata = (CAS, MW, similarity_variable, iscyclic_aliphatic)
        self.Cp = Cp = HeatCapacity(sdata, ldata, gdata, Cp)
        
        # Heat of vaporization
        data = (CAS, Tb, Tc, Pc, omega, similarity_variable, Psat, V)
        self.Hvap = Hvap = EnthalpyVaporization(data)
        
        # Viscosity
        ldata = (CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, V.l)
        gdata = (CAS, MW, Tc, Pc, Zc, dipole)
        self.mu = mu = Viscosity(None, ldata, gdata)
        
        # Conductivity
        ldata = (CAS, MW, Tm, Tb, Tc, Pc, omega, Hfus)
        gdata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, V.g, Cp.g, mu.g)
        self.k = ThermalConductivity(None, ldata, gdata)
        
        # Surface tension
        Hvap_Tb = Hvap(Tb) if Tb else None
        Cpl_Tb = Cp.l(Tb) if (Cp.l and Tb) else None
        rhol = 1/V.l(Tb, 101325.) * MW if (V.l and MW) else None
        data = (CAS, MW, Tb, Tc, Pc, Vc, Zc,
                omega, StielPolar, Hvap_Tb, rhol, Cpl_Tb)
        self.sigma = SurfaceTension(data)
        
        # Other
        self.epsilon = Permittivity((CAS, V.l,))
        # self.delta = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def _init_energies(self, Cp, Hvap, Psat, Hfus, Tm, Tb, eos, eos_1atm, phase_ref=None):        
        # Reference
        P_ref = self.P_ref
        T_ref = self.T_ref
        H_ref = self.H_ref
        S_ref = self.S_ref
        Sfus = Hfus / Tm if Hfus and Tm else None
        
        if hasattr(Cp, 's'):
            Cp_s = Cp.s
            Cp_l = Cp.l
            Cp_g = Cp.g
            has_Cps = bool(Cp_s)
            has_Cpl = bool(Cp_l)
            has_Cpg = bool(Cp_g)
        elif Cp and phase_ref:
            has_Cps = phase_ref == 's'
            has_Cpl = phase_ref == 'l'
            has_Cpg = phase_ref == 'g'
        else:
            has_Cps = has_Cpl = has_Cpg = False
        
        if any((has_Cps, has_Cpl, has_Cpg)):
            if not phase_ref:
                if Tm and T_ref <= Tm:
                    self._phase_ref = phase_ref = 's'
                elif Psat and P_ref <= Psat(T_ref):
                    self._phase_ref = phase_ref = 'g'
                else:
                    self._phase_ref = phase_ref = 'l'
            
            Hvap_Tb = Hvap(Tb) if Tb else None
            Svap_Tb = Hvap_Tb / Tb if Tb else None
            
            # Enthalpy and entropy integrals
            if phase_ref != 'l' and has_Cpl and (Tm and Tb):
                H_int_Tm_to_Tb_l = Cp_l.integrate_by_T(Tm, Tb)
                S_int_Tm_to_Tb_l = Cp_l.integrate_by_T_over_T(Tm, Tb)
            else:
                H_int_Tm_to_Tb_l = S_int_Tm_to_Tb_l = None
            if phase_ref == 's' and has_Cps and Tm:
                H_int_T_ref_to_Tm_s = Cp_s.integrate_by_T(T_ref, Tm)
                S_int_T_ref_to_Tm_s = Cp_s.integrate_by_T_over_T(T_ref, Tm)
            else:
                H_int_T_ref_to_Tm_s = S_int_T_ref_to_Tm_s = None
            if phase_ref == 'g' and has_Cpg and Tb:
                H_int_Tb_to_T_ref_g = Cp_g.integrate_by_T(Tb, T_ref)
                S_int_Tb_to_T_ref_g = Cp_g.integrate_by_T_over_T(Tb, T_ref)
            else:
                H_int_Tb_to_T_ref_g = S_int_Tb_to_T_ref_g = None
            if phase_ref == 'l':
                if has_Cpl:
                    if Tb:
                        H_int_T_ref_to_Tb_l = Cp_l.integrate_by_T(T_ref, Tb)
                        S_int_T_ref_to_Tb_l = Cp_l.integrate_by_T_over_T(T_ref, Tb)
                    else:
                        H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
                    if Tm:
                        H_int_Tm_to_T_ref_l = Cp_l.integrate_by_T(Tm, T_ref)
                        S_int_Tm_to_T_ref_l = Cp_l.integrate_by_T_over_T(Tm, T_ref)
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
                sdata = (Cp_s, T_ref, H_ref)
                ldata = (Cp_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref)
                gdata = (Cp_g, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref)
                self.H = EnthalpyRefSolid(sdata, ldata, gdata)
                sdata = (Cp_s, T_ref, S_ref)
                ldata = (Cp_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S_ref)
                gdata = (Cp_g, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S_ref)
                self.S = EntropyRefSolid(sdata, ldata, gdata)
            elif phase_ref == 'l':
                sdata = (Cp_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref)
                ldata = (Cp_l, T_ref, H_ref)
                gdata = (Cp_g, H_int_T_ref_to_Tb_l, Hvap_Tb, T_ref, H_ref)
                self.H = EnthalpyRefLiquid(sdata, ldata, gdata)
                sdata = (Cp_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S_ref)
                ldata = (Cp_l, T_ref, S_ref)
                gdata = (Cp_g, S_int_T_ref_to_Tb_l, Svap_Tb, T_ref, P_ref, S_ref)
                self.S = EntropyRefLiquid(sdata, ldata, gdata)
            elif phase_ref == 'g':
                sdata = (Cp_s, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref)
                ldata = (Cp_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref)
                gdata = (Cp_g, T_ref, H_ref)
                self.H = EnthalpyRefGas(sdata, ldata, gdata)
                sdata = (Cp_s, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S_ref)
                ldata = (Cp_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S_ref)
                gdata = (Cp_g, T_ref, P_ref, S_ref)
                self.S = EntropyRefGas(sdata, ldata, gdata)
            
            # Excess energies
            if phase_ref == 's':
                self.H_excess = ExcessEnthalpyRefSolid((), (), ())
                self.S_excess = ExcessEntropyRefSolid((), (), ())
            elif phase_ref == 'l':
                gdata = (eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g)
                self.H_excess = ExcessEnthalpyRefLiquid((), (), gdata)
                gdata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_1atm)
                self.S_excess = ExcessEntropyRefLiquid((), (), gdata)
            elif phase_ref == 'g':
                ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_1atm)
                gdata = (eos, H_dep_ref_g)
                self.H_excess = ExcessEnthalpyRefGas((), ldata, gdata)
                ldata = (eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g)
                gdata = (eos, S_dep_ref_g)
                self.S_excess = ExcessEntropyRefGas((), ldata, gdata)
        else:
            self.H = self.S = self.S_excess = self.H_excess = None

    def to_phase(self, phase):
        if any(self._phaseTP):
            raise ValueError(f"state already selected for {repr(self)}")            
        getfield = getattr
        setfield = setattr
        for field in _phase_properties:
            phase_property = getfield(self, field)
            model_handle = getfield(phase_property, phase)
            setfield(self, field, model_handle)
        for field in _free_energies:
            phase_property = getfield(self, field)
            functor = getfield(phase_property, phase)
            setfield(self, field, functor)
        self._phaseTP = (phase, None, None)

    def to_TP(self, T, P):
        if any(self._phaseTP):
            raise ValueError(f"state already selected for {repr(self)}")    
        getfield = getattr
        phases = ('s', 'l', 'g')
        for field in _phase_properties:
            phase_property = getfield(self, field)
            for phase in phases:
                model_handle = getfield(phase_property, phase)
                model_handle.to_TP(T, P)
        for field in _liquid_only_properties:
            model_handle = getfield(self, field) 
            model_handle.to_TP(T, P)
        self._phaseTP = (None, T, P)

    def to_phaseTP(self, phase, T, P):
        if any(self._phaseTP):
            raise ValueError(f"state already selected for {repr(self)}")            
        getfield = getattr
        setfield = setattr
        for field in _phase_properties:
            phase_property = getfield(self, field)
            model_handle = getfield(phase_property, phase)
            model_handle.to_TP(T, P)
            setfield(self, field, model_handle)
        for field in _liquid_only_properties:
            model_handle = getfield(self, field) 
            model_handle.to_TP(T, P)
        for field in _free_energies:
            phase_property = getfield(self, field)
            functor = getfield(phase_property, phase)
            setfield(self, field, functor)
        self._phaseTP = (phase, T, P)
    
    def default(self, slots=None):
        if not slots:
            slots = self.missing(slots)   
        has = hasattr
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
            if has(mu, 'l'):
                mu.l.model(0.00091272)
            if not mu:
                mu.model(0.00091272)
        if 'k' in slots:
            k = self.k
            if has(k, 'l'):
                k.l.model(0.5942)
            if not k:
                k.model(0.5942)
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
        if 'Cp' in slots:
            Cp = self.Cp
            phase_ref = self.phase_ref
            getfield = getattr
            has_phase_ref = has(Cp, phase_ref)
            if has_phase_ref:
                Cp_phase = getfield(Cp, phase_ref)
                Cp_phase.model(4.18*MW, var='Cp')
            else:
                Cp.model(4.18*MW, var='Cp')
                Cp_phase = Cp
            self._init_energies(Cp, self.Hvap, self.Psat, self.Hfus, self.Tm,
                                self.Tb, self.eos, self.eos_1atm, self.phase_ref)
            if not has_phase_ref:
                self.H = getfield(self.H, phase_ref)
                self.S = getfield(self.S, phase_ref)
                self.H_excess = getfield(self.H_excess, phase_ref)
                self.S_excess = getfield(self.S_excess, phase_ref)
        missing = set(slots)
        missing.difference_update({'MW', 'CAS', 'Cp', 'Hf', 'sigma',
                                   'mu', 'k', 'Hc', 'epsilon', 'H',
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
        self = object.__new__(cls)
        setfield = setattr
        self.eos = self.eos_1atm = self._phase_ref = None
        for i in _names: setfield(self, i, None)
        for i in _groups: setfield(self, i, None)
        for i in _data: setfield(self, i, None)
        for i in _free_energies: setfield(self, i, None)
        for i in ('k', 'mu', 'V'):
            setfield(self, i, ChemicalPhaseTPProperty())
        self.Cp = ChemicalPhaseTProperty()
        for i in ('sigma', 'epsilon', 'Psat', 'Hvap'):
            setfield(self, i, TDependentModelHandle())
        self._phaseTP = (None, None, None)
        self.ID = ID
        return self
    
    @classmethod
    def new(cls, ID, *sources, slots=None, default=True):
        try: self = cls(ID)
        except: self = cls.blank(ID)
        self.fill(*sources, slots=slots, default=default)
        return self
    
    def phase(self, T=298.15, P=101325.):
        if self.Tm and T <= self.Tm: return 's'
        if self.Psat and P <= self.Psat(T): return 'g'
        else: return 'l'
    
    @property
    def phase_ref(self):
        return self._phase_ref
    
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
                            units = units_of_measure.get(field, "")
                            if units: line += ' ' + units
                section.append(line)
            if section:
                info += header + ("\n" + 9*" ").join(section)
        print(info)
        
    _ipython_display_ = show
    
    def __str__(self):
        return self.ID
    
    def __repr__(self):
        return f'<{chemical_identity(self)}>'
    


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
    