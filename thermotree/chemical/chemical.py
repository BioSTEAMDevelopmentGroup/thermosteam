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
from .unifac import DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments
from .viscosity import Viscosity
from .thermal_conductivity import ThermalConductivity
from .free_energy import EnthalpyRefSolid, EnthalpyRefLiquid, EnthalpyRefGas, \
                         EntropyRefSolid, EntropyRefLiquid, EntropyRefGas, \
                         ExcessEnthalpyRefSolid, ExcessEnthalpyRefLiquid, ExcessEnthalpyRefGas, \
                         ExcessEntropyRefSolid, ExcessEntropyRefLiquid, ExcessEntropyRefGas
from ..base import PhaseProperty, ChemicalPhaseTProperty, ChemicalPhaseTPProperty, display_asfunctor
from ..base.units_of_measure import units_of_measure
from .dipole import dipole_moment as dipole
from .utils import Z, R#, isentropic_exponent, Joule_Thomson, B_from_Z, isobaric_expansion
from .permittivity import Permittivity
from .interface import SurfaceTension
# from .solubility import SolubilityParameter
# from .safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen



# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

# %% Utilities
                         
def create_eos(eos, Tc, Pc, omega):
    try: return eos(T=298.15, P=101325., Tc=Tc, Pc=Pc, omega=omega)
    except: return GCEOS_DUMMY(T=298.15, P=101325.)


# %% Chemical fields

names = ('CAS', 'InChI', 'InChI_key',
         'common_name', 'iupac_name',
         'pubchemid', 'smiles')

groups = ('UNIFAC_Dortmund', 'UNIFAC', 'PSRK')

thermo = ('S_excess', 'H_excess', 'k', 'V', 'S', 'H', 'Cp',
          'mu', 'Psat', 'Hvap', 'sigma', 'epsilon')

_optional_data = ('S_excess', 'H_excess', 'k', 'V', 'Cp',
                       'mu', 'sigma', 'epsilon')

_optional_single_phase_nondata = ('S', 'H')

_optional_single_phase = ('S_excess', 'H_excess', 'k', 'V', 'S', 'H', 'Cp',
                          'mu', 'sigma')

_optional_properties = ('Psat', 'Hvap')

data = ('MW', 'Tm', 'Tb', 'Tt', 'Tc', 'Pt', 'Pc', 'Vc', 'Zc',
        'Hf', 'Hc', 'Hfus', 'Hsub', 'rhoc', 'omega', 'dipole',
        'StielPolar', 'similarity_variable', 'iscyclic_aliphatic')

chemical_fields = {'\n[Names]  ': names,
                   '\n[Groups] ': groups,
                   '\n[Thermo] ': thermo,
                   '\n[Data]   ': data}

def _full_chemical_identity(chemical, pretty=False):
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


# %% Chemical

class Chemical:
    __slots__ = ('ID', 'eos', 'eos_T_101325', '_phaseTP', '_phase_ref') \
                + names + groups + thermo + data
    T_ref = 298.15; P_ref = 101325.; H_ref = 0.; S_ref = 0.
    _cached = {}
    
    def __new__(cls, ID, *, eos=PR, CAS=None):
        CAS = CAS or CAS_from_any(ID) or ID
        if CAS in cls._cached:
            self = cls._cached[CAS]
            if self.ID != ID:
                self = self.copy()
                self.ID = ID
        else:
            self = object.__new__(cls)
            self._phaseTP = (None, None, None)
            info = pubchem_db.search_CAS(CAS)
            self._init_groups(info.InChI_key)
            self._init_names(CAS, info.smiles, info.InChI, info.InChI_key, 
                             info.pubchemid, info.iupac_name, info.common_name)
            atoms = simple_formula_parser(info.formula)
            self._init_data(CAS, info.MW, atoms)
            self.eos = create_eos(eos, self.Tc, self.Pc, self.omega)
            self.eos_T_101325 = self.eos.to_TP(298.15, 101325)
            self._init_thermo(CAS, self.eos, self.eos_T_101325)
            self.ID = ID
        cls._cached[CAS] = self
        return self

    def copy(self):
        new = object.__new__(self.__class__)
        getfield = getattr
        setfield = setattr
        for field in self.__slots__: setfield(new, field, getfield(self, field))
        return new

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
            self.UNIFAC_Dortmund = DDBST_MODIFIED_UNIFAC_assignments[InChI_key]
        else:
            self.UNIFAC_Dortmund = None
        if InChI_key in DDBST_PSRK_assignments:
            self.PSRK = DDBST_PSRK_assignments[InChI_key]
        else:
            self.PSRK = None

    def _init_thermo(self, CAS, eos, eos_T_101325, phase_ref=None):
        # Reference
        P_ref = self.P_ref
        T_ref = self.T_ref
        H_ref = self.H_ref
        S_ref = self.S_ref
        
        # data
        MW = self.MW
        Tm = self.Tm
        Tb = self.Tb
        Tc = self.Tc
        Pc = self.Pc
        Zc = self.Zc
        Vc = self.Vc
        Hfus = self.Hfus
        omega = self.omega
        dipole = self.dipole
        similarity_variable = self.similarity_variable
        iscyclic_aliphatic = self.iscyclic_aliphatic
        Sfus = Hfus / Tm if Hfus and Tm else None
        
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
        
        has_Cps = bool(Cp.s)
        has_Cpl = bool(Cp.l)
        has_Cpg = bool(Cp.g)
        has_Cp = any((has_Cps, has_Cpl, has_Cpg))
        if has_Cp:
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
                H_int_Tm_to_Tb_l = Cp.l.integrate_by_T(Tm, Tb)
                S_int_Tm_to_Tb_l = Cp.l.integrate_by_T_over_T(Tm, Tb)
            else:
                H_int_Tm_to_Tb_l = S_int_Tm_to_Tb_l = None
            if phase_ref == 's' and has_Cps and Tm:
                H_int_T_ref_to_Tm_s = Cp.s.integrate_by_T(T_ref, Tm)
                S_int_T_ref_to_Tm_s = Cp.s.integrate_by_T_over_T(T_ref, Tm)
            else:
                H_int_T_ref_to_Tm_s = S_int_T_ref_to_Tm_s = None
            if phase_ref == 'g' and has_Cpg and Tb:
                H_int_Tb_to_T_ref_g = Cp.g.integrate_by_T(Tb, T_ref)
                S_int_Tb_to_T_ref_g = Cp.g.integrate_by_T_over_T(Tb, T_ref)
            else:
                H_int_Tb_to_T_ref_g = S_int_Tb_to_T_ref_g = None
            if phase_ref == 'l':
                if has_Cpl:
                    if Tb:
                        H_int_T_ref_to_Tb_l = Cp.l.integrate_by_T(T_ref, Tb)
                        S_int_T_ref_to_Tb_l = Cp.l.integrate_by_T_over_T(T_ref, Tb)
                    else:
                        H_int_T_ref_to_Tb_l = S_int_T_ref_to_Tb_l = None
                    if Tm:
                        H_int_Tm_to_T_ref_l = Cp.l.integrate_by_T(Tm, T_ref)
                        S_int_Tm_to_T_ref_l = Cp.l.integrate_by_T_over_T(Tm, T_ref)
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
                    H_dep_T_ref_Pb = eos.to_TP(T_ref, 101325).H_dep_l
                    S_dep_T_ref_Pb = eos.to_TP(T_ref, 101325).S_dep_l
                if Tb:
                    eos_Tb = eos.to_TP(Tb, 101325)
                    H_dep_Tb_Pb_g = eos_Tb.H_dep_g
                    H_dep_Tb_P_ref_g = eos.to_TP(Tb, P_ref).H_dep_g
                    S_dep_Tb_P_ref_g = eos.to_TP(Tb, P_ref).S_dep_g
                    S_dep_Tb_Pb_g = eos_Tb.S_dep_g
                else:
                    S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = H_dep_Tb_P_ref_g = \
                    H_dep_Tb_Pb_g = 0.
            
            # Enthalpy and Entropy
            if phase_ref == 's':
                sdata = (Cp, T_ref, H_ref)
                ldata = (Cp, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref)
                gdata = (Cp, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref)
                self.H = EnthalpyRefSolid(sdata, ldata, gdata)
                sdata = (Cp, T_ref, S_ref)
                ldata = (Cp, S_int_T_ref_to_Tm_s, Sfus, Tm, S_ref)
                gdata = (Cp, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S_ref)
                self.S = EntropyRefSolid(sdata, ldata, gdata)
            elif phase_ref == 'l':
                sdata = (Cp, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref)
                ldata = (Cp, T_ref, H_ref)
                gdata = (Cp, H_int_T_ref_to_Tb_l, Hvap_Tb, T_ref, H_ref)
                self.H = EnthalpyRefLiquid(sdata, ldata, gdata)
                sdata = (Cp, S_int_Tm_to_T_ref_l, Sfus, Tm, S_ref)
                ldata = (Cp, T_ref, S_ref)
                gdata = (Cp, S_int_T_ref_to_Tb_l, Svap_Tb, T_ref, P_ref, S_ref)
                self.S = EntropyRefLiquid(sdata, ldata, gdata)
            elif phase_ref == 'g':
                sdata = (Cp, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref)
                ldata = (Cp, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref)
                gdata = (Cp, T_ref, H_ref)
                self.H = EnthalpyRefGas(sdata, ldata, gdata)
                sdata = (Cp, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S_ref)
                ldata = (Cp, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S_ref)
                gdata = (Cp, T_ref, P_ref, S_ref)
                self.S = EntropyRefGas(sdata, ldata, gdata)
            
            # Excess energies
            if phase_ref == 's':
                self.H_excess = ExcessEnthalpyRefSolid((), (), ())
                self.S_excess = ExcessEntropyRefSolid((), (), ())
            elif phase_ref == 'l':
                gdata = (eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g)
                self.H_excess = ExcessEnthalpyRefLiquid((), (), gdata)
                gdata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_T_101325)
                self.S_excess = ExcessEntropyRefLiquid((), (), gdata)
            elif phase_ref == 'g':
                ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_T_101325)
                gdata = (eos, H_dep_ref_g)
                self.H_excess = ExcessEnthalpyRefGas((), ldata, gdata)
                ldata = (eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g)
                gdata = (eos, S_dep_ref_g)
                self.S_excess = ExcessEntropyRefGas((), ldata, gdata)
        else:
            self.H = self.S = self.S_excess = self.H_excess = None
        
        # # Conductivity
        ldata = (CAS, MW, Tm, Tb, Tc, Pc, omega, Hfus)
        gdata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, V.g, self.Cv, mu.g)
        self.k = ThermalConductivity(None, ldata, gdata)
        
        # # Surface tension
        Cpl_Tb = Cp.l(Tb) if (Cp.l and Tb) else None
        rhol = 1/V.l(Tb, 101325.) * MW if (V.l and MW) else None
        data = (CAS, MW, Tb, Tc, Pc, Vc, Zc,
                omega, StielPolar, Hvap_Tb, rhol, Cpl_Tb)
        self.sigma = SurfaceTension(data)
        
        # # Other
        self.epsilon = Permittivity((CAS, V.l,))
        # self.solubility_parameter = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def to_phase(self, phase):
        current_phase, T, P = self._phaseTP
        if current_phase == phase:
            return self
        elif current_phase:
            raise ValueError(f"state not available for {repr(self)}")
        new = self.copy()        
        getfield = getattr
        setfield = setattr
        isa = isinstance
        for field in _optional_single_phase:
            obj = getfield(new, field)
            if isa(obj, PhaseProperty): setfield(new, field, getfield(obj, phase))
        new._phaseTP = (phase, T, P)
        return new

    def to_TP(self, TP):
        phase, *current_TP = self._phaseTP
        if current_TP == TP:
            return self
        elif any(current_TP):
            raise ValueError(f"state not available for {repr(self)}")
        T, P = TP
        new = self.copy()
        getfield = getattr
        setfield = setattr
        isa = isinstance
        iscallable = callable
        for field in _optional_data:
            obj = getfield(new, field)
            if isa(obj, ChemicalPhaseTPProperty): 
                for phase in ('s', 'l', 'g'):
                    setfield(obj, phase, obj(phase, T, P))
            elif isa (obj, ChemicalPhaseTProperty):
                for phase in ('s', 'l', 'g'):
                    setfield(obj, phase, obj(phase, T))
            elif iscallable(obj):
                try:
                    value = obj(T, P)
                except:
                    value = obj(T)
                setfield(new, field, value)
        new._phaseTP = (phase, T, P)
        return new

    def to_phaseTP(self, phaseTP):
        if self._phaseTP == phaseTP: return self
        for i, j in zip(self._phaseTP, phaseTP):
            if i and i != j:
                raise ValueError(f"state not available for {repr(self)}")            
        phase, T, P = phaseTP
        new = self.copy()
        getfield = getattr
        setfield = setattr
        isa = isinstance
        iscallable = callable
        for field in _optional_data:
            obj = getfield(new, field)
            if isa(obj, ChemicalPhaseTPProperty):
                value = obj(phase, T, P)
            elif isa (obj, ChemicalPhaseTProperty):
                value = obj(phase, T)
            elif iscallable(obj):
                try:
                    value = obj(T, P)
                except:
                    value = obj(T)
            else: continue
            setfield(new, field, value)
        for field in _optional_single_phase_nondata:
            obj = getfield(new, field)
            if isa(obj, PhaseProperty): setfield(new, field, getfield(obj, phase))
        new._phaseTP = phaseTP
        return new

    def Cv(self, T):
        return self.Cp.g(T) - R
    
    def fill(self, properties=None, like=None, fallback=None):
        getfield = getattr
        setfield = setattr
        has_properties = bool(properties)
        for key in self.__slots__:
            if getfield(self, key): continue
            elif has_properties and key in properties:
                field = properties[key]
            elif like:
                field = getfield(like, key)
                if not field and fallback: 
                    field = getfield(fallback, key)
            else:
                field = None
            setfield(self, key, field)
        return self

    @classmethod
    def build(cls, ID, *, properties=None, like=None, fallback=None):
        self = object.__new__(cls)
        self.ID = ID
        self.fill(properties, like, fallback)
        return self
    
    def phase(self, T, P):
        if self.Tm and T <= self.Tm: return 's'
        if self.Psat and P <= self.Psat(T): return 'g'
        else: return 'l'
    
    @property
    def phase_ref(self):
        return self._phase_ref
    
    def show(self):
        getfield = getattr
        info = _full_chemical_identity(self, pretty=True)
        for header, fields in chemical_fields.items():
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
        return f'<{type(self).__name__}: {_full_chemical_identity(self)}>'
    


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
    