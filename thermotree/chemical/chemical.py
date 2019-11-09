ch# -*- coding: utf-8 -*-
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

__all__ = ('Chemical', 'ChemicalThermoMethods', 'ChemicalData',
           'ChemicalSubgroups', 'ChemicalNames')

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
from ..base import ChemicalPhaseTProperty, display_asfunctor
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


# %% Chemical descriptor

class ChemicalDescriptor:
    __slots__ = ()
    
    @classmethod
    def new(cls):
        self = cls.__new__(cls)
        setfield = setattr
        for slot in cls.__slots__: setfield(self, slot, None)
        return self
    
    def fill(self, properties=None, like=None, fallback=None):
        getfield = getattr
        setfield = setattr
        has_properties = bool(properties)
        for key in self.__slots__:
            if getfield(self, key):
                continue
            elif has_properties and key in properties:
                field = properties[key]
            elif like:
                field = getfield(like, key)
                if not field and fallback: 
                    field = getfield(fallback, key)
            setfield(self, key, field)
        return self
    
    def __repr__(self):
        return f"<{type(self).__name__}: {', '.join(self.__slots__)}>"
    
    def show(self):
        attr = getattr
        info = f"{type(self).__name__}:"
        for key in self.__slots__:
            value = attr(self, key)
            if value is None:
                info += f"\n {key}: {value}"
                continue
            else:
                try:
                    info += f"\n {key}: {value:.5g}"
                except:
                    info += f"\n {key}: {value}"    
                else:
                    units = units_of_measure.get(key, "")
                    if units: info += ' ' + units
        print(info)
    
    _ipython_display_ = show
    


# %% Chemical descriptors

class ChemicalData(ChemicalDescriptor):
    __slots__ = ('MW', 'Tm', 'Tb', 'Tt', 'Tc', 'Pt', 'Pc', 'Vc', 'Zc',
                 'Hf', 'Hc', 'Hfus', 'Hsub', 'rhoc', 'omega', 'dipole',
                 'StielPolar', 'similarity_variable', 'iscyclic_aliphatic')
    
    def __init__(self, CAS, MW, atoms):
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


class ChemicalSubgroups(ChemicalDescriptor):
    __slots__ = ('UNIFAC_Dortmund', 'UNIFAC', 'PSRK')
    
    def __init__(self, InChI_key):
        if InChI_key in DDBST_UNIFAC_assignments:
            self.UNIFAC = DDBST_UNIFAC_assignments[InChI_key]
        if InChI_key in DDBST_MODIFIED_UNIFAC_assignments:
            self.UNIFAC_Dortmund = DDBST_MODIFIED_UNIFAC_assignments[InChI_key]
        if InChI_key in DDBST_PSRK_assignments:
            self.PSRK = DDBST_PSRK_assignments[InChI_key]


class ChemicalNames(ChemicalDescriptor):
    __slots__ = ('CAS', 'InChI', 'InChI_key',
                 'common_name', 'iupac_name',
                 'pubchemid', 'smiles')
    
    def __init__(self, CAS, smiles, InChI, InChI_key,
                 pubchemid, iupac_name, common_name):
        self.CAS = CAS
        self.smiles = smiles
        self.InChI = InChI
        self.InChI_key = InChI_key
        self.pubchemid = pubchemid
        self.iupac_name = iupac_name
        self.common_name = common_name
    

class ChemicalThermoMethods(ChemicalDescriptor):
    __slots__ = ('epsilon', 'sigma', 'Hvap', 'Psat',
                 'mu', 'Cp', 'H', 'S', 'V', 'k',
                 'H_excess', 'S_excess')

    def __init__(self, CAS, data, eos, eos_T_101325,
                 T_ref=298.15, P_ref=101325., H_ref=0, S_ref=0):
        # Constants
        MW = data.MW
        Tm = data.Tm
        Tb = data.Tb
        Tc = data.Tc
        Pc = data.Pc
        Zc = data.Zc
        Vc = data.Vc
        Hfus = data.Hfus
        Sfus = Hfus / Tm
        omega = data.omega
        dipole = data.dipole
        similarity_variable = data.similarity_variable
        iscyclic_aliphatic = data.iscyclic_aliphatic
        
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
        
        if Cp:
            Cps_int = Cp.s.integrate_by_T
            Cpl_int = Cp.l.integrate_by_T
            Cpg_int = Cp.g.integrate_by_T
            Cps_int_T = Cp.s.integrate_by_T_over_T
            Cpl_int_T = Cp.l.integrate_by_T_over_T
            Cpg_int_T = Cp.g.integrate_by_T_over_T
            
            if Tm and T_ref <= Tm:
                phase_ref = 's'
            elif P_ref <= Psat(T_ref):
                phase_ref = 'g'
            else:
                phase_ref = 'l'
            
            Hvap_Tb = Hvap(Tb) if Tb else None
            Svap_Tb = Hvap_Tb / Tb if Tb else None
            
            # Enthalpy integrals
            if phase_ref != 'l':
                H_int_Tm_to_Tb_l = Cpl_int(Tm, Tb) if (Tm and Tb) else None
            if phase_ref == 's':
                H_int_T_ref_to_Tm_s = Cps_int(T_ref, Tm) if Tm else None
            if phase_ref == 'g':
                H_int_Tb_to_T_ref_g = Cpg_int(Tb, T_ref) if Tb else None
            if phase_ref == 'l':
                H_int_T_ref_to_Tb_l = Cpl_int(T_ref, Tb) if Tb else None
                H_int_Tm_to_T_ref_l = Cpl_int(Tm, T_ref) if Tm else None
    
            # Entropy integrals
            if phase_ref != 'l':
                S_int_Tm_to_Tb_l = Cpl_int_T(Tm, Tb) if (Tm and Tb) else None
            if phase_ref == 's':
                S_int_T_ref_to_Tm_s = Cps_int_T(T_ref, Tm) if Tm else None
            if phase_ref == 'g':
                S_int_Tb_to_T_ref_g = Cpg_int_T(Tb, T_ref) if Tb else None
            if phase_ref == 'l':
                S_int_T_ref_to_Tb_l = Cpl_int_T(T_ref, Tb) if Tb else None
                S_int_Tm_to_T_ref_l = Cpl_int_T(Tm, T_ref) if Tm else None
    
            # Excess constants
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
                S_dep_Tb_Pb_g = S_dep_Tb_P_ref_g = H_dep_Tb_P_ref_g = H_dep_Tb_Pb_g = eos_Tb = None
            
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
        Cpl_Tb = Cp.l(Tb)
        rhol = 1/V.l(Tb, 101325.) * MW
        data = (CAS, MW, Tb, Tc, Pc, Vc, Zc,
                omega, StielPolar, Hvap_Tb, rhol, Cpl_Tb)
        self.sigma = SurfaceTension(data)
        
        # # Other
        self.epsilon = Permittivity((CAS, V.l,))
        # self.solubility_parameter = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def make_static(self, phase, T, P):
        pass

    def Cv(self, T):
        return self.Cp.g(T) - R

    def show(self):
        info = f"{type(self).__name__}:"
        getfield = getattr
        for i in self.__slots__:
            f = getfield(self, i)
            info += f"\n {display_asfunctor(f, name=i, var=i, show_var=False)}"
        print(info)



# %% Chemical

class Chemical:
    __slots__ = ('ID', 'names', 'subgroups', 'data', 'methods', 'eos', 'eos_T_101325')
    cached_chemicals = {}
    
    def __init__(self, ID, *, eos=PR, CAS=None):
        self.ID = ID
        CAS = CAS or CAS_from_any(ID) or ID
        if CAS in self.cached_chemicals:
            chemical = self.cached_chemicals[CAS]
            for i in chemical.__slots__[1:]: setattr(self, i, getattr(chemical, i))
        else:
            info = pubchem_db.search_CAS(CAS)
            self.subgroups = ChemicalSubgroups(info.InChI_key)
            self.names = ChemicalNames(CAS, info.smiles, info.InChI, info.InChI_key, 
                                       info.pubchemid, info.iupac_name, info.common_name)
            atoms = simple_formula_parser(info.formula)
            self.data = data = ChemicalData(CAS, info.MW, atoms)
            self.eos = create_eos(eos, data.Tc, data.Pc, data.omega)
            self.eos_T_101325 = self.eos.to_TP(298.15, 101325)
            self.methods = ChemicalThermoMethods(CAS, data, self.eos, self.eos_T_101325)
            self.cached_chemicals[CAS] = self

    @classmethod
    def new(cls):
        self = object.__new__(cls)
        self.eos_T_101325 = self.eos = self.ID = None
        self.data = ChemicalData.new()
        self.methods = ChemicalThermoMethods.new()
        self.subgroups = ChemicalSubgroups.new()
        self.names = ChemicalNames.new()

    @classmethod
    def substance(cls, ID, *, properties=None, like=None, fallback=None, phaseTP=None):
        self = cls.new(ID)
        getfield = getattr
        for field in ('data', 'methods', 'subgroups', 'names'):
            field_like = like and getfield(like, field)
            field_fallback = fallback and getfield(fallback, field)
            getfield(self, field).fill(properties, field_like, field_fallback)
        if phaseTP: self.methods.make_static(*phaseTP)
        return self

    def phase(self, T, P):
        Tm = self.data.Tm
        if Tm and T <= Tm: return 's'
        elif P <= self.methods.Psat(T): return 'g'
        else: return 'l'
    
    def __repr__(self):
        return f'<{type(self).__name__}: {self.ID}>'
    


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
    