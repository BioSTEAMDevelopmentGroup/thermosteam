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
from .acentric import omega#, StielPolar
from .triple import Tt, Pt
from .volume import Volume
from .heat_capacity import HeatCapacity
from .reaction import Hf
from .combustion import Hcombustion
from .elements import similarity_variable, simple_formula_parser
from .eos import GCEOS_DUMMY, PR
from .unifac import DDBST_UNIFAC_assignments, DDBST_MODIFIED_UNIFAC_assignments, DDBST_PSRK_assignments
from .free_energy import EnthalpyRefSolid, EnthalpyRefLiquid, EnthalpyRefGas, \
                         EntropyRefSolid, EntropyRefLiquid, EntropyRefGas, \
                         ExcessEnthalpyRefSolid, ExcessEnthalpyRefLiquid, ExcessEnthalpyRefGas, \
                         ExcessEntropyRefSolid, ExcessEntropyRefLiquid, ExcessEntropyRefGas
from ..base import PhaseProperty, display_asfunctor
from ..base.units_of_measure import units_of_measure
from .utils import Z#, R, isentropic_exponent, Joule_Thomson, B_from_Z, isobaric_expansion
# from .thermal_conductivity import ThermalConductivityLiquid, ThermalConductivityGas
# from .permittivity import Permittivity
# from .interface import SurfaceTension
# from .viscosity import ViscosityLiquid, ViscosityGas
# from .safety import Tflash, Tautoignition, LFL, UFL, TWA, STEL, Ceiling, Skin, Carcinogen
# from .solubility import SolubilityParameter
# from .dipole import dipole_moment as dipole

# from .lennard_jones import Stockmayer, MolecularDiameter
# from .environment import GWP, ODP, logP
# from .refractivity import refractive_index
# from .electrochem import conductivity

            
# %% Utilities
                         
def create_eos(eos, Tc, Pc, omega):
    try: return eos(T=298.15, P=101325., Tc=Tc, Pc=Pc, omega=omega)
    except: return GCEOS_DUMMY(T=298.15, P=101325.)
    

# %% Chemical data and descriptors

def show_chemical_descriptor(self):
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

def repr_chemical_decriptor(self):
    return f"<{type(self).__name__}: {', '.join(self.__slots__)}>"

def chemical_descriptor(cls):
    cls.__repr__ = repr_chemical_decriptor
    cls.show = cls._ipython_display_ = show_chemical_descriptor
    return cls

@chemical_descriptor
class ChemicalData:
    __slots__ = ('MW', 'Tm', 'Tb', 'Tt', 'Tc', 'Pt', 'Pc', 'Vc', 'Zc',
                 'Hfus', 'Hsub', 'Hf', 'Hc', 'omega', 'rhoc', 
                 'similarity_variable', 'iscyclic_aliphatic')
    
    def __init__(self, CAS, MW, atoms):
        self.MW = MW
        self.iscyclic_aliphatic = None
        self.Tm = Tm(CAS)
        self.Tb = Tb(CAS)
        self.similarity_variable = similarity_variable(atoms, MW)

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


@chemical_descriptor
class ChemicalSubgroups:
    __slots__ = ('UNIFAC_Dortmund', 'UNIFAC', 'PSRK')
    
    def __init__(self, InChI_key):
        if InChI_key in DDBST_UNIFAC_assignments:
            self.UNIFAC = DDBST_UNIFAC_assignments[InChI_key]
        if InChI_key in DDBST_MODIFIED_UNIFAC_assignments:
            self.UNIFAC_Dortmund = DDBST_MODIFIED_UNIFAC_assignments[InChI_key]
        if InChI_key in DDBST_PSRK_assignments:
            self.PSRK = DDBST_PSRK_assignments[InChI_key]


@chemical_descriptor
class ChemicalNames:
    __slots__ = ('CAS', 'InChI', 'InChI_key', 'common_name', 'iupac_name', 'pubchemid', 'smiles')
    
    def __init__(self, CAS, smiles, InChI, InChI_key, pubchemid, iupac_name, common_name):
        self.CAS = CAS
        self.smiles = smiles
        self.InChI = InChI
        self.InChI_key = InChI_key
        self.pubchemid = pubchemid
        self.iupac_name = iupac_name
        self.common_name = common_name
    

# %% Thermo models

class ChemicalThermoMethods:
    __slots__ = ('Hvap', 'Psat', 'Cp', 'H', 'S', 'V', 'H_excess', 'S_excess')

    def __init__(self, CAS, data, eos, eos_T_101325,
                 T_ref=298.15, P_ref=101325., H_ref=0, S_ref=0):
        # Constants
        MW = data.MW
        Tm = data.Tm
        Tb = data.Tb
        Tc = data.Tc
        Pc = data.Pc
        Zc = data.Zc
        Hfus = data.Hfus
        Sfus = Hfus / Tm
        omega = data.omega
        similarity_variable = data.similarity_variable
        iscyclic_aliphatic = data.iscyclic_aliphatic
        
        # Vapor pressure
        self.Psat = Psat = VaporPressure('Psat', (CAS, Tb, Tc, Pc, omega))
        
        # Volume
        sdata = (CAS,)
        ldata = (CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, Psat, eos)
        gdata = (CAS, Tc, Pc, omega, eos)
        self.V = V = Volume("V", sdata, ldata, gdata)
        
        # Heat capacity
        Cp = PhaseProperty("Cp")
        sdata = (CAS, similarity_variable, MW)
        ldata = (CAS, Tb, Tc, omega, MW, similarity_variable, Cp)
        gdata = (CAS, MW, similarity_variable, iscyclic_aliphatic)
        self.Cp = Cp = HeatCapacity("Cp", sdata, ldata, gdata, Cp)
        
        # Heat of vaporization
        data = (CAS, Tb, Tc, Pc, omega, similarity_variable, Psat, V)
        self.Hvap = Hvap = EnthalpyVaporization("Hvap", data)
        
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
        if phase_ref != 'l' and Tm and Tb:
            H_int_Tm_to_Tb_l = Cpl_int(Tm, Tb)
        if phase_ref == 's' and Tm:
            H_int_T_ref_to_Tm_s = Cps_int(T_ref, Tm)
        if phase_ref == 'g' and Tb:
            H_int_Tb_to_T_ref_g = Cpg_int(Tb, T_ref)
        if phase_ref == 'l' and Tm and Tb:
            H_int_T_ref_to_Tb_l = Cpl_int(T_ref, Tb)
            H_int_Tm_to_T_ref_l = Cpl_int(Tm, T_ref)

        # Entropy integrals
        if phase_ref != 'l' and Tm and Tb:
            S_int_Tm_to_Tb_l = Cpl_int_T(Tm, Tb)
        if phase_ref == 's' and Tm:
            S_int_T_ref_to_Tm_s = Cps_int_T(T_ref, Tm)
        if phase_ref == 'g' and Tb:
            S_int_Tb_to_T_ref_g = Cpg_int_T(Tb, T_ref)
        if phase_ref == 'l' and Tm and Tb:
            S_int_T_ref_to_Tb_l = Cpl_int_T(T_ref, Tb)
            S_int_Tm_to_T_ref_l = Cpl_int_T(Tm, T_ref)

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
        
        # Enthalpy and Entropy
        if phase_ref == 's':
            sdata = Cp, T_ref, H_ref
            ldata = Cp, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref
            gdata = Cp, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref
            self.H = EnthalpyRefSolid('H', sdata, ldata, gdata)
            sdata = Cp, T_ref, S_ref
            ldata = Cp, S_int_T_ref_to_Tm_s, Sfus, Tm, S_ref
            gdata = Cp, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S_ref
            self.S = EntropyRefSolid('S', sdata, ldata, gdata)
        elif phase_ref == 'l':
            sdata = Cp, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref
            ldata = Cp, T_ref, H_ref
            gdata = Cp, H_int_T_ref_to_Tb_l, Hvap_Tb, T_ref, H_ref
            self.H = EnthalpyRefLiquid('H', sdata, ldata, gdata)
            sdata = Cp, S_int_Tm_to_T_ref_l, Sfus, Tm, S_ref
            ldata = Cp, T_ref, S_ref
            gdata = Cp, S_int_T_ref_to_Tb_l, Svap_Tb, T_ref, P_ref, S_ref
            self.S = EntropyRefLiquid('S', sdata, ldata, gdata)
        elif phase_ref == 'g':
            sdata = Cp, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref
            ldata = Cp, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref
            gdata = Cp, T_ref, H_ref
            self.H = EnthalpyRefGas('H', sdata, ldata, gdata)
            sdata = Cp, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S_ref
            ldata = Cp, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S_ref
            gdata = Cp, T_ref, P_ref, S_ref
            self.S = EntropyRefGas('S', sdata, ldata, gdata)
        
        # Excess energies
        if phase_ref == 's':
            gdata = (Cp, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref)
            self.H_excess = ExcessEnthalpyRefSolid('H_exess', (), (), gdata)
            gdata = (Cp, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, S_ref)
            self.S_excess = ExcessEntropyRefSolid('S_excess', (), (), gdata)
        elif phase_ref == 'l':
            gdata = (eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g)
            self.H_excess = ExcessEnthalpyRefLiquid('H_excess', (), (), gdata)
            gdata = (eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_T_101325)
            self.S_excess = ExcessEntropyRefLiquid('S_excess', (), (), gdata)
        elif phase_ref == 'g':
            ldata = (eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_T_101325)
            gdata = eos, H_dep_ref_g
            self.H_excess = ExcessEnthalpyRefGas('H_excess', (), ldata, gdata)
            ldata = (eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g)
            gdata = (eos, S_dep_ref_g)
            self.S_excess = ExcessEntropyRefGas('S_excess', (), ldata, gdata)
        
        # # Viscosity
        # self.mul = ViscosityLiquid(self)
        # self.mug = ViscosityGas(self)
        
        # # Conductivity
        # self.kl = ThermalConductivityLiquid(self)
        # self.kg = ThermalConductivityGas(self)
        
        # # Surface tension
        # self.sigma = SurfaceTension(self)
        
        # # Other
        # self.permittivity = Permittivity(self)
        # self.solubility_parameter = SolubilityParameter(self)
        # self.molecular_diameter = MolecularDiameter(self)

    def show(self):
        info = f"{type(self).__name__}:"
        attr = getattr
        for i in self.__slots__:
            f = attr(self, i)
            info += f"\n {display_asfunctor(f, name=f.var, show_var=False)}"
        print(info)

    __repr__ = repr_chemical_decriptor    
    _ipython_display_  = show


# %% Chemical

class Chemical:
    __slots__ = ('ID', 'names', 'subgroups', 'data', 'thermo', 'eos', 'eos_T_101325')

    def __init__(self, ID, *, eos=PR, CAS=None):
        self.ID = ID
        if not CAS: CAS = CAS_from_any(ID)
        info = pubchem_db.search_CAS(CAS)
        self.subgroups = ChemicalSubgroups(info.InChI_key)
        self.names = ChemicalNames(CAS, info.smiles, info.InChI, info.InChI_key, 
                                   info.pubchemid, info.iupac_name, info.common_name)
        atoms = simple_formula_parser(info.formula)
        self.data = data = ChemicalData(CAS, info.MW, atoms)
        self.eos = create_eos(eos, data.Tc, data.Pc, data.omega)
        self.eos_T_101325 = self.eos.to_TP(298.15, 101325)
        self.thermo = ChemicalThermoMethods(CAS, data, self.eos, self.eos_T_101325)

    def phase(self, T, P):
        Tm = self.data.Tm
        if Tm and T <= Tm: return 's'
        elif P <= self.thermo.Psat(T): return 'g'
        else: return 'l'
    
    def __repr__(self):
        return f'<{type(self).__name__}: {self.ID}>'
    
   

# # Critical Point
# self.StielPolar = StielPolar(CASRN=CAS, Tc=self.Tc, Pc=self.Pc, omega=self.omega)

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
    