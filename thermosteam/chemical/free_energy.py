# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 06:56:43 2019

@author: yoelr
"""
from ..constants import R
from ..base import H, S, ChemicalPhaseTPropertyBuilder, ChemicalPhaseTPPropertyBuilder
from math import log
    
@H.l(njitcompile=False)
def Liquid_Enthalpy_Ref_Liquid(T, Cn_l, T_ref, H_ref):
    """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
    return H_ref + Cn_l.integrate_by_T(T_ref, T)

@H.l(njitcompile=False)
def Liquid_Enthalpy_Ref_Gas(T, Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref):
    return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb + Cn_l.integrate_by_T(Tb, T)
    
@H.l(njitcompile=False)
def Liquid_Enthalpy_Ref_Solid(T, Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref):
    return H_ref + H_int_T_ref_to_Tm_s + Hfus + Cn_l.integrate_by_T(Tm, T)
    
@H.s(njitcompile=False)
def Solid_Enthalpy_Ref_Solid(T, Cn_s, T_ref, H_ref):
    return H_ref + Cn_s.integrate_by_T(T_ref, T)

@H.s(njitcompile=False)
def Solid_Enthalpy_Ref_Liquid(T, Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref):
    return H_ref - H_int_Tm_to_T_ref_l - Hfus + Cn_s.integrate_by_T(Tm, T)

@H.s(njitcompile=False)
def Solid_Enthalpy_Ref_Gas(T, Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb, H_int_Tm_to_Tb_l, Hfus, Tm, H_ref):
    return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb - H_int_Tm_to_Tb_l - Hfus + Cn_s.integrate_by_T(Tm, T)
    
@H.g(njitcompile=False)
def Gas_Enthalpy_Ref_Gas(T, Cn_g, T_ref, H_ref):
    return H_ref + Cn_g.integrate_by_T(T_ref, T)

@H.g(njitcompile=False)
def Gas_Enthalpy_Ref_Liquid(T, Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, T_ref, H_ref):
    return H_ref + H_int_T_ref_to_Tb_l + Hvap_Tb + Cn_g.integrate_by_T(T_ref, T)

@H.g(njitcompile=False)
def Gas_Enthalpy_Ref_Solid(T, Cn_g, H_int_T_ref_to_Tm_s, Hfus, H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref):
    return H_ref + H_int_T_ref_to_Tm_s + Hfus + H_int_Tm_to_Tb_l + Hvap_Tb + Cn_g.integrate_by_T(Tb, T)


EnthalpyRefLiquid = ChemicalPhaseTPropertyBuilder(Solid_Enthalpy_Ref_Liquid,
                                                  Liquid_Enthalpy_Ref_Liquid,
                                                  Gas_Enthalpy_Ref_Liquid,
                                                  'H')

EnthalpyRefSolid = ChemicalPhaseTPropertyBuilder(Solid_Enthalpy_Ref_Solid,
                                                 Liquid_Enthalpy_Ref_Solid,
                                                 Gas_Enthalpy_Ref_Solid,
                                                 'H')

EnthalpyRefGas = ChemicalPhaseTPropertyBuilder(Solid_Enthalpy_Ref_Gas,
                                               Liquid_Enthalpy_Ref_Gas,
                                               Gas_Enthalpy_Ref_Gas,
                                               'H')

@S.l(njitcompile=False)
def Liquid_Entropy_Ref_Liquid(T, Cn_l, T_ref, S_ref):
    """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
    return S_ref + Cn_l.integrate_by_T_over_T(T_ref, T)

@S.l(njitcompile=False)
def Liquid_Entropy_Ref_Gas(T, Cn_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S_ref):
    return S_ref - S_int_Tb_to_T_ref_g - Svap_Tb + Cn_l.integrate_by_T_over_T(Tb, T)
    
@S.l(njitcompile=False)
def Liquid_Entropy_Ref_Solid(T, Cn_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S_ref):
    return S_ref + S_int_T_ref_to_Tm_s + Sfus + Cn_l.integrate_by_T_over_T(Tm, T)
    
@S.s(njitcompile=False)
def Solid_Entropy_Ref_Solid(T, Cn_s, T_ref, S_ref):
    return S_ref + Cn_s.integrate_by_T_over_T(T_ref, T)

@S.s(njitcompile=False)
def Solid_Entropy_Ref_Liquid(T, Cn_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S_ref):
    return S_ref - S_int_Tm_to_T_ref_l - Sfus + Cn_s.integrate_by_T_over_T(Tm, T)

@S.s(njitcompile=False)
def Solid_Entropy_Ref_Gas(T, Cn_s, S_int_Tb_to_T_ref_g, Svap_Tb, S_int_Tm_to_Tb_l, Sfus, Tm, S_ref):
    return S_ref - S_int_Tb_to_T_ref_g - Svap_Tb - S_int_Tm_to_Tb_l - Sfus + Cn_s.integrate_by_T_over_T(Tm, T)
    
@S.g(njitcompile=False)
def Gas_Entropy_Ref_Gas(T, P, Cn_g, T_ref, P_ref, S_ref):
    return S_ref + Cn_g.integrate_by_T_over_T(T_ref, T) - R*log(P/P_ref)

@S.g(njitcompile=False)
def Gas_Entropy_Ref_Liquid(T, P, Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb, T_ref, P_ref, S_ref):
    return S_ref + S_int_T_ref_to_Tb_l + Svap_Tb + Cn_g.integrate_by_T_over_T(T_ref, T) - R*log(P/P_ref)

@S.g(njitcompile=False)
def Gas_Entropy_Ref_Solid(T, P, Cn_g, S_int_T_ref_to_Tm_s, Sfus, S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S_ref):
    return S_ref + S_int_T_ref_to_Tm_s + Sfus + S_int_Tm_to_Tb_l + Svap_Tb + Cn_g.integrate_by_T_over_T(Tb, T) - R*log(P/P_ref)


EntropyRefLiquid = ChemicalPhaseTPPropertyBuilder(Solid_Entropy_Ref_Liquid,
                                                  Liquid_Entropy_Ref_Liquid,
                                                  Gas_Entropy_Ref_Liquid,
                                                  'S')

EntropyRefSolid = ChemicalPhaseTPPropertyBuilder(Solid_Entropy_Ref_Solid,
                                                 Liquid_Entropy_Ref_Solid,
                                                 Gas_Entropy_Ref_Solid,
                                                 'S')

EntropyRefGas = ChemicalPhaseTPPropertyBuilder(Solid_Entropy_Ref_Gas,
                                               Liquid_Entropy_Ref_Gas,
                                               Gas_Entropy_Ref_Gas,
                                               'S')

@H.l(njitcompile=False)
def Excess_Liquid_Enthalpy_Ref_Liquid(T, P):
    return 0

@H.l(njitcompile=False)
def Excess_Liquid_Enthalpy_Ref_Gas(T, P, eos, H_dep_Tb_Pb_g, H_dep_Tb_P_ref_g, eos_1atm):
    return (H_dep_Tb_Pb_g - H_dep_Tb_P_ref_g
            + eos.to_TP(T, P).H_dep_l - eos_1atm.H_dep_l)
    
@H.l(njitcompile=False)
def Excess_Liquid_Enthalpy_Ref_Solid(T, P):
    return 0
    
@H.s(njitcompile=False)
def Excess_Solid_Enthalpy_Ref_Solid(T, P):
    return 0

@H.s(njitcompile=False)
def Excess_Solid_Enthalpy_Ref_Liquid(T, P):
    return 0

@H.s(njitcompile=False)
def Excess_Solid_Enthalpy_Ref_Gas(T, P):
    return 0
    
@H.g(njitcompile=False)
def Excess_Gas_Enthalpy_Ref_Gas(T, P, eos, H_dep_ref_g):
    return eos.to_TP(T, P).H_dep_g - H_dep_ref_g

@H.g(njitcompile=False)
def Excess_Gas_Enthalpy_Ref_Liquid(T, P, eos, H_dep_T_ref_Pb, H_dep_ref_l, H_dep_Tb_Pb_g):
    return H_dep_T_ref_Pb - H_dep_ref_l + eos.to_TP(T, P).H_dep_g - H_dep_Tb_Pb_g

@H.g(njitcompile=False)
def Excess_Gas_Enthalpy_Ref_Solid(T):
    return 0

ExcessEnthalpyRefLiquid = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Enthalpy_Ref_Liquid,
                                                         Excess_Liquid_Enthalpy_Ref_Liquid,
                                                         Excess_Gas_Enthalpy_Ref_Liquid,
                                                         'H_excess')

ExcessEnthalpyRefSolid = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Enthalpy_Ref_Solid,
                                                        Excess_Liquid_Enthalpy_Ref_Solid,
                                                        Excess_Gas_Enthalpy_Ref_Solid,
                                                        'H_excess')

ExcessEnthalpyRefGas = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Enthalpy_Ref_Gas,
                                                      Excess_Liquid_Enthalpy_Ref_Gas,
                                                      Excess_Gas_Enthalpy_Ref_Gas,
                                                      'H_excess')

@S.l(njitcompile=False)
def Excess_Liquid_Entropy_Ref_Liquid(T, P):
    return 0

@S.l(njitcompile=False)
def Excess_Liquid_Entropy_Ref_Gas(T, P, eos, S_dep_Tb_Pb_g, S_dep_Tb_P_ref_g, eos_1atm):
    return (S_dep_Tb_Pb_g - S_dep_Tb_P_ref_g
            + eos.to_TP(T, P).S_dep_l - eos_1atm.S_dep_l)
    
@S.l(njitcompile=False)
def Excess_Liquid_Entropy_Ref_Solid(T, P):
    return 0
    
@S.s(njitcompile=False)
def Excess_Solid_Entropy_Ref_Solid(T, P):
    return 0

@S.s(njitcompile=False)
def Excess_Solid_Entropy_Ref_Liquid(T, P):
    return 0

@S.s(njitcompile=False)
def Excess_Solid_Entropy_Ref_Gas(T, P):
    return 0
    
@S.g(njitcompile=False)
def Excess_Gas_Entropy_Ref_Gas(T, P, eos, S_dep_ref_g):
    return eos.to_TP(T, P).S_dep_g - S_dep_ref_g

@S.g(njitcompile=False)
def Excess_Gas_Entropy_Ref_Liquid(T, P, eos, S_dep_T_ref_Pb, S_dep_ref_l, S_dep_Tb_Pb_g):
    return S_dep_T_ref_Pb - S_dep_ref_l + eos.to_TP(T, P).S_dep_g - S_dep_Tb_Pb_g

@S.g(njitcompile=False)
def Excess_Gas_Entropy_Ref_Solid(T):
    return 0

ExcessEntropyRefLiquid = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Entropy_Ref_Liquid,
                                                        Excess_Liquid_Entropy_Ref_Liquid,
                                                        Excess_Gas_Entropy_Ref_Liquid,
                                                        'S_excess')

ExcessEntropyRefSolid = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Entropy_Ref_Solid,
                                                       Excess_Liquid_Entropy_Ref_Solid,
                                                       Excess_Gas_Entropy_Ref_Solid,
                                                       'S_excess')

ExcessEntropyRefGas = ChemicalPhaseTPPropertyBuilder(Excess_Solid_Entropy_Ref_Gas,
                                                     Excess_Liquid_Entropy_Ref_Gas,
                                                     Excess_Gas_Entropy_Ref_Gas,
                                                     'S_excess')