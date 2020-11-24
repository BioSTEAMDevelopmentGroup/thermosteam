# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Free energy functors for Chemical objects.

"""
from .constants import R
from .base import functor, PhaseTFunctorBuilder, PhaseTPFunctorBuilder
from math import log
    
@functor(var='H')
def Enthalpy(T, Cn, T_ref, H_ref):
    return H_ref + Cn.integrate_by_T(T_ref, T)

@functor(var='S')
def Entropy(T, Cn, T_ref, S0):
    return S0 + Cn.integrate_by_T_over_T(T_ref, T)

@functor(var='H.l')
def Liquid_Enthalpy_Ref_Liquid(T, Cn_l, T_ref, H_ref):
    """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
    return H_ref + Cn_l.integrate_by_T(T_ref, T)

@functor(var='H.l')
def Liquid_Enthalpy_Ref_Gas(T, Cn_l, H_int_Tb_to_T_ref_g, Hvap_Tb, Tb, H_ref):
    return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb + Cn_l.integrate_by_T(Tb, T)
    
@functor(var='H.l')
def Liquid_Enthalpy_Ref_Solid(T, Cn_l, H_int_T_ref_to_Tm_s, Hfus, Tm, H_ref):
    return H_ref + H_int_T_ref_to_Tm_s + Hfus + Cn_l.integrate_by_T(Tm, T)
    
@functor(var='H.s')
def Solid_Enthalpy_Ref_Solid(T, Cn_s, T_ref, H_ref):
    return H_ref + Cn_s.integrate_by_T(T_ref, T)

@functor(var='H.s')
def Solid_Enthalpy_Ref_Liquid(T, Cn_s, H_int_Tm_to_T_ref_l, Hfus, Tm, H_ref):
    return H_ref - H_int_Tm_to_T_ref_l - Hfus + Cn_s.integrate_by_T(Tm, T)

@functor(var='H.s')
def Solid_Enthalpy_Ref_Gas(T, Cn_s, H_int_Tb_to_T_ref_g, Hvap_Tb,
                           H_int_Tm_to_Tb_l, Hfus, Tm, H_ref):
    return H_ref - H_int_Tb_to_T_ref_g - Hvap_Tb - H_int_Tm_to_Tb_l - Hfus + Cn_s.integrate_by_T(Tm, T)
    
@functor(var='H.g')
def Gas_Enthalpy_Ref_Gas(T, Cn_g, T_ref, H_ref):
    return H_ref + Cn_g.integrate_by_T(T_ref, T)

@functor(var='H.g')
def Gas_Enthalpy_Ref_Liquid(T, Cn_g, H_int_T_ref_to_Tb_l, Hvap_Tb, 
                            T_ref, H_ref, Tb):
    return H_ref + H_int_T_ref_to_Tb_l + Hvap_Tb + Cn_g.integrate_by_T(Tb, T)

@functor(var='H.g')
def Gas_Enthalpy_Ref_Solid(T, Cn_g, H_int_T_ref_to_Tm_s, Hfus,
                           H_int_Tm_to_Tb_l, Hvap_Tb, Tb, H_ref):
    return H_ref + H_int_T_ref_to_Tm_s + Hfus + H_int_Tm_to_Tb_l + Hvap_Tb + Cn_g.integrate_by_T(Tb, T)


EnthalpyRefLiquid = PhaseTFunctorBuilder('H',
                                        Solid_Enthalpy_Ref_Liquid.functor,
                                        Liquid_Enthalpy_Ref_Liquid.functor,
                                        Gas_Enthalpy_Ref_Liquid.functor)

EnthalpyRefSolid = PhaseTFunctorBuilder('H',
                                       Solid_Enthalpy_Ref_Solid.functor,
                                       Liquid_Enthalpy_Ref_Solid.functor,
                                       Gas_Enthalpy_Ref_Solid.functor)

EnthalpyRefGas = PhaseTFunctorBuilder('H',
                                     Solid_Enthalpy_Ref_Gas.functor,
                                     Liquid_Enthalpy_Ref_Gas.functor,
                                     Gas_Enthalpy_Ref_Gas.functor)

@functor(var='S.l')
def Liquid_Entropy_Ref_Liquid(T, Cn_l, T_ref, S0):
    """Enthapy (kJ/kmol) disregarding pressure and assuming the specified phase."""
    return S0 + Cn_l.integrate_by_T_over_T(T_ref, T)

@functor(var='S.l')
def Liquid_Entropy_Ref_Gas(T, Cn_l, S_int_Tb_to_T_ref_g, Svap_Tb, Tb, S0):
    return S0 - S_int_Tb_to_T_ref_g - Svap_Tb + Cn_l.integrate_by_T_over_T(Tb, T)
    
@functor(var='S.l')
def Liquid_Entropy_Ref_Solid(T, Cn_l, S_int_T_ref_to_Tm_s, Sfus, Tm, S0):
    return S0 + S_int_T_ref_to_Tm_s + Sfus + Cn_l.integrate_by_T_over_T(Tm, T)
    
@functor(var='S.s')
def Solid_Entropy_Ref_Solid(T, Cn_s, T_ref, S0):
    return S0 + Cn_s.integrate_by_T_over_T(T_ref, T)

@functor(var='S.s')
def Solid_Entropy_Ref_Liquid(T, Cn_s, S_int_Tm_to_T_ref_l, Sfus, Tm, S0):
    return S0 - S_int_Tm_to_T_ref_l - Sfus + Cn_s.integrate_by_T_over_T(Tm, T)

@functor(var='S.s')
def Solid_Entropy_Ref_Gas(T, Cn_s, S_int_Tb_to_T_ref_g, Svap_Tb, 
                          S_int_Tm_to_Tb_l, Sfus, Tm, S0):
    return S0 - S_int_Tb_to_T_ref_g - Svap_Tb - S_int_Tm_to_Tb_l - Sfus + Cn_s.integrate_by_T_over_T(Tm, T)
    
@functor(var='S.g')
def Gas_Entropy_Ref_Gas(T, P, Cn_g, T_ref, P_ref, S0):
    return S0 + Cn_g.integrate_by_T_over_T(T_ref, T) - R*log(P/P_ref)

@functor(var='S.g')
def Gas_Entropy_Ref_Liquid(T, P, Cn_g, S_int_T_ref_to_Tb_l, Svap_Tb,
                           T_ref, P_ref, S0):
    return S0 + S_int_T_ref_to_Tb_l + Svap_Tb + Cn_g.integrate_by_T_over_T(T_ref, T) - R*log(P/P_ref)

@functor(var='S.g')
def Gas_Entropy_Ref_Solid(T, P, Cn_g, S_int_T_ref_to_Tm_s, Sfus,
                          S_int_Tm_to_Tb_l, Svap_Tb, Tb, P_ref, S0):
    return S0 + S_int_T_ref_to_Tm_s + Sfus + S_int_Tm_to_Tb_l + Svap_Tb + Cn_g.integrate_by_T_over_T(Tb, T) - R*log(P/P_ref)


EntropyRefLiquid = PhaseTPFunctorBuilder('S',
                                        Solid_Entropy_Ref_Liquid.functor,
                                        Liquid_Entropy_Ref_Liquid.functor,
                                        Gas_Entropy_Ref_Liquid.functor)

EntropyRefSolid = PhaseTPFunctorBuilder('S',
                                       Solid_Entropy_Ref_Solid.functor,
                                       Liquid_Entropy_Ref_Solid.functor,
                                       Gas_Entropy_Ref_Solid.functor)

EntropyRefGas = PhaseTPFunctorBuilder('S',
                                     Solid_Entropy_Ref_Gas.functor,
                                     Liquid_Entropy_Ref_Gas.functor,
                                     Gas_Entropy_Ref_Gas.functor)

@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Liquid(T, P):
    return 0

@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Gas(T, P, eos, H_dep_Tb_Pb_g,
                                   H_dep_Tb_P_ref_g, eos_1atm):
    return (H_dep_Tb_Pb_g - H_dep_Tb_P_ref_g
            + eos.to_TP(T, P).H_dep_l - eos_1atm.H_dep_l)
    
@functor(var='H.l')
def Excess_Liquid_Enthalpy_Ref_Solid(T, P):
    return 0
    
@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Solid(T, P):
    return 0

@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Liquid(T, P):
    return 0

@functor(var='H.s')
def Excess_Solid_Enthalpy_Ref_Gas(T, P):
    return 0
    
@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Gas(T, P, eos, H_dep_ref_g):
    return eos.to_TP(T, P).H_dep_g - H_dep_ref_g

@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Liquid(T, P, eos, H_dep_T_ref_Pb,
                                   H_dep_ref_l, H_dep_Tb_Pb_g):
    return H_dep_T_ref_Pb - H_dep_ref_l + eos.to_TP(T, P).H_dep_g - H_dep_Tb_Pb_g

@functor(var='H.g')
def Excess_Gas_Enthalpy_Ref_Solid(T):
    return 0

ExcessEnthalpyRefLiquid = PhaseTPFunctorBuilder('H_excess',
                                               Excess_Solid_Enthalpy_Ref_Liquid.functor,
                                               Excess_Liquid_Enthalpy_Ref_Liquid.functor,
                                               Excess_Gas_Enthalpy_Ref_Liquid.functor)

ExcessEnthalpyRefSolid = PhaseTPFunctorBuilder('H_excess',
                                                Excess_Solid_Enthalpy_Ref_Solid.functor,
                                                Excess_Liquid_Enthalpy_Ref_Solid.functor,
                                                Excess_Gas_Enthalpy_Ref_Solid.functor)

ExcessEnthalpyRefGas = PhaseTPFunctorBuilder('H_excess',
                                            Excess_Solid_Enthalpy_Ref_Gas.functor,
                                            Excess_Liquid_Enthalpy_Ref_Gas.functor,
                                            Excess_Gas_Enthalpy_Ref_Gas.functor)

@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Liquid(T, P):
    return 0

@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Gas(T, P, eos, S_dep_Tb_Pb_g,
                                  S_dep_Tb_P_ref_g, eos_1atm):
    return (S_dep_Tb_Pb_g - S_dep_Tb_P_ref_g
            + eos.to_TP(T, P).S_dep_l - eos_1atm.S_dep_l)
    
@functor(var='S.l')
def Excess_Liquid_Entropy_Ref_Solid(T, P):
    return 0
    
@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Solid(T, P):
    return 0

@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Liquid(T, P):
    return 0

@functor(var='S.s')
def Excess_Solid_Entropy_Ref_Gas(T, P):
    return 0
    
@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Gas(T, P, eos, S_dep_ref_g):
    return eos.to_TP(T, P).S_dep_g - S_dep_ref_g

@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Liquid(T, P, eos, S_dep_T_ref_Pb, 
                                  S_dep_ref_l, S_dep_Tb_Pb_g):
    return S_dep_T_ref_Pb - S_dep_ref_l + eos.to_TP(T, P).S_dep_g - S_dep_Tb_Pb_g

@functor(var='S.g')
def Excess_Gas_Entropy_Ref_Solid(T):
    return 0

ExcessEntropyRefLiquid = PhaseTPFunctorBuilder('S_excess',
                                              Excess_Solid_Entropy_Ref_Liquid.functor,
                                              Excess_Liquid_Entropy_Ref_Liquid.functor,
                                              Excess_Gas_Entropy_Ref_Liquid.functor)

ExcessEntropyRefSolid = PhaseTPFunctorBuilder('S_excess',
                                             Excess_Solid_Entropy_Ref_Solid.functor,
                                             Excess_Liquid_Entropy_Ref_Solid.functor,
                                             Excess_Gas_Entropy_Ref_Solid.functor)

ExcessEntropyRefGas = PhaseTPFunctorBuilder('S_excess',
                                           Excess_Solid_Entropy_Ref_Gas.functor,
                                           Excess_Liquid_Entropy_Ref_Gas.functor,
                                           Excess_Gas_Entropy_Ref_Gas.functor)