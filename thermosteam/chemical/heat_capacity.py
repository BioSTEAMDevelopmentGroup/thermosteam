# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

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
import os
from io import open
from math import log, exp
from cmath import log as clog, exp as cexp
import numpy as np
from numba import njit
from ..base import InterpolatedTDependentModel, thermo_model, TDependentHandleBuilder, \
                   ChemicalPhaseTPropertyBuilder, S, H, Cn
from .utils import to_nums, CASDataReader
from ..constants import R, calorie
from ..functional import polylog2
from .miscdata import _VDISaturationDict, VDI_tabular_data
# from .electrochem import (LaliberteHeatCapacityModel,
#                           _Laliberte_Heat_Capacity_ParametersDict)
# from .coolprop import has_CoolProp, coolprop_dict, CoolProp_T_dependent_property,\
#                       coolprop_fluids, PropsSI

__all__ = ('HeatCapacity',)


# %% Utilities

def CnHS(FCn, FH, FS, data):
    fCn = FCn(data)
    return {'evaluate': fCn,
            'integrate_by_T': FH(data, fCn.kwargs),
            'integrate_by_T_over_T': FS(data, fCn.kwargs)}

def CnHSModel(FCn, FH, FS, data=None, Tmin=None, Tmax=None, name=None):
    funcs = CnHS(FCn, FH, FS, data)
    return thermo_model(Tmin=Tmin, Tmax=Tmax, name=name, **funcs)

class ZabranskyModelBuilder:
    __slots__ = ('name', 'casdata', 'funcs', 'many')
    
    def __init__(self, name, casdata, funcs, many=False):
        self.name = name
        self.casdata = casdata
        self.funcs = funcs
        self.many = many
    
    def add_model(self, CAS, models):
        if CAS in self.casdata:
            if self.many:
                models.extend([self.build_model(i) for i in self.casdata[CAS]])
            else:
                models.append(self.build_model(self.casdata[CAS]))
    
    def build_model(self, data):
        *args, Tmin, Tmax = data
        return CnHSModel(*self.funcs, args, Tmin, Tmax, self.name)


# %% Data

read = CASDataReader(__file__, 'Heat Capacity')
_Poling = read('PolingDatabank.tsv')
_TRC_gas = read('TRC Thermodynamics of Organic Compounds in the Gas State.tsv')
_CRC_standard = read('CRC Standard Thermodynamic Properties of Chemical Substances.tsv')
_PerryI = {}

with open(os.path.join(read.folder, 'Perrys Table 2-151.tsv'), encoding='utf-8') as f:
    '''Read in a dict of heat capacities of irnorganic and elemental solids.
    These are in section 2, table 151 in:
    Green, Don, and Robert Perry. Perry's Chemical Engineers' Handbook,
    Eighth Edition. McGraw-Hill Professional, 2007.

    Formula:
    Cn(Cal/mol/K) = Const + Lin*T + Quadinv/T^2 + Quadinv*T^2

    Phases: c, gls, l, g.
    '''
    next(f)
    for line in f:
        values = to_nums(line.strip('\n').split('\t'))
        (CASRN, _formula, _phase, _subphase, Const, Lin, Quadinv, Quad, Tmin,
         Tmax, err) = values
        if Lin is None:
            Lin = 0
        if Quadinv is None:
            Quadinv = 0
        if Quad is None:
            Quad = 0
        if CASRN in _PerryI and CASRN:
            a = _PerryI[CASRN]
            a.update({_phase: {"Formula": _formula, "Phase": _phase,
                               "Subphase": _subphase, "Const": Const,
                               "Lin": Lin, "Quadinv": Quadinv, "Quad": Quad,
                               "Tmin": Tmin, "Tmax": Tmax, "Error": err}})
            _PerryI[CASRN] = a
        else:
            _PerryI[CASRN] = {_phase: {"Formula": _formula, "Phase": _phase,
                                       "Subphase": _subphase, "Const": Const,
                                       "Lin": Lin, "Quadinv": Quadinv,
                                       "Quad": Quad, "Tmin": Tmin,
                                       "Tmax": Tmax, "Error": err}}



# %% Heat Capacities Gas

@Cn.g
def Lastovka_Shaw(T, MW, term, a):
    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    return MW * (term
                 + (B11 + B12*a)*((C11+C12*a)/T)**2*exp(-(C11 + C12*a)/T)/(1.-exp(-(C11+C12*a)/T))**2
                 + (B21 + B22*a)*((C21+C22*a)/T)**2*exp(-(C21 + C22*a)/T)/(1.-exp(-(C21+C22*a)/T))**2)

@Lastovka_Shaw.wrapper
def Lastovka_Shaw(MW, similarity_variable, iscyclic_aliphatic=False):
    a = similarity_variable
    if iscyclic_aliphatic:
        A1 = -0.1793547
        A2 = 3.86944439
        term = A1 + A2*a
    else:
        A1 = 0.58
        A2 = 1.25
        A3 = 0.17338003 # 803 instead of 8003 in another paper
        A4 = 0.014
        term = A2 + (A1-A2)/(1. + exp((a - A3)/A4))
        # Personal communication confirms the change
    return {'MW': MW, 'term':term, 'a':a}

@H.g(wrap=Lastovka_Shaw.wrap)
def Lastovka_Shaw_Integral(T, MW, term, a):
    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    return MW * (T*term - (B11 + B12*a)*(-C11 - C12*a)**2/(-C11 - C12*a + (C11 
                 + C12*a)*exp((-C11 - C12*a)/T)) - (B21 + B22*a)*(-C21 - C22*a)**2/(-C21 
                 - C22*a + (C21 + C22*a)*exp((-C21 - C22*a)/T)))

@S.g(wrap=Lastovka_Shaw.wrap)
def Lastovka_Shaw_Integral_Over_T(T, MW, term, a):
    a2 = a*a
    B11 = 0.73917383
    B12 = 8.88308889
    C11 = 1188.28051
    C12 = 1813.04613
    B21 = 0.0483019
    B22 = 4.35656721
    C21 = 2897.01927
    C22 = 5987.80407
    S = (term*clog(T) + (-B11 - B12*a)*clog(cexp((-C11 - C12*a)/T) - 1.) 
        + (-B11*C11 - B11*C12*a - B12*C11*a - B12*C12*a2)/(T*cexp((-C11
        - C12*a)/T) - T) - (B11*C11 + B11*C12*a + B12*C11*a + B12*C12*a2)/T)
    S += ((-B21 - B22*a)*clog(cexp((-C21 - C22*a)/T) - 1.) + (-B21*C21 - B21*C22*a
        - B22*C21*a - B22*C22*a2)/(T*cexp((-C21 - C22*a)/T) - T) - (B21*C21
        + B21*C22*a + B22*C21*a + B22*C22*a**2)/T)
    # There is a non-real component, but it is only a function of similariy 
    # variable and so will always cancel out.
    return MW * S.real
    
@Cn.g
def TRCCn(T, a0, a1, a2, a3, a4, a5, a6, a7):
    if T <= a7:
        y = 0.
    else:
        y = (T - a7)/(T + a6)
    return R*(a0 + (a1/T**2)*exp(-a2/T) + a3*y**2 + (a4 - a5/(T-a7)**2 )*y**8.)

@njit
def TRCCn_Indefinite_Integral(T, a0, a1, a2, a3, a4, a5, a6, a7):
    first = a6 + a7    
    y = 0. if T <= a7 else (T - a7)/(T + a6)
    y2 = y*y
    y4 = y2*y2
    if T <= a7:
        h = 0.0
    else:
        second = (2.*a3 + 8.*a4)*log(1. - y)
        third = (a3*(1. + 1./(1. - y)) + a4*(7. + 1./(1. - y)))*y
        fourth = a4*(3.*y2 + 5./3.*y*y2 + y4 + 0.6*y4*y + 1/3.*y4*y2)
        fifth = 1/7.*(a4 - a5/((a6 + a7)**2))*y4*y2*y
        h = first*(second + third + fourth + fifth)
    return (a0 + a1*exp(-a2/T)/(a2*T) + h/T)*R*T

@H.g
def TRCCn_Integral(Ta, Tb, a0, a1, a2, a3, a4, a5, a6, a7):
    return (TRCCn_Indefinite_Integral(Tb, a0, a1, a2, a3, a4, a5, a6, a7)
            - TRCCn_Indefinite_Integral(Ta, a0, a1, a2, a3, a4, a5, a6, a7))

@njit
def TRCCn_Over_T_Indefinite_Integral(T, a0, a1, a2, a3, a4, a5, a6, a7):
    # Possible optimizations: pre-cache as much as possible.
    # If this were replaced by a cache, much of this would not need to be computed.
    y = 0. if T <= a7 else (T - a7)/(T + a6)
    z = T/(T + a6)*(a7 + a6)/a7
    if T <= a7:
        s = 0.
    else:
        a72 = a7*a7
        a62 = a6*a6
        a7_a6 = a7/a6 # a7/a6
        a7_a6_2 = a7_a6*a7_a6
        a7_a6_4 = a7_a6_2*a7_a6_2
        x1 = (a4*a72 - a5)/a62 # part of third, sum
        first = (a3 + ((a4*a72 - a5)/a62)*a7_a6_4)*a7_a6_2*log(z)
        second = (a3 + a4)*log((T + a6)/(a6 + a7))
        fourth = -(a3/a6*(a6 + a7) + a5*y**6/(7.*a7*(a6 + a7)))*y
        third = np.array([(x1*(-a7_a6)**(6-i) - a4)*y**i/i for i in np.arange(1, 8)]).sum()
        s = first + second + third + fourth
    return a0*log(T) + a1/(a2*a2)*(1. + a2/T)*exp(-a2/T) + s

@S.g
def TRCCn_Integral_Over_T(Ta, Tb, a0, a1, a2, a3, a4, a5, a6, a7):
    return R*(TRCCn_Over_T_Indefinite_Integral(Tb, a0, a1, a2, a3, a4, a5, a6, a7)
              - TRCCn_Over_T_Indefinite_Integral(Ta, a0, a1, a2, a3, a4, a5, a6, a7))

@Cn.g
def Poling(T, a, b, c, d, e):
    return R*(a + b*T + c*T**2 + d*T**3 + e*T**4)

@njit
def Poling_Indefinite_Integral(T, a, b, c, d, e):
    return (((((0.2*e)*T + 0.25*d)*T + c/3.)*T + 0.5*b)*T + a)*T

@H.g
def Poling_Integral(Ta, Tb, a, b, c, d, e):
    return R*(Poling_Indefinite_Integral(Tb, a, b, c, d, e)
              - Poling_Indefinite_Integral(Tb, a, b, c, d, e))

@njit
def Poling_Over_T_Indefinite_Integral(T, a, b, c, d, e):
    return ((((0.25*e)*T + d/3.)*T + 0.5*c)*T + b)*T

@S.g
def Poling_Integral_Over_T(Ta, Tb, a, b, c, d, e):
    return R*(Poling_Over_T_Indefinite_Integral(Tb, a, b, c, d, e)
              - Poling_Over_T_Indefinite_Integral(Ta, a, b, c, d, e)
              + a*log(Tb/Ta))

# Heat capacity gas methods
TRCIG = 'TRC Thermodynamics of Organic Compounds in the Gas State (1994)'
POLING = 'Poling et al. (2001)'
POLING_CONST = 'Poling et al. (2001) constant'
CRCSTD = 'CRC Standard Thermodynamic Properties of Chemical Substances'
VDI_TABULAR = 'VDI Heat Atlas'
LASTOVKA_SHAW = 'Lastovka and Shaw (2013)'

TRCCn_Functors = (TRCCn,
                  TRCCn_Integral,
                  TRCCn_Integral_Over_T)
Poling_Functors = (Poling,
                   Poling_Integral, 
                   Poling_Integral_Over_T)


@TDependentHandleBuilder
def HeatCapacityGas(handle, CAS, MW, similarity_variable, iscyclic_aliphatic):
    if CAS in _TRC_gas:
        _, Tmin, Tmax, a0, a1, a2, a3, a4, a5, a6, a7, _, _, _ = _TRC_gas[CAS]
        funcs = CnHS(*TRCCn_Functors, (a0, a1, a2, a3, a4, a5, a6, a7))
        handle.model(Tmin=Tmin, Tmax=Tmax, name=TRCIG, **funcs)
    if CAS in _Poling:
        _, Tmin, Tmax, a, b, c, d, e, Cn_g, _ = _Poling[CAS]
        if not np.isnan(a):
            funcs = CnHS(*Poling_Functors, (a, b, c, d, e))
            handle.model(Tmin=Tmin, Tmax=Tmax, **funcs, name=POLING)
        if not np.isnan(Cn_g):
            handle.model(Cn_g, Tmin, Tmax, var='Cn.g', name=POLING_CONST)
    if CAS in _CRC_standard:
        Cn_g = _CRC_standard[CAS][-1]
        if not np.isnan(Cn_g):
            handle.model(Cn_g, name=CRCSTD)
    if CAS in _VDISaturationDict:
        # NOTE: VDI data is for the saturation curve, i.e. at increasing
        # pressure; it is normally substantially higher than the ideal gas
        # value
        Ts, Cn_gs = VDI_tabular_data(CAS, 'Cp (g)')
        handle.model(InterpolatedTDependentModel(Ts, Cn_gs, Tmin=Ts[0], Tmax=Ts[-1], name=VDI_TABULAR))
    if MW and similarity_variable:
        data = (MW, similarity_variable, iscyclic_aliphatic)
        handle.model(Lastovka_Shaw(data), name=LASTOVKA_SHAW)
    
### Heat capacities of liquids

@Cn.l
def Rowlinson_Poling(T, Tc, ω, Cn_g):
    Tr = T/Tc
    return Cn_g + R*(1.586 + 0.49/(1.-Tr) + ω*(4.2775 + 6.3*(1-Tr)**(1/3.)/Tr + 0.4355/(1.-Tr)))

@Cn.l
def Rowlinson_Bondi(T, Tc, ω, Cn_g):
    Tr = T/Tc
    return Cn_g + R*(1.45 + 0.45/(1.-Tr) + 0.25*ω*(17.11 + 25.2*(1-Tr)**(1/3.)/Tr + 1.742/(1.-Tr)))

@Cn.l
def Dadgostar_Shaw(T, MW, first, second, third):
    return (first + second*T + third*T**2) * MW

@Dadgostar_Shaw.wrapper
def Dadgostar_Shaw(similarity_variable, MW):
    a = similarity_variable
    a2 = a*a
    a11 = -0.3416
    a12 = 2.2671
    a21 = 0.1064
    a22 = -0.3874
    a31 = -9.8231E-05
    a32 = 4.182E-04
    # Didn't seem to improve the comparison; sum of errors on some
    # points included went from 65.5  to 286.
    # Author probably used more precision in their calculation.
    #    theta = 151.8675
    #    constant = 3*R*(theta/T)**2*exp(theta/T)/(exp(theta/T)-1)**2
    constant = 24.5
    return {'MW': MW,
            'first': constant*(a11*a + a12*a2),
            'second': a21*a + a22*a2,
            'third': a31*a + a32*a2}

@njit
def Dadgostar_Shaw_Indefinte_Integral(T, MW, first, second, third):
    T2 = T*T
    return T2*T/3.*third + T2*0.5*second + T*first

@H.l(wrap=Dadgostar_Shaw.wrap)
def Dadgostar_Shaw_Integral(Ta, Tb, MW, first, second, third):
    return MW*(Dadgostar_Shaw_Indefinte_Integral(Tb, MW, first, second, third)
               - Dadgostar_Shaw_Indefinte_Integral(Ta, MW, first, second, third))

@njit
def Dadgostar_Shaw_Over_T_Indefinte_Integral(T, MW, first, second, third):
    return T*T*0.5*third + T*second + first*log(T)

@H.l(wrap=Dadgostar_Shaw.wrap) 
def Dadgostar_Shaw_Over_T_Integral(Ta, Tb, MW, first, second, third):
    return MW*(Dadgostar_Shaw_Over_T_Indefinte_Integral(Tb)
               - Dadgostar_Shaw_Over_T_Indefinte_Integral(Ta))

zabransky_dict_sat_s = {}
zabransky_dict_sat_p = {}
zabransky_dict_const_s = {}
zabransky_dict_const_p = {}
zabransky_dict_iso_s = {}
zabransky_dict_iso_p = {}

# C means average heat capacity values, from less rigorous experiments
# sat means heat capacity along the saturation line
# p means constant-pressure values, 
# True means it has a splie, False otherwise
type_to_zabransky_dict = {('C', True): zabransky_dict_const_s, 
                          ('C', False):   zabransky_dict_const_p,
                          ('sat', True):  zabransky_dict_sat_s,
                          ('sat', False): zabransky_dict_sat_p,
                          ('p', True):    zabransky_dict_iso_s,
                          ('p', False):   zabransky_dict_iso_p}

with open(os.path.join(read.folder, 'Zabransky.tsv'), encoding='utf-8') as f:
    next(f)
    for line in f:
        values = to_nums(line.strip('\n').split('\t'))
        (CAS, name, Type, uncertainty, Tmin, Tmax, a1s, a2s, a3s, a4s, a1p, a2p, a3p, a4p, a5p, a6p, Tc) = values
        spline = bool(a1s) # False if Quasypolynomial, True if spline
        d = type_to_zabransky_dict[(Type, spline)]
        if spline:
            if CAS not in d:
                d[CAS] = [(a1s, a2s, a3s, a4s, Tmin, Tmax)]
            else:
                d[CAS].append((a1s, a2s, a3s, a4s, Tmin, Tmax))
        else:
            # No duplicates for quasipolynomials
            d[CAS] = (Tc, a1p, a2p, a3p, a4p, a5p, a6p, Tmin, Tmax)

@Cn.l
def Zabransky_Quasi_Polynomial(T, Tc, a1, a2, a3, a4, a5, a6):
    Tr = T/Tc
    return R*(a1*log(1-Tr) + a2/(1-Tr) + a3 + a4*Tr + a5*Tr**2 + a6*Tr**3)

@njit
def Zabransky_Quasi_Polynomial_Indefinte_Integral(T, Tc, a1, a2, a3, a4, a5, a6):
    Tc2 = Tc*Tc
    Tc3 = Tc2*Tc    
    term = (T - Tc)**2
    return R*(T*(T*(T*(T*a6/(4.*Tc3) + a5/(3.*Tc2)) + a4/(2.*Tc)) - a1 + a3) 
              + T*a1*log(1. - T/Tc) - 0.5*Tc*(a1 + a2)*log(term))

@H.l
def Zabransky_Quasi_Polynomial_Integral(Ta, Tb, Tc, a1, a2, a3, a4, a5, a6):
    return (Zabransky_Quasi_Polynomial_Indefinte_Integral(Tb, Tc, a1, a2, a3, a4, a5, a6)
            - Zabransky_Quasi_Polynomial_Indefinte_Integral(Ta, Tc, a1, a2, a3, a4, a5, a6))

#@njit
def Zabransky_Quasi_Polynomial_Over_T_Indefinte_Integral(T, Tc, a1, a2, a3, a4, a5, a6):
    Tc2 = Tc*Tc
    Tc3 = Tc2*Tc    
    term = (T - Tc)**2
    term = T - Tc
    logT = log(T)
    return R*(a3*logT -a1*polylog2(T/Tc) - a2*(-logT + 0.5*log(term*term))
              + T*(T*(T*a6/(3.*Tc3) + a5/(2.*Tc2)) + a4/Tc))

@S.l(njitcompile=False)
def Zabransky_Quasi_Polynomial_Over_T_Integral(Ta, Tb, Tc, a1, a2, a3, a4, a5, a6):
    return (Zabransky_Quasi_Polynomial_Over_T_Indefinte_Integral(Tb, Tc, a1, a2, a3, a4, a5, a6)
            - Zabransky_Quasi_Polynomial_Over_T_Indefinte_Integral(Ta, Tc, a1, a2, a3, a4, a5, a6))

@Cn.l
def Zabransky_Cubic(T, a1, a2, a3, a4):
    T = T/100.
    return R*(((a4*T + a3)*T + a2)*T + a1)
   
@njit
def Zabransky_Cubic_Indefinite_Integral(T, a1, a2, a3, a4):
    T = T/100.
    return 100.*R*T*(T*(T*(T*a4*0.25 + a3/3.) + a2*0.5) + a1)

@H.l
def Zabransky_Cubic_Integral(Ta, Tb, a1, a2, a3, a4):
    return (Zabransky_Cubic_Indefinite_Integral(Tb, a1, a2, a3, a4)
            - Zabransky_Cubic_Indefinite_Integral(Ta, a1, a2, a3, a4))

@njit
def Zabransky_Cubic_Over_T_Indefinite_Integral(T, a1, a2, a3, a4):
    T = T/100.
    return R*(T*(T*(T*a4/3. + a3/2.) + a2) + a1*log(T))

@S.l
def Zabransky_Cubic_Over_T_Integral(Ta, Tb, a1, a2, a3, a4):
    return (Zabransky_Cubic_Over_T_Indefinite_Integral(Tb, a1, a2, a3, a4)
            - Zabransky_Cubic_Over_T_Indefinite_Integral(Ta, a1, a2, a3, a4))
    

# Heat capacity liquid methods:
ZABRANSKY_SPLINE = 'Zabransky spline, averaged heat capacity'
ZABRANSKY_QUASIPOLYNOMIAL = 'Zabransky quasipolynomial, averaged heat capacity'
ZABRANSKY_SPLINE_C = 'Zabransky spline, constant-pressure'
ZABRANSKY_QUASIPOLYNOMIAL_C = 'Zabransky quasipolynomial, constant-pressure'
ZABRANSKY_SPLINE_SAT = 'Zabransky spline, saturation'
ZABRANSKY_QUASIPOLYNOMIAL_SAT = 'Zabransky quasipolynomial, saturation'
ROWLINSON_POLING = 'Rowlinson and Poling (2001)'
ROWLINSON_BONDI = 'Rowlinson and Bondi (1969)'
DADGOSTAR_SHAW = 'Dadgostar and Shaw (2011)'

Zabransky_Cubic_Functors = (Zabransky_Cubic,
                            Zabransky_Cubic_Integral,
                            Zabransky_Cubic_Over_T_Integral)
Zabransky_Quasi_Polynomial_Functors = (Zabransky_Quasi_Polynomial,
                                    Zabransky_Quasi_Polynomial_Integral, 
                                    Zabransky_Quasi_Polynomial_Over_T_Integral)
Dadgostar_Shaw_Functors = (Dadgostar_Shaw,
                           Dadgostar_Shaw_Integral,
                           Dadgostar_Shaw_Over_T_Integral)

zabransky_model_data = ((ZABRANSKY_SPLINE,
                         zabransky_dict_const_s,
                         Zabransky_Cubic_Functors),
                        (ZABRANSKY_QUASIPOLYNOMIAL,
                         zabransky_dict_const_p,
                         Zabransky_Quasi_Polynomial_Functors),
                        (ZABRANSKY_SPLINE_C,
                         zabransky_dict_iso_s, 
                         Zabransky_Cubic_Functors),
                        (ZABRANSKY_QUASIPOLYNOMIAL_C, 
                         zabransky_dict_iso_p,
                         Zabransky_Quasi_Polynomial_Functors),
                        (ZABRANSKY_SPLINE_SAT,
                         zabransky_dict_sat_s,
                         Zabransky_Cubic_Functors),
                        (ZABRANSKY_QUASIPOLYNOMIAL_SAT, 
                         zabransky_dict_sat_p,
                         Zabransky_Quasi_Polynomial_Functors))

zabransky_model_builders = [ZabranskyModelBuilder(*i) for i in zabransky_model_data]
zabransky_model_builders[0].many = True
zabransky_model_builders[2].many = True
zabransky_model_builders[4].many = True

@TDependentHandleBuilder
def HeatCapacityLiquid(handle, CAS, Tb, Tc, omega, MW, similarity_variable, Cn):
    Cn_g = Cn.g(Tb) if (Tb and Cn.g) else None
    for i in zabransky_model_builders: i.add_model(CAS, handle.models)        
    if CAS in _VDISaturationDict:
        # NOTE: VDI data is for the saturation curve, i.e. at increasing
        # pressure; it is normally substantially higher than the ideal gas
        # value
        Ts, Cn_ls = VDI_tabular_data(CAS, 'Cp (l)')
        handle.model(InterpolatedTDependentModel(Ts, Cn_ls, Ts[0], Ts[-1], name=VDI_TABULAR))
    if Tc and omega and Cn_g:
        args = (Tc, omega, Cn_g, 200, Tc)
        handle.model(Rowlinson_Bondi(args), name=ROWLINSON_BONDI)
        handle.model(Rowlinson_Poling(args), name=ROWLINSON_POLING)
    # Constant models
    if CAS in _Poling:
        _, Tmin, Tmax, a, b, c, d, e, Cn_g, Cn_l = _Poling[CAS]
        if not np.isnan(Cn_g):
            handle.model(Cn_l, Tmin, Tmax, name=POLING_CONST, var="Cn.l")
    if CAS in _CRC_standard:
        Cn_l = _CRC_standard[CAS][-5]
        if not np.isnan(Cn_l):
            handle.model(Cn_l, 0, Tc, name=CRCSTD, var="Cn.l")
    # Other
    if MW and similarity_variable:
        handle.model(CnHSModel(*Dadgostar_Shaw_Functors,
                                 data=(similarity_variable, MW),
                                 name=DADGOSTAR_SHAW))

# %% Heat Capacity Solid

@Cn.s
def Lastovka_Solid(T, similarity_variable, MW):
    A1 = 0.013183
    A2 = 0.249381
    theta = 151.8675
    C1 = 0.026526
    C2 = -0.024942
    D1 = 0.000025
    D2 = -0.000123
    sim2 = similarity_variable*similarity_variable
    return MW * (3*(A1*similarity_variable + A2*sim2)*R*(theta/T)**2
                 * exp(theta/T)/(exp(theta/T)-1)**2
                 + (C1*similarity_variable + C2*sim2)*T
                 + (D1*similarity_variable + D2*sim2)*T**2)

@njit
def Lastovka_Solid_Indefinite_Integral(T, similarity_variable, MW):
    A1 = 0.013183
    A2 = 0.249381
    theta = 151.8675
    C1 = 0.026526
    C2 = -0.024942
    D1 = 0.000025
    D2 = -0.000123
    sim2 = similarity_variable*similarity_variable
    return MW * (T**3*(1000.*D1*similarity_variable/3. 
        + 1000.*D2*sim2/3.) + T*T*(500.*C1*similarity_variable 
        + 500.*C2*sim2)
        + (3000.*A1*R*similarity_variable*theta
        + 3000.*A2*R*sim2*theta)/(exp(theta/T) - 1.))

@H.s
def Lastovka_Solid_Integral(Ta, Tb, similarity_variable, MW):
    return (Lastovka_Solid_Indefinite_Integral(Tb, similarity_variable, MW)
            - Lastovka_Solid_Indefinite_Integral(Ta, similarity_variable, MW))

@njit
def Lastovka_Solid_Over_T_Indefinite_Integral(T, similarity_variable, MW):
    A1 = 0.013183
    A2 = 0.249381
    theta = 151.8675
    C1 = 0.026526
    C2 = -0.024942
    D1 = 0.000025
    D2 = -0.000123
    sim2 = similarity_variable*similarity_variable
    exp_theta_T = exp(theta/T)
    return MW * (-3000.*R*similarity_variable*(A1 + A2*similarity_variable)*log(exp_theta_T - 1.) 
                 + T**2*(500.*D1*similarity_variable + 500.*D2*sim2)
                 + T*(1000.*C1*similarity_variable + 1000.*C2*sim2)
                 + (3000.*A1*R*similarity_variable*theta 
                    + 3000.*A2*R*sim2*theta)/(T*exp_theta_T - T) 
                 + (3000.*A1*R*similarity_variable*theta 
                    + 3000.*A2*R*sim2*theta)/T)
 
@S.s
def Lastovka_Solid_Over_T_Integral(Ta, Tb, similarity_variable, MW):
    return (Lastovka_Solid_Over_T_Indefinite_Integral(Tb, similarity_variable, MW)
            - Lastovka_Solid_Over_T_Indefinite_Integral(Ta, similarity_variable, MW))
    
    
@Cn.s
def Perry_151(T, Const, Lin, Quad, Quadinv):
    return (Const + Lin*T + Quad/T**2 +Quadinv*T**2)*calorie

@H.s
def Perry_151_Integral(Ta, Tb, Const, Lin, Quad, Quadinv):
    H1 = (Const*Ta + 0.5*Lin*Ta**2 - Quadinv/Ta + Quad*Ta**3/3.)
    H2 = (Const*Tb + 0.5*Lin*Tb**2 - Quadinv/Tb + Quad*Tb**3/3.)
    return (H2 - H1)*calorie

@S.s
def Perry_151_Over_T_Integral(Ta, Tb, Const, Lin, Quad, Quadinv):
    S1 = Const*log(Ta) + Lin*Ta - Quadinv/(2.*Ta**2) + 0.5*Quad*Ta**2
    S2 = Const*log(Tb) + Lin*Tb - Quadinv/(2.*Tb**2) + 0.5*Quad*Tb**2
    return (S2 - S1)*calorie

Lastovka_Solid_Functors = (Lastovka_Solid, Lastovka_Solid_Integral, Lastovka_Solid_Over_T_Integral)
Perry_151_Functors = (Perry_151, Perry_151_Integral, Perry_151_Over_T_Integral)

# Heat capacity solid methods
LASTOVKA_S = 'Lastovka, Fulem, Becerra and Shaw (2008)'
PERRY151 = '''Perry's Table 2-151'''

@TDependentHandleBuilder
def HeatCapacitySolid(handle, CAS, similarity_variable, MW):
    Tmin = 0
    Tmax = 2000
    if CAS in _PerryI:
        vals = _PerryI[CAS]
        if 'c' in vals:
            c = vals['c']
            Tmin = c['Tmin']
            Tmax = c['Tmax']
            data = (c['Const'], c['Lin'], c['Quad'], c['Quadinv'])
            handle.model(CnHSModel(*Perry_151_Functors, data), Tmin, Tmax)
    if CAS in _CRC_standard:
        Cnc = _CRC_standard[CAS][3]
        if not np.isnan(Cnc):
            handle.model(float(Cnc), 200, 350)
    if similarity_variable and MW:
        data = (similarity_variable, MW)
        handle.model(CnHSModel(*Lastovka_Solid_Functors, data), Tmin, Tmax)


HeatCapacity = ChemicalPhaseTPropertyBuilder(HeatCapacitySolid, HeatCapacityLiquid, HeatCapacityGas, 'Cn')

