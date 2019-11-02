# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:04:44 2019

@author: yoelr
"""
from ..base import Psat, HandleBuilder
from .dippr import DIPPR_EQ101
from .utils import CASDataReader
from math import log, exp
import numpy as np

# %% Data

read = CASDataReader(__file__, 'Vapor Pressure')
_WagnerMcGarry = read("Wagner Original McGarry.tsv")
_Wagner = read("Wagner Collection Poling.tsv")
_Antoine = read("Antoine Collection Poling.tsv")
_AntoineExtended = read("Antoine Extended Collection Poling.tsv")
_Perrys2_8 = read("Table 2-8 Vapor Pressure of Inorganic and Organic Liquids.tsv")
_VDI_PPDS_3 = read("VDI PPDS Boiling temperatures at different pressures.tsv")

# %% Vapor pressure

@Psat
def Antoine(T, A, B, C):
    return 10.0**(A - B / (T + C))

@Psat
def Antoine_Extended(T, Tc, to, A, B, C, n, E, F):
    x = max((T - to - 273.15) / Tc, 0.0)
    return 10.0**(A - B / (T + C) + 0.43429 * x**n + E * x**8 + F * x**12)

@Psat
def Wagner_McGarry(T, A, B, C, D, Tc, Pc):
    Tr = T / Tc
    tau = 1.0 - Tr
    return Pc * exp((A * tau + B * tau**1.5 + C * tau**3 + D * tau**6) / Tr)

@Psat
def Wagner(T, Tc, Pc, A, B, C, D):
    Tr = T / Tc
    τ = 1.0 - Tr
    return Pc * exp((A*τ + B*τ**1.5 + C*τ**2.5 + D*τ**5) / Tr)

@Psat
def Boiling_Critical_Relation(T, Tc, Pc, Tbr, h):
    return exp(h * (1 - Tc / T)) * Pc

@Boiling_Critical_Relation.wrapper
def Boiling_Critical_Relation(Tb, Tc, Pc):
    Tbr = Tb / Tc
    h = Tbr * log(Pc / 101325.0) / (1 - Tbr)
    return {'Tc':Tc, 'Pc':Pc, 'Tbr': Tbr, 'h':h}

@Psat
def Lee_Kesler(T, Tc, Pc, ω):
    Tr = T / Tc
    Tra = Tr**6
    logTr = log(Tr)
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * logTr + 0.169347 * Tra
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * logTr + 0.43577 * Tra
    return exp(f0 + ω * f1) * Pc

@Psat
def Ambrose_Walton(T, Tc, Pc, ω):
    Tr = T / Tc
    τ = 1 - Tr
    τa = τ**1.5
    τb = τ**2.5
    τc = τ**5
    f0 = -5.97616 * τ + 1.29874 * τa - 0.60394 * τb - 1.06841 * τc
    f1 = -5.03365 * τ + 1.11505 * τa - 5.41217 * τb - 7.46628 * τc
    f2 = -0.64771 * τ + 2.41539 * τa - 4.26979 * τb + 3.25259 * τc
    return Pc * exp((f0 + f1 * ω + f2 * ω**2) / Tr)

@Psat
def Sanjari(T, Tc, Pc, ω):
    Tr = T / Tc
    logTr = log(Tr)
    Ta = Tr**1.9
    f0 = 6.83377 + -5.76051 / Tr + 0.90654 * logTr + -1.16906 * Ta
    f1 = 5.32034 + -28.1460 / Tr + -58.0352 * logTr + 23.57466 * Ta
    f2 = 18.19967 + 16.33839 / Tr + 65.6995 * logTr + -35.9739 * Ta
    return Pc * exp(f0 + f1 * ω + f2 * ω**2)

@Psat
def Edalat(T, Tc, Pc, ω, Tmin, Tmax):
    τ = 1.0 - T / Tc
    a = -6.1559 - 4.0855 * ω
    c = -0.8747 - 7.8874 * ω
    d = 1.0 / (-0.4893 - 0.9912 * ω + 3.1551 * ω**2)
    b = 1.5737 - 1.0540 * ω - 4.4365E-3 * d
    lnPr = (a * τ + b * τ**1.5 + c * τ**3.0 + d * τ**6.0) / (1.0 - τ)
    return exp(lnPr) * Pc

@HandleBuilder
def VaporPressure(handle, CAS, Tb, Tc, Pc, omega):
    if CAS in _WagnerMcGarry:
        _, A, B, C, D, Pc, Tc, Tmin = _WagnerMcGarry[CAS]
        Tmax = Tc
        handle.model(Wagner_McGarry(data=(A, B, C, D, Tc, Pc)), Tmin, Tmax)
    if CAS in _Wagner:
        _, A, B, C, D, Tc, Pc, Tmin, Tmax = _Wagner[CAS]
        # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
        if np.isnan(Tmin): Tmin = Tmax * 0.1
        handle.model(Wagner(data=(A, B, C, D, Tc, Pc)), Tmin, Tmax)
    if CAS in _AntoineExtended:
        _, A, B, C, Tc, to, n, E, F, Tmin, Tmax = _AntoineExtended[CAS]
        handle.model(Antoine_Extended(data=(A, B, C, Tc, to, n, E, F)), Tmin, Tmax)
    if CAS in _Antoine:
        _, A, B, C, Tmin, Tmax = _Antoine[CAS]
        handle.model(Antoine(data=(A, B, C)), Tmin, Tmax)
    if CAS in _Perrys2_8:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = _Perrys2_8[CAS]
        handle.model(DIPPR_EQ101(data=(C1, C2, C3, C4, C5)), Tmin, Tmax,)
    if CAS in _VDI_PPDS_3:
        _, Tm, Tc, Pc, A, B, C, D = _VDI_PPDS_3[CAS]
        handle.model(Wagner(data=(Tc, Pc, A, B, C, D)), 0., Tc,)
    data = (Tb, Tc, Pc)
    if all(data):
        handle.model(Boiling_Critical_Relation(data), 0., Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for f in (Lee_Kesler, Ambrose_Walton, Sanjari, Edalat):
            handle.model(f(data), 0., Tc)
    