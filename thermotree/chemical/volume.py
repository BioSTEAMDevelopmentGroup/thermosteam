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

import numpy as np
from numba import njit
from scipy.interpolate import interp1d
from math import log, exp
from .utils import R, CASDataReader
from .virial import BVirial_Pitzer_Curl, BVirial_Abbott, BVirial_Tsonopoulos, BVirial_Tsonopoulos_extended
from .miscdata import _VDISaturationDict, VDI_tabular_data
from .dippr import DIPPR_EQ105
# from .electrochem import _Laliberte_Density_ParametersDict, Laliberte_Density
from ..base import V, InterpolatedTDependentModel, TPDependentModel, TPDependentHandleBuilder, ChemicalPhaseTPPropertyBuilder

__all__ = ("Volume",)

read = CASDataReader(__file__, "Density")
_COSTALD = read('COSTALD Parameters.tsv')
_SNM0 = read('Mchaweh SN0 deltas.tsv')
_Perry_l = read('Perry Parameters 105.tsv')
_VDI_PPDS_2 = read('VDI PPDS Density of Saturated Liquids.tsv')
_CRC_inorg_l = read('CRC Inorganics densties of molten compounds and salts.tsv')
_CRC_inorg_l_const = read('CRC Liquid Inorganic Constant Densities.tsv')
_CRC_inorg_s_const = read('CRC Solid Inorganic Constant Densities.tsv')
_CRC_virial = read('CRC Virial polynomials.tsv')

# %% Liquids

@V.l
def Yen_Woods(T, Tc, Vc, A, B, D):
    Tr = T/Tc
    return Vc/(1 + A*(1-Tr)**(1/3.) + B*(1-Tr)**(2/3.) + D*(1-Tr)**(4./3.))

@Yen_Woods.wrapper
def Yen_Woods(Tc, Vc, Zc):
    Zc2 = Zc*Zc
    Zc3 = Zc2*Zc
    A = 17.4425 - 214.578*Zc + 989.625*Zc2 - 1522.06*Zc3
    if Zc <= 0.26:
        B = -3.28257 + 13.6377*Zc + 107.4844*Zc2 - 384.211*Zc3
    else:
        B = 60.2091 - 402.063*Zc + 501.0*Zc2 + 641.0*Zc3
    D = 0.93 - B
    return {'Tc':Tc, 'Vc':Vc, 'A':A, 'B':B, 'D':D}

@V.l
def Rackett(T, Tc, Pc, Zc, R=R):
    return R*Tc/Pc*Zc**(1. + (1. - T/Tc)**(2./7.))

@V.l
def Yamada_Gunn(T, P, Tc, k):
    return k**(1 + (1 - T/Tc)**(2/7.))

@Yamada_Gunn.wrapper
def Yamada_Gunn(Tc, Pc, omega):
    k = R*Tc/Pc*(0.29056 - 0.08775*omega)
    return {'Tc':Tc, k:'k'}

@V.l
def Townsend_Hales(T, Tc, Vc, omega):
    Tr = T/Tc
    return Vc/(1 + 0.85*(1-Tr) + (1.692 + 0.986*omega)*(1-Tr)**(1/3.))

Bhirud_normal_Trs = [0.98, 0.982, 0.984, 0.986,
                     0.988, 0.99, 0.992, 0.994,
                     0.996, 0.998, 0.999, 1]
Bhirud_normal_lnU0s = [-1.6198, -1.604, -1.59, -1.578,
                       -1.564, -1.548, -1.533, -1.515,
                       -1.489, -1.454, -1.425, -1.243]
Bhirud_normal_lnU1 = [-0.4626, -0.459, -0.451, -0.441,
                      -0.428, -0.412, -0.392, -0.367,
                      -0.337, -0.302, -0.283, -0.2629]
Bhirud_normal_lnU0_interp = interp1d(Bhirud_normal_Trs,
                                     Bhirud_normal_lnU0s, kind='cubic')
Bhirud_normal_lnU1_interp = interp1d(Bhirud_normal_Trs,
                                     Bhirud_normal_lnU1, kind='cubic')

@V.l
def Bhirud_Normal(T, Tc, Pc, omega):
    Tr = T/Tc
    if Tr <= 0.98:
        lnU0 = 1.39644 - 24.076*Tr + 102.615*Tr**2 - 255.719*Tr**3 \
            + 355.805*Tr**4 - 256.671*Tr**5 + 75.1088*Tr**6
        lnU1 = 13.4412 - 135.7437*Tr + 533.380*Tr**2-1091.453*Tr**3 \
            + 1231.43*Tr**4 - 728.227*Tr**5 + 176.737*Tr**6
    elif Tr > 1:
        raise Exception('Critical phase, correlation does not apply')
    else:
        lnU0 = Bhirud_normal_lnU0_interp(Tr)
        lnU1 = Bhirud_normal_lnU1_interp(Tr)

    Unonpolar = exp(lnU0 + omega*lnU1)
    return (Unonpolar*R*T)/Pc

@V.l
def Costald(T, Tc, Vc, omega):
    Tr = T/Tc
    V_delta = (-0.296123 + 0.386914*Tr - 0.0427258*Tr**2
        - 0.0480645*Tr**3)/(Tr - 1.00001)
    V_0 = 1 - 1.52816*(1-Tr)**(1/3.) + 1.43907*(1-Tr)**(2/3.) \
        - 0.81446*(1-Tr) + 0.190454*(1-Tr)**(4/3.)
    return Vc*V_0*(1-omega*V_delta)

@V.l
def Campbell_Thodos(T, Tb, Tc, Pc, M, dipole=None, hydroxyl=False):
    Pc = Pc/101325.
    Tr = T/Tc
    Tbr = Tb/Tc
    s = Tbr * log(Pc)/(1-Tbr)
    Lambda = Pc**(1/3.)/(M**0.5*Tc**(5/6.))
    alpha = 0.3883 - 0.0179*s
    beta = 0.00318*s - 0.0211 + 0.625*Lambda**(1.35)
    if dipole:
        theta = Pc*dipole**2/Tc**2
        alpha -= 130540 * theta**2.41
        beta += 9.74E6 * theta**3.38
    if hydroxyl:
        beta = 0.00318*s - 0.0211 + 0.625*Lambda**(1.35) + 5.90*theta**0.835
        alpha = (0.69*Tbr - 0.3342 + 5.79E-10/Tbr**32.75)*Pc**0.145
    Zra = alpha + beta*(1-Tr)
    Vs = R*Tc/(Pc*101325)*Zra**(1+(1-Tr)**(2/7.))
    return Vs

@V.l
def SNM0(T, Tc, Vc, m, delta_SRK):
    Tr = T/Tc
    alpha_SRK = (1. + m*(1. - Tr**0.5))**2
    tau = 1. - Tr/alpha_SRK
    rho0 = 1. + 1.169*tau**(1/3.) + 1.818*tau**(2/3.) - 2.658*tau + 2.161*tau**(4/3.)
    V0 = 1./rho0
    return Vc*V0/(1. + delta_SRK*(alpha_SRK - 1.)**(1/3.)) if delta_SRK else Vc*V0

@SNM0.wrapper
def SNM0(Tc, Vc, omega, delta_SRK=None):
    m = 0.480 + 1.574*omega - 0.176*omega*omega
    return {'Tc':Tc, 'Vc':Vc, 'm':m, 'delta_SRK':delta_SRK}

@V.l
def CRC_Inorganic(T, rho0, k, Tm, MW):
    f = MW/1000
    return (rho0 - k*(T-Tm))*f

@V.l
def VDI_PPDS(T, Tc, k, Vc, A, B, C, D):
    tau = 1. - T/Tc
    return Vc + A*tau**0.35 + B*tau**(2/3.) + C*tau + D*tau**(4/3.)

@VDI_PPDS.wrapper
def VDI_PPDS(Tc, A, B, C, D, rhoc, MW):
    k = MW / 1e9
    return {'Tc':Tc, 'k':k, 'Vc':k*rhoc, 'A':k*A, 'B':k*B, 'C':k*C, 'D':k*D}
   
@V.l
def Costald_Compressed(T, P, Psat, Tc, Pc, omega, Vs):
    a = -9.070217
    b = 62.45326
    d = -135.1102
    f = 4.79594
    g = 0.250047
    h = 1.14188
    j = 0.0861488
    k = 0.0344483
    e = exp(f + g*omega + h*omega**2)
    C = j + k*omega
    tau = 1 - T/Tc
    B = Pc*(-1 + a*tau**(1/3.) + b*tau**(2/3.) + d*tau + e*tau**(4/3.))
    return Vs(T, P)*(1 - C*log((B + P)/(B + Psat(T))))

@TPDependentHandleBuilder
def VolumeLiquid(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, Psat, eos):
    Tmin = 50
    Tmax = 1000
    if CAS in _CRC_inorg_l:
        _, MW, rho, k, Tm, Tmax = _CRC_inorg_l[CAS]
        data = (MW, rho, k, Tm)
        handle.model(CRC_Inorganic(data), Tm, Tmax)
    if CAS in _Perry_l:
        _, C1, C2, C3, C4, Tmin, Tmax = _Perry_l[CAS]
        data = (C1, C2, C3, C4, True)
        handle.model(DIPPR_EQ105(data), Tmin, Tmax)
    if Tc and Pc and CAS in _COSTALD:
        Zc_ = _COSTALD.at[CAS, 'Z_RA']
        if not np.isnan(Zc_): Zc_ = float(Zc_)
        data = (Tc, Pc, Zc_)
        handle.model(Rackett(data), Tmin, Tmax)
        # Roughly data at STP; not guaranteed however; not used for Trange
    if Tc and CAS in _COSTALD:
        Vc = float(_COSTALD.at[CAS, 'Vchar'])
        omega = float(_COSTALD.at[CAS, 'omega_SRK'])
        data = (Tc, Vc, omega)
        handle.model(Costald(data), 0, Tc, )
    if CAS in _VDI_PPDS_2:
        _, MW, Tc_, rhoc, A, B, C, D = _VDI_PPDS_2[CAS]
        data = (Tc, A, B, C, D, rhoc, MW)
        handle.model(VDI_PPDS(data), 0., Tc_)
    if CAS in _VDISaturationDict:
        Ts, Vls = VDI_tabular_data(CAS, 'Volume (l)')
        handle.model(InterpolatedTDependentModel(Ts, Vls, Tmin=Ts[0], Tmax=Ts[-1]))
    if all((Tc, Pc, omega)):
        data = (Psat, Tc, Pc, omega, handle)
        handle.model(Costald_Compressed(data), 50, 500)
    if all((Tc, Vc, Zc)):
        data = (Tc, Vc, Zc)
        handle.model(Yen_Woods(data))
    if all((Tc, Pc, Zc)):
        data = (Tc, Pc, Zc)
        handle.model(Rackett(data), 0, Tc)
    if all((Tc, Pc, omega)):
        data = (Tc, Pc, omega)
        handle.model(Yamada_Gunn(data), 0, Tc)
        handle.model(Bhirud_Normal(data), 0, Tc)
    if all((Tc, Vc, omega)):
        data = (Tc, Vc, omega)
        handle.model(Townsend_Hales(data), 0, Tc)
        handle.model(Rackett(data), 0, Tc)
        if CAS in _SNM0:
            SNM0_delta_SRK = float(_SNM0.at[CAS, 'delta_SRK'])
            data = (Tc, Vc, omega, SNM0_delta_SRK)
            handle.model(SNM0(data))
        else:
            handle.model(SNM0(data), 0, Tc)
    if all((Tc, Vc, omega, Tb, MW)):
        handle.model(Campbell_Thodos(data), 0, Tc)
    if CAS in _CRC_inorg_l_const:
        Vl = float(_CRC_inorg_l_const.at[CAS, 'Vm'])
        handle.model(Vl, Tmin, Tmax, name="CRC_inorganic_liquid_constant")
        

# %% Gases

@njit
def ideal_gas(T, P): return R*T/P

ideal_gas_model = TPDependentModel(ideal_gas,
                                   0, 10e6,
                                   0, 10e12)

@V.g
def Tsonopoulos_extended(T, P, Tc, Pc, omega, a=0, b=0, species_type='', dipole=0, order=0):
    return ideal_gas(T, P) + BVirial_Tsonopoulos_extended(T, Tc, Pc, omega, a, b,
                                                          species_type, dipole, order)
@V.g 
def Tsonopoulos(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Tsonopoulos(T, Tc, Pc, omega)

@V.g
def Abbott(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Abbott(T, Tc, Pc, omega)

@V.g
def Pitzer_Curl(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Pitzer_Curl(T, Tc, Pc, omega)
    
@V.g
def CRCVirial(T, P, a1, a2, a3, a4, a5):
    t = 298.15/T - 1.
    return ideal_gas(T, P) + (a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4)/1e6

@TPDependentHandleBuilder
def VolumeGas(handle, CAS, Tc, Pc, omega, eos):
    # no point in getting Tmin, Tmax
    if all((Tc, Pc, omega)):
        data = (Tc, Pc, omega)
        handle.model(Tsonopoulos_extended(data))
        handle.model(Tsonopoulos(data))
        handle.model(Abbott(data))
        handle.model(Pitzer_Curl(data))
    if CAS in _CRC_virial:
        _, *data = _CRC_virial[CAS]
        handle.model(CRCVirial(data))
    handle.model(ideal_gas_model)


# %% Solids

@V.s
def Goodman(T, Tt, Vl):
    return (1.28 - 0.16*(T/Tt))*Vl

@TPDependentHandleBuilder
def VolumeSolid(handle, CAS):
    if CAS in _CRC_inorg_s_const:
        CRC_INORG_S_Vm = float(_CRC_inorg_s_const.at[CAS, 'Vm'])
        handle.model(CRC_INORG_S_Vm, 0, 1e6, 0, 1e12)


Volume = ChemicalPhaseTPPropertyBuilder(VolumeSolid, VolumeLiquid, VolumeGas)

