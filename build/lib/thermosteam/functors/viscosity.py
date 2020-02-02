# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import newton
from math import log, exp
from .utils import CASDataReader
from ..functional import horner_polynomial
from ..base import InterpolatedTDependentModel, mu, TPDependentHandleBuilder, TDependentModel, PhaseTPPropertyBuilder
from .miscdata import _VDISaturationDict, VDI_tabular_data
# from .electrochem import _Laliberte_Viscosity_ParametersDict, Laliberte_viscosity
from .dippr import DIPPR_EQ101, DIPPR_EQ102

__all__= ('Viswanath_Natarajan2', 'Viswanath_Natarajan3', 'Letsou_Stiel', 'Przedziecki_Sridhar',
          'VDI', 'Lucas'
)

read = CASDataReader(__file__, 'Viscosity')
_Dutt_Prasad = read('Dutt Prasad 3 term.tsv')
_VN3 = read('Viswanath Natarajan Dynamic 3 term.tsv')
_VN2 = read('Viswanath Natarajan Dynamic 2 term.tsv')
_VN2E_data = read('Viswanath Natarajan Dynamic 2 term Exponential.tsv')
_Perrys2_313 = read('Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv')
_Perrys2_312 = read('Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv')
_VDI_PPDS_7 = read('VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv')
_VDI_PPDS_8 = read('VDI PPDS Dynamic viscosity of gases polynomials.tsv')

@mu.l
def Viswanath_Natarajan2(T, A, B):
    return exp(A + B/T) / 100.

@mu.l
def Viswanath_Natarajan3(T, A, B, C):
    return 10**(A + B/(C - T))/1000.

@mu.l
def Letsou_Stiel(T, MW, Tc, Pc, omega):
    Tr = T/Tc
    xi0 = (1.5174-2.135*Tr + 0.75*Tr**2)*1E-5
    xi1 = (4.2552-7.674*Tr + 3.4*Tr**2)*1E-5
    xi = 2173.424*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    return (xi0 + omega*xi1)/xi

@mu.l
def Przedziecki_Sridhar(T, Tm, Tc, Pc, Vc, Vm, omega, MW):
    Pc = Pc/1E5  # Pa to atm
    Vm, Vc = Vm*1E6, Vc*1E6  # m^3/mol to mL/mol
    Tr = T/Tc
    Gamma = 0.29607 - 0.09045*Tr - 0.04842*Tr**2
    VrT = 0.33593-0.33953*Tr + 1.51941*Tr**2 - 2.02512*Tr**3 + 1.11422*Tr**4
    V = VrT*(1-omega*Gamma)*Vc

    Vo = 0.0085*omega*Tc - 2.02 + Vm/(0.342*(Tm/Tc) + 0.894)  # checked
    E = -1.12 + Vc/(12.94 + 0.1*MW - 0.23*Pc + 0.0424*Tm - 11.58*(Tm/Tc))
    return Vo/(E*(V-Vo))/1000.

@mu.l
def VDI(T, A, B, C, D, E):
    term = (C - T)/(T-D)
    if term < 0:
        term1 = -((T - C)/(T-D))**(1/3.)
    else:
        term1 = term**(1/3.)
    term2 = term*term1
    return E*exp(A*term1 + B*term2)

@mu.l
def Lucas(T, P, Tc, Pc, omega, P_sat, mu):
    Tr = T/Tc
    C = -0.07921+2.1616*Tr - 13.4040*Tr**2 + 44.1706*Tr**3 - 84.8291*Tr**4 \
        + 96.1209*Tr**5-59.8127*Tr**6+15.6719*Tr**7
    D = 0.3257/((1.0039-Tr**2.573)**0.2906) - 0.2086
    A = 0.9991 - 4.674E-4/(1.0523*Tr**-0.03877 - 1.0513)
    dPr = (P-P_sat(T))/Pc
    if dPr < 0: dPr = 0
    return (1. + D*(dPr/2.118)**A)/(1. + C*omega*dPr)*mu.l(T)

@TPDependentHandleBuilder
def ViscosityLiquid(handle, CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, Vl):
    if CAS in _VDISaturationDict:
        Ts, Ys = VDI_tabular_data(CAS, 'Mu (l)')
        handle.model(InterpolatedTDependentModel(Ts, Ys, Ts[0], Ts[-1]))
    if CAS in _Dutt_Prasad:
        _, A, B, C, Tmin, Tmax = _Dutt_Prasad[CAS]
        data = (A, B, C)
        handle.model(Viswanath_Natarajan3.from_args(data), Tmin, Tmax)
    if CAS in _VN3:
        _, _, A, B, C, Tmin, Tmax = _VN3[CAS]
        data = (A, B, C)
        handle.model(Viswanath_Natarajan3.from_args(data), Tmin, Tmax)
    if CAS in _VN2:
        _, _, A, B, Tmin, Tmax = _VN2[CAS]
        data = (A, B)
        handle.model(Viswanath_Natarajan2.from_args(data), Tmin ,Tmax)
    if CAS in _Perrys2_313:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = _Perrys2_313[CAS]
        data = (C1, C2, C3, C4, C5)
        handle.model(DIPPR_EQ101.from_args(data), Tmin, Tmax)
    if CAS in _VDI_PPDS_7:
        coef = _VDI_PPDS_7[CAS][2:]
        handle.model(VDI.from_args(coef))
    data = (MW, Tc, Pc, omega)
    if all(data):
        handle.model(Letsou_Stiel.from_args(data), Tc/4, Tc)
    data = (MW, Tm, Tc, Pc, Vc, omega, Vl)
    if all(data):
        handle.model(Przedziecki_Sridhar.from_args(data), Tm, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for mu_l in handle.models:
            if isinstance(mu_l, TDependentModel): break
        data = (Tc, Pc, omega, Psat, mu_l)
        handle.model(Lucas.from_args(data), Tm, Tc)


### Viscosity of Gases - low pressure
@mu.g
def Yoon_Thodos(T, Tc, Pc, MW):
    Tr = T/Tc
    xi = 2173.4241*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    a = 46.1
    b = 0.618
    c = 20.4
    d = -0.449
    e = 19.4
    f = -4.058
    return (1. + a*Tr**b - c * exp(d*Tr) + e*exp(f*Tr))/(1E8*xi)

@mu.g
def Stiel_Thodos(T, Tc, Pc, MW):
    Pc = Pc/101325.
    Tr = T/Tc
    xi = Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    if Tr > 1.5:
        mu_g = 17.78E-5*(4.58*Tr-1.67)**.625/xi
    else:
        mu_g = 34E-5*Tr**0.94/xi
    return mu_g/1000.

_lucas_Q_dict = {'7440-59-7': 1.38, '1333-74-0': 0.76, '7782-39-0': 0.52}

@mu.g
def lucas_gas(T, Tc, Pc, Zc, MW, Q, dipole=0):
    Tr = T/Tc
    xi = 0.176*(Tc/MW**3/(Pc/1E5)**4)**(1/6.)  # bar arrording to example in Poling
    if dipole is None:
        dipole = 0
    dipoler = 52.46*dipole**2*(Pc/1E5)/Tc**2  # bar arrording to example in Poling
    if dipoler < 0.022:
        Fp = 1
    elif 0.022 <= dipoler < 0.075:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72
    else:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72*abs(0.96 + 0.1*(Tr-0.7))
    if Q:
        if Tr - 12 > 0:
            value = 1
        else:
            value = -1
        FQ = 1.22*Q**0.15*(1 + 0.00385*((Tr-12)**2)**(1./MW)*value)
    else:
        FQ = 1
    eta = (0.807*Tr**0.618 - 0.357*exp(-0.449*Tr) + 0.340*exp(-4.058*Tr) + 0.018)*Fp*FQ/xi
    return eta/1E7

@mu.g
def Gharagheizi_gas_viscosity(T, Tc, Pc, MW):
    Tr = T/Tc
    mu_g = 1E-5*Pc*Tr + (0.091 - 0.477/MW)*T + MW*(1E-5*Pc - 8*MW**2/T**2)*(10.7639/Tc - 4.1929/T)
    return 1E-7 * abs(mu_g)


GHARAGHEIZI = 'GHARAGHEIZI'
YOON_THODOS = 'YOON_THODOS'
STIEL_THODOS = 'STIEL_THODOS'
LUCAS_GAS = 'LUCAS_GAS'


@TPDependentHandleBuilder
def ViscosityGas(handle, CAS, MW, Tc, Pc, Zc, dipole):
    if CAS in _Perrys2_312:
        _, C1, C2, C3, C4, Tmin, Tmax = _Perrys2_312[CAS]
        data = (C1, C2, C3, C4)
        handle.model(DIPPR_EQ102.from_args(data), Tmin, Tmax)
    if CAS in _VDI_PPDS_8:
        data = _VDI_PPDS_8[CAS].tolist()[1:]
        data.reverse()
        handle.model(horner_polynomial.from_kwargs({'coeffs':data}))
    data = (Tc, Pc, Zc, MW)
    if all(data):
        Tmin = 0; Tmax = 1e3
        handle.model(lucas_gas.from_args(data), Tmin, Tmax)
    data = (Tc, Pc, MW)
    if all(data):
        Tmin = 0; Tmax = 5e3
        handle.model(Gharagheizi_gas_viscosity.from_args(data), Tmin, Tmax)
        handle.model(Yoon_Thodos.from_args(data), Tmin, Tmax)
        handle.model(Stiel_Thodos.from_args(data), Tmin, Tmax)
        # Intelligently set limit
        # GHARAGHEIZI turns nonsensical at ~15 K, YOON_THODOS fine to 0 K,
        # same as STIEL_THODOS
    if CAS in _VDISaturationDict:
        Ts, Ys = VDI_tabular_data(CAS, 'Mu (g)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        handle.model(InterpolatedTDependentModel(Ts, Ys, Tmin, Tmax))

Viscosity = PhaseTPPropertyBuilder(None, ViscosityLiquid, ViscosityGas, 'mu')

# %% Other

VI_nus = np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1,
                   3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3,
                   4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5,
                   5.6, 5.7, 5.8, 5.9, 6, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
                   6.8, 6.9, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
                   8, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9, 9.1,
                   9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10, 10.1, 10.2,
                   10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11, 11.1, 11.2,
                   11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12, 12.1, 12.2,
                   12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13, 13.1, 13.2,
                   13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14, 14.1, 14.2,
                   14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2,
                   15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16, 16.1, 16.2,
                   16.3, 16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17, 17.1, 17.2,
                   17.3, 17.4, 17.5, 17.6, 17.7, 17.8, 17.9, 18, 18.1, 18.2,
                   18.3, 18.4, 18.5, 18.6, 18.7, 18.8, 18.9, 19, 19.1, 19.2,
                   19.3, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20, 20.2, 20.4,
                   20.6, 20.8, 21, 21.2, 21.4, 21.6, 21.8, 22, 22.2, 22.4,
                   22.6, 22.8, 23, 23.2, 23.4, 23.6, 23.8, 24, 24.2, 24.4,
                   24.6, 24.8, 25, 25.2, 25.4, 25.6, 25.8, 26, 26.2, 26.4,
                   26.6, 26.8, 27, 27.2, 27.4, 27.6, 27.8, 28, 28.2, 28.4,
                   28.6, 28.8, 29, 29.2, 29.4, 29.6, 29.8, 30, 30.5, 31,
                   31.5, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 36.5,
                   37, 37.5, 38, 38.5, 39, 39.5, 40, 40.5, 41, 41.5, 42,
                   42.5, 43, 43.5, 44, 44.5, 45, 45.5, 46, 46.5, 47, 47.5,
                   48, 48.5, 49, 49.5, 50, 50.5, 51, 51.5, 52, 52.5, 53,
                   53.5, 54, 54.5, 55, 55.5, 56, 56.5, 57, 57.5, 58, 58.5,
                   59, 59.5, 60, 60.5, 61, 61.5, 62, 62.5, 63, 63.5, 64,
                   64.5, 65, 65.5, 66, 66.5, 67, 67.5, 68, 68.5, 69, 69.5, 70])
VI_Ls = np.array([7.994, 8.64, 9.309, 10, 10.71, 11.45, 12.21, 13, 13.8,
                  14.63, 15.49, 16.36, 17.26, 18.18, 19.12, 20.09, 21.08,
                  22.09, 23.13, 24.19, 25.32, 26.5, 27.75, 29.07, 30.48,
                  31.96, 33.52, 35.13, 36.79, 38.5, 40.23, 41.99, 43.76,
                  45.53, 47.31, 49.09, 50.87, 52.64, 54.42, 56.2, 57.97,
                  59.74, 61.52, 63.32, 65.18, 67.12, 69.16, 71.29, 73.48,
                  75.72, 78, 80.25, 82.39, 84.53, 86.66, 88.85, 91.04, 93.2,
                  95.43, 97.72, 100, 102.3, 104.6, 106.9, 109.2, 111.5, 113.9,
                  116.2, 118.5, 120.9, 123.3, 125.7, 128, 130.4, 132.8, 135.3,
                  137.7, 140.1, 142.7, 145.2, 147.7, 150.3, 152.9, 155.4, 158,
                  160.6, 163.2, 165.8, 168.5, 171.2, 173.9, 176.6, 179.4,
                  182.1, 184.9, 187.6, 190.4, 193.3, 196.2, 199, 201.9, 204.8,
                  207.8, 210.7, 213.6, 216.6, 219.6, 222.6, 225.7, 228.8,
                  231.9, 235, 238.1, 241.2, 244.3, 247.4, 250.6, 253.8, 257,
                  260.1, 263.3, 266.6, 269.8, 273, 276.3, 279.6, 283, 286.4,
                  289.7, 293, 296.5, 300, 303.4, 306.9, 310.3, 313.9, 317.5,
                  321.1, 324.6, 328.3, 331.9, 335.5, 339.2, 342.9, 346.6,
                  350.3, 354.1, 358, 361.7, 365.6, 369.4, 373.3, 377.1, 381,
                  384.9, 388.9, 392.7, 396.7, 400.7, 404.6, 408.6, 412.6,
                  416.7, 420.7, 424.9, 429, 433.2, 437.3, 441.5, 445.7,
                  449.9, 454.2, 458.4, 462.7, 467, 471.3, 475.7, 479.7,
                  483.9, 488.6, 493.2, 501.5, 510.8, 519.9, 528.8, 538.4,
                  547.5, 556.7, 566.4, 575.6, 585.2, 595, 604.3, 614.2,
                  624.1, 633.6, 643.4, 653.8, 663.3, 673.7, 683.9, 694.5,
                  704.2, 714.9, 725.7, 736.5, 747.2, 758.2, 769.3, 779.7,
                  790.4, 801.6, 812.8, 824.1, 835.5, 847, 857.5, 869, 880.6,
                  892.3, 904.1, 915.8, 927.6, 938.6, 951.2, 963.4, 975.4,
                  987.1, 998.9, 1011, 1023, 1055, 1086, 1119, 1151, 1184,
                  1217, 1251, 1286, 1321, 1356, 1391, 1427, 1464, 1501, 1538,
                  1575, 1613, 1651, 1691, 1730, 1770, 1810, 1851, 1892, 1935,
                  1978, 2021, 2064, 2108, 2152, 2197, 2243, 2288, 2333, 2380,
                  2426, 2473, 2521, 2570, 2618, 2667, 2717, 2767, 2817, 2867,
                  2918, 2969, 3020, 3073, 3126, 3180, 3233, 3286, 3340, 3396,
                  3452, 3507, 3563, 3619, 3676, 3734, 3792, 3850, 3908, 3966,
                  4026, 4087, 4147, 4207, 4268, 4329, 4392, 4455, 4517, 4580,
                  4645, 4709, 4773, 4839, 4905])
VI_Hs = np.array([6.394, 6.894, 7.41, 7.944, 8.496, 9.063, 9.647, 10.25,
                  10.87, 11.5, 12.15, 12.82, 13.51, 14.21, 14.93, 15.66,
                  16.42, 17.19, 17.97, 18.77, 19.56, 20.37, 21.21, 22.05,
                  22.92, 23.81, 24.71, 25.63, 26.57, 27.53, 28.49, 29.46,
                  30.43, 31.4, 32.37, 33.34, 34.32, 35.29, 36.26, 37.23,
                  38.19, 39.17, 40.15, 41.13, 42.14, 43.18, 44.24, 45.33,
                  46.44, 47.51, 48.57, 49.61, 50.69, 51.78, 52.88, 53.98,
                  55.09, 56.2, 57.31, 58.45, 59.6, 60.74, 61.89, 63.05,
                  64.18, 65.32, 66.48, 67.64, 68.79, 69.94, 71.1, 72.27,
                  73.42, 74.57, 75.73, 76.91, 78.08, 79.27, 80.46, 81.67,
                  82.87, 84.08, 85.3, 86.51, 87.72, 88.95, 90.19, 91.4,
                  92.65, 93.92, 95.19, 96.45, 97.71, 98.97, 100.2, 101.5,
                  102.8, 104.1, 105.4, 106.7, 108, 109.4, 110.7, 112, 113.3,
                  114.7, 116, 117.4, 118.7, 120.1, 121.5, 122.9, 124.2,
                  125.6, 127, 128.4, 129.8, 131.2, 132.6, 134, 135.4, 136.8,
                  138.2, 139.6, 141, 142.4, 143.9, 145.3, 146.8, 148.2,
                  149.7, 151.2, 152.6, 154.1, 155.6, 157, 158.6, 160.1,
                  161.6, 163.1, 164.6, 166.1, 167.7, 169.2, 170.7, 172.3,
                  173.8, 175.4, 177, 178.6, 180.2, 181.7, 183.3, 184.9,
                  186.5, 188.1, 189.7, 191.3, 192.9, 194.6, 196.2, 197.8,
                  199.4, 201, 202.6, 204.3, 205.9, 207.6, 209.3, 211, 212.7,
                  214.4, 216.1, 217.7, 219.4, 221.1, 222.8, 224.5, 226.2,
                  227.7, 229.5, 233, 236.4, 240.1, 243.5, 247.1, 250.7,
                  254.2, 257.8, 261.5, 264.9, 268.6, 272.3, 275.8, 279.6,
                  283.3, 286.8, 290.5, 294.4, 297.9, 301.8, 305.6, 309.4,
                  313, 317, 320.9, 324.9, 328.8, 332.7, 336.7, 340.5, 344.4,
                  348.4, 352.3, 356.4, 360.5, 364.6, 368.3, 372.3, 376.4,
                  380.6, 384.6, 388.8, 393, 396.6, 401.1, 405.3, 409.5,
                  413.5, 417.6, 421.7, 432.4, 443.2, 454, 464.9, 475.9, 487,
                  498.1, 509.6, 521.1, 532.5, 544, 555.6, 567.1, 579.3,
                  591.3, 603.1, 615, 627.1, 639.2, 651.8, 664.2, 676.6,
                  689.1, 701.9, 714.9, 728.2, 741.3, 754.4, 767.6, 780.9,
                  794.5, 808.2, 821.9, 835.5, 849.2, 863, 876.9, 890.9,
                  905.3, 919.6, 933.6, 948.2, 962.9, 977.5, 992.1, 1007,
                  1021, 1036, 1051, 1066, 1082, 1097, 1112, 1127, 1143,
                  1159, 1175, 1190, 1206, 1222, 1238, 1254, 1270, 1286,
                  1303, 1319, 1336, 1352, 1369, 1386, 1402, 1419, 1436,
                  1454, 1471, 1488, 1506, 1523, 1541, 1558])


def viscosity_index(nu_40, nu_100, rounding=False):
    r'''Calculates the viscosity index of a liquid. Requires dynamic viscosity
    of a liquid at 40°C and 100°C. Value may either be returned with or
    without rounding. Rounding is performed per the standard.

    if nu_100 < 70:

    .. math::
        L, H = interp(nu_100)

    else:

    .. math::
        L = 0.8353\nu_{100}^2 + 14.67\nu_{100} - 216

        H = 0.1684\nu_{100}^2 + 11.85\nu_{100} - 97

    if nu_40 > H:

    .. math::
        VI = \frac{L-nu_{40}}{L-H}\cdot 100

    else:

    .. math::
        N = \frac{\log(H) - \log(\nu_{40})}{\log (\nu_{100})}

         VI = \frac{10^N-1}{0.00715} + 100

    Parameters
    ----------
    nu_40 : float
        Dynamic viscosity of fluid at 40°C, [m^2/s]
    nu_100 : float
        Dynamic viscosity of fluid at 100°C, [m^2/s]
    rounding : bool, optional
        Whether to round the value or not.

    Returns
    -------
    VI: float
        Viscosity index [-]

    Notes
    -----
    VI is undefined for nu_100 under 2 mm^2/s. None is returned if this is the
    case. Internal units are mm^2/s. Higher values of viscosity index suggest
    a lesser decrease in kinematic viscosity as temperature increases.
    
    Note that viscosity is a pressure-dependent property, and that the 
    viscosity index is defined for a fluid at whatever pressure it is at.
    The viscosity index is thus also a function of pressure.

    Examples
    --------
    >>> viscosity_index(73.3E-6, 8.86E-6, rounding=True)
    92

    References
    ----------
    .. [1] ASTM D2270-10(2016) Standard Practice for Calculating Viscosity
       Index from Kinematic Viscosity at 40 °C and 100 °C, ASTM International,
       West Conshohocken, PA, 2016, http://dx.doi.org/10.1520/D2270-10R16
    '''
    nu_40, nu_100 = nu_40*1E6, nu_100*1E6  # m^2/s to mm^2/s
    if nu_100 < 2:
        return None  # Not defined for under this
    elif nu_100 < 70:
        L = np.interp(nu_100, VI_nus, VI_Ls)
        H = np.interp(nu_100, VI_nus, VI_Hs)
    else:
        L = 0.8353*nu_100**2 + 14.67*nu_100 - 216
        H = 0.1684*nu_100**2 + 11.85*nu_100 - 97
    if nu_40 > H:
        VI = (L-nu_40)/(L-H)*100
    else:
        N = (log(H) - log(nu_40))/log(nu_100)
        VI = (10**N-1)/0.00715 + 100
    if rounding:
        VI = _round_whole_even(VI)
    return VI





# All results in units of seconds, except engler and barbey which are degrees
# Data from Hydraulic Institute Handbook

viscosity_scales = {}

SSU_SSU = [31.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 15000.0, 20000.0]
SSU_nu = [1, 2.56, 4.3, 7.4, 10.3, 13.1, 15.7, 18.2, 20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['saybolt universal'] = (SSU_SSU, SSU_nu)

SSF_SSF = [12.95, 13.7, 14.44, 15.24, 19.3, 23.5, 28, 32.5, 41.9, 51.6, 61.4, 71.1, 81, 91, 100.7, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]
SSF_nu = [13.1, 15.7, 18.2, 20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['saybolt furol'] = (SSF_SSF, SSF_nu)

SRS_SRS = [29, 32.1, 36.2, 44.3, 52.3, 60.9, 69.2, 77.6, 85.6, 128, 170, 212, 254, 338, 423, 508, 592, 677, 762, 896, 1270, 1690, 2120, 2540, 3380, 4230, 5080, 5920, 6770, 7620, 8460, 13700, 18400]
SRS_nu = SSU_nu
viscosity_scales['redwood standard'] = (SRS_SRS, SRS_nu)

SRA_SRA = [5.1, 5.83, 6.77, 7.6, 8.44, 9.3, 10.12, 14.48, 18.9, 23.45, 28, 37.1, 46.2, 55.4, 64.6, 73.8, 83, 92.1, 138.2, 184.2, 230, 276, 368, 461, 553, 645, 737, 829, 921]
SRA_nu = [4.3, 7.4, 10.3, 13.1, 15.7, 18.2, 20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200]
viscosity_scales['redwood admiralty'] = (SRA_SRA, SRA_nu)

Engler_degrees = [1, 1.16, 1.31, 1.58, 1.88, 2.17, 2.45, 2.73, 3.02, 4.48, 5.92, 7.35, 8.79, 11.7, 14.6, 17.5, 20.45, 23.35, 26.3, 29.2, 43.8, 58.4, 73, 87.6, 117, 146, 175, 204.5, 233.5, 263, 292, 438, 584]
Engler_nu = SSU_nu
viscosity_scales['engler'] = (Engler_degrees, Engler_nu)

# Note: Barbey is decreasing not increasing
Barbey_degrees = [6200, 2420, 1440, 838, 618, 483, 404, 348, 307, 195, 144, 114, 95, 70.8, 56.4, 47, 40.3, 35.2, 31.3, 28.2, 18.7, 14.1, 11.3, 9.4, 7.05, 5.64, 4.7, 4.03, 3.52, 3.13, 2.82, 2.5, 1.4]
Barbey_nu = SSU_nu
viscosity_scales['barbey'] = (Barbey_degrees, Barbey_nu)
#
PC7_PC7 = [40, 46, 52.5, 66, 79, 92, 106, 120, 135, 149]
PC7_nu = [43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220]
viscosity_scales['parlin cup #7'] = (PC7_PC7, PC7_nu)

PC10_PC10 = [15, 21, 25, 30, 35, 39, 41, 43, 65, 86, 108, 129, 172, 215, 258, 300, 344, 387, 430, 650, 860]
PC10_nu = [65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['parlin cup #10'] = (PC10_PC10, PC10_nu)

PC15_PC15 = [6, 7.2, 7.8, 8.5, 9, 9.8, 10.7, 11.5, 15.2, 19.5, 24, 28.5, 37, 47, 57, 67, 76, 86, 96, 147, 203]
PC15_nu = PC10_nu
viscosity_scales['parlin cup #15'] = (PC15_PC15, PC15_nu)

PC20_PC20 = [3, 3.2, 3.4, 3.6, 3.9, 4.1, 4.3, 4.5, 6.3, 7.5, 9, 11, 14, 18, 22, 25, 29, 32, 35, 53, 70]
PC20_nu = PC10_nu
viscosity_scales['parlin cup #20'] = (PC20_PC20, PC20_nu)

FC3_FC3 = [30, 42, 50, 58, 67, 74, 82, 90, 132, 172, 218, 258, 337, 425, 520, 600, 680, 780, 850, 1280, 1715]
FC3_nu = PC10_nu
viscosity_scales['ford cup #3'] = (FC3_FC3, FC3_nu)

FC4_FC4 = [20, 28, 34, 40, 45, 50, 57, 62, 90, 118, 147, 172, 230, 290, 350, 410, 465, 520, 575, 860, 1150]
FC4_nu = PC10_nu
viscosity_scales['ford cup #4'] = (FC4_FC4, FC4_nu)

MM_MM = [125, 145, 165, 198, 225, 270, 320, 370, 420, 470, 515, 570, 805, 1070, 1325, 1690, 2110, 2635, 3145, 3670, 4170, 4700, 5220, 7720, 10500]
MM_nu = [20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['mac michael'] = (MM_MM, MM_nu)

ZC1_ZC1 = [38, 47, 54, 62, 73, 90]
ZC1_nu = [20.6, 32.1, 43.2, 54, 65, 87.6]
viscosity_scales['zahn cup #1'] = (ZC1_ZC1, ZC1_nu)

ZC2_ZC2 = [18, 20, 23, 26, 29, 37, 46, 55, 63, 72, 80, 88]
ZC2_nu = [20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220]
viscosity_scales['zahn cup #2'] = (ZC2_ZC2, ZC2_nu)

ZC3_ZC3 = [22.5, 24.5, 27, 29, 40, 51, 63, 75]
ZC3_nu = [154, 176, 198, 220, 330, 440, 550, 660]
viscosity_scales['zahn cup #3'] = (ZC3_ZC3, ZC3_nu)

ZC4_ZC4 = [18, 20, 28, 34, 41, 48, 63, 77]
ZC4_nu = [198, 220, 330, 440, 550, 660, 880, 1100]
viscosity_scales['zahn cup #4'] = (ZC4_ZC4, ZC4_nu)

ZC5_ZC5 = [13, 18, 24, 29, 33, 43, 50, 65, 75, 86, 96]
ZC5_nu = [220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980]
viscosity_scales['zahn cup #5'] = (ZC5_ZC5, ZC5_nu)

D1_D1 = [1.3, 2.3, 3.2, 4.1, 4.9, 5.7, 6.5, 10, 13.5, 16.9, 20.4, 27.4, 34.5, 41, 48, 55, 62, 69, 103, 137, 172, 206, 275, 344, 413, 481, 550, 620, 690, 1030, 1370]
D1_nu = [4.3, 7.4, 10.3, 13.1, 15.7, 18.2, 20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['demmier #1'] = (D1_D1, D1_nu)

D10_D10 = [1, 1.4, 1.7, 2, 2.7, 3.5, 4.1, 4.8, 5.5, 6.2, 6.9, 10.3, 13.7, 17.2, 20.6, 27.5, 34.4, 41.3, 48, 55, 62, 69, 103, 137]
D10_nu = [32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['demmier #10'] = (D10_D10, D10_nu)

S100_S100 = [2.6, 3.6, 4.6, 5.5, 6.4, 7.3, 11.3, 15.2, 19, 23, 31, 39, 46, 54, 62, 70, 77, 116, 154, 193, 232, 308, 385, 462, 540, 618, 695, 770, 1160, 1540]
S100_nu = [7.4, 10.3, 13.1, 15.7, 18.2, 20.6, 32.1, 43.2, 54, 65, 87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['stormer 100g load'] = (S100_S100, S100_nu)

PLF_PLF = [7, 8, 9, 9.5, 10.8, 11.9, 12.4, 16.8, 22, 27.6, 33.7, 45, 55.8, 65.5, 77, 89, 102, 113, 172, 234]
PLF_nu = [87.6, 110, 132, 154, 176, 198, 220, 330, 440, 550, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200, 3300, 4400]
viscosity_scales['pratt lambert f'] = (PLF_PLF, PLF_nu)

viscosity_scales['kinematic viscosity'] = (SSU_nu, SSU_nu)


viscosity_converters_to_nu = {}
viscosity_converters_from_nu = {}
viscosity_converter_limits = {}

for key, val in viscosity_scales.items():
    if key == 'barbey':
        continue
    values, nus = val
    viscosity_converter_limits[key] = (values[0], values[-1], nus[0], nus[-1])
    values, nus = np.log(values), np.log(nus)
    viscosity_converters_to_nu[key] = UnivariateSpline(values, nus, k=3, s=0)
    viscosity_converters_from_nu[key] = UnivariateSpline(nus, values, k=3, s=0)

# Barbey gets special treatment because of its reversed values
viscosity_converter_limits['barbey'] = (Barbey_degrees[-1], Barbey_degrees[0], Barbey_nu[0], Barbey_nu[-1])
barbey_values, barbey_nus = np.log(list(reversed(Barbey_degrees))), np.log(list(reversed(Barbey_nu)))
viscosity_converters_to_nu['barbey'] = UnivariateSpline(barbey_values, barbey_nus, k=3, s=0)
viscosity_converters_from_nu['barbey'] = UnivariateSpline(np.log(Barbey_nu), np.log(Barbey_degrees), k=3, s=0)

    

# originally from  Euverard, M. R., "The Efflux Type Viscosity Cup," National 
# Paint, Varnish, and Lacquer Association, 9 April 1948.
# actually found in the Paint Testing Manual
# stored are (coefficient, and minimum time (seconds))
# some of these overlap with the tabulated values; those are used in preference

viscosity_scales_linear = {
    'american can': (3.5, 35), 
    'astm 0.07': (1.4, 60), 
    'astm 0.10': (4.8, 25), 
    'astm 0.15': (21, 9), 
    'astm 0.20': (61, 5), 
    'astm 0.25': (140, 4), 
    'a&w b': (18.5, 10), 
    'a&w crucible': (11.7, 12), 
    'caspers tin plate': (3.6, 39), 
    'continental can': (3.3, 12), 
    'crown cork and seal': (3.3, 12), 
    'engler': (7.3, 18), 
    'ford cup #3': (2.4, 34), 
    'ford cup #4': (3.7, 23), 
    'murphy varnish': (3.1, 24), 
    'parlin cup #7': (1.3, 60), 
    'parlin cup #10': (4.8, 21), 
    'parlin cup #15': (21.5, 10), 
    'parlin cup #20': (60, 5), 
    'parlin cup #25': (140, 15), 
    'parlin cup #30': (260, 10), 
    'pratt lambert a': (0.61, 70), 
    'pratt lambert b': (1.22, 60), 
    'pratt lambert c': (2.43, 40), 
    'pratt lambert d': (4.87, 25), 
    'pratt lambert e': (9.75, 15), 
    'pratt lambert f': (19.5, 9), 
    'pratt lambert g': (38, 7), 
    'pratt lambert h': (76, 5), 
    'pratt lambert i': (152, 4), 
    'redwood standard': (0.23, 320), 
    'saybolt furol': (2.1, 17), 
    'saybolt universal': (0.21, 70), 
    'scott': (1.6, 20), 
    'westinghouse': (3.4, 30), 
    'zahn cup #1': (0.75, 50), 
    'zahn cup #2': (3.1, 30), 
    'zahn cup #3': (9.8, 25), 
    'zahn cup #4': (12.5, 14), 
    'zahn cup #5': (23.6, 12)
}


def viscosity_converter(val, old_scale, new_scale, extrapolate=False):
    r'''Converts kinematic viscosity values from different scales which have
    historically been used. Though they may not be in use much, some standards
    still specify values in these scales.

    Parameters
    ----------
    val : float
        Viscosity value in the specified scale; [m^2/s] if 
        'kinematic viscosity'; [degrees] if Engler or Barbey; [s] for the other
        scales.
    old_scale : str
        String representing the scale that `val` is in originally.
    new_scale : str
        String representing the scale that `val` should be converted to.
    extrapolate : bool
        If True, a conversion will be performed even if outside the limits of
        either scale; if False, and either value is outside a limit, an
        exception will be raised.
        
    Returns
    -------
    result : float
        Viscosity value in the specified scale; [m^2/s] if 
        'kinematic viscosity'; [degrees] if Engler or Barbey; [s] for the other
        scales

    Notes
    -----
    The valid scales for this function are any of the following:
        
    ['a&w b', 'a&w crucible', 'american can', 'astm 0.07', 'astm 0.10', 
    'astm 0.15', 'astm 0.20', 'astm 0.25', 'barbey', 'caspers tin plate', 
    'continental can', 'crown cork and seal', 'demmier #1', 'demmier #10', 
    'engler', 'ford cup #3', 'ford cup #4', 'kinematic viscosity', 
    'mac michael', 'murphy varnish', 'parlin cup #10', 'parlin cup #15', 
    'parlin cup #20', 'parlin cup #25', 'parlin cup #30', 'parlin cup #7', 
    'pratt lambert a', 'pratt lambert b', 'pratt lambert c', 'pratt lambert d', 
    'pratt lambert e', 'pratt lambert f', 'pratt lambert g', 'pratt lambert h',
    'pratt lambert i', 'redwood admiralty', 'redwood standard', 
    'saybolt furol', 'saybolt universal', 'scott', 'stormer 100g load', 
    'westinghouse', 'zahn cup #1', 'zahn cup #2', 'zahn cup #3', 'zahn cup #4',
    'zahn cup #5']
    
    Some of those scales are converted linearly; the rest use tabulated data
    and splines.

    Because the conversion is performed by spline functions, a re-conversion
    of a value will not yield exactly the original value. However, it is quite
    close.
    
    The method 'Saybolt universal' has a special formula implemented for its
    conversion, from [4]_. It is designed for maximum backwards compatibility
    with prior experimental data. It is solved by newton's method when 
    kinematic viscosity is desired as an output.
    
    .. math::
        SUS_{eq} = 4.6324\nu_t + \frac{[1.0 + 0.03264\nu_t]}
        {[(3930.2 + 262.7\nu_t + 23.97\nu_t^2 + 1.646\nu_t^3)\times10^{-5})]}

    Examples
    --------
    >>> viscosity_converter(8.79, 'engler', 'parlin cup #7')
    52.5
    >>> viscosity_converter(700, 'Saybolt Universal Seconds', 'kinematic viscosity')
    0.00015108914751515542

    References
    ----------
    .. [1] Hydraulic Institute. Hydraulic Institute Engineering Data Book. 
       Cleveland, Ohio: Hydraulic Institute, 1990.
    .. [2] Gardner/Sward. Paint Testing Manual. Physical and Chemical 
       Examination of Paints, Varnishes, Lacquers, and Colors. 13th Edition. 
       ASTM, 1972.
    .. [3] Euverard, M. R., The Efflux Type Viscosity Cup. National Paint, 
       Varnish, and Lacquer Association, 1948.
    .. [4] API Technical Data Book: General Properties & Characterization.
       American Petroleum Institute, 7E, 2005.
    .. [5] ASTM. Standard Practice for Conversion of Kinematic Viscosity to 
       Saybolt Universal Viscosity or to Saybolt Furol Viscosity. D 2161 - 93.
    '''

    def range_check(visc, scale):
        scale_min, scale_max, nu_min, nu_max = viscosity_converter_limits[scale]
        
        if visc < scale_min*(1.-1E-7) or visc > scale_max*(1.+1E-7):
            raise Exception('Viscosity conversion is outside the limits of the '
                            '%s scale; given value is %s, but the range of the '
                            'scale is from %s to %s. Set `extrapolate` to True '
                            'to perform the conversion anyway.' %(scale, visc, scale_min, scale_max))

    def range_check_linear(val, c, tmin, scale):
        if val < tmin:
            raise Exception('Viscosity conversion is outside the limits of the '
                            '%s scale; given value is %s, but the minimum time '
                            'for this scale is %s s. Set `extrapolate` to True '
                            'to perform the conversion anyway.' %(scale, val, tmin))

    old_scale = old_scale.lower().replace('degrees', '').replace('seconds', '').strip()
    new_scale = new_scale.lower().replace('degrees', '').replace('seconds', '').strip()
    
    def Saybolt_universal_eq(nu):
        return (4.6324*nu + (1E5 + 3264.*nu)/(nu*(nu*(1.646*nu + 23.97) 
                                              + 262.7) + 3930.2))

    # Convert to kinematic viscosity
    if old_scale == 'kinematic viscosity':
        val = 1E6*val # convert to centistokes, the basis of the functions
    elif old_scale == 'saybolt universal':
        if not extrapolate:
            range_check(val, old_scale)
        to_solve = lambda nu: Saybolt_universal_eq(nu) - val
        val = newton(to_solve, 1)
    elif old_scale in viscosity_converters_to_nu:
        if not extrapolate:
            range_check(val, old_scale)
        val = exp(viscosity_converters_to_nu[old_scale](log(val)))
    elif old_scale in viscosity_scales_linear:
        c, tmin = viscosity_scales_linear[old_scale]
        if not extrapolate:
            range_check_linear(val, c, tmin, old_scale)
        val = c*val # convert from seconds to centistokes
    else:
        keys = sorted(set(list(viscosity_scales.keys()) + list(viscosity_scales_linear.keys())))
        raise Exception('Scale "%s" not recognized - allowable values are any of %s.' %(old_scale, keys))

    # Convert to desired scale
    if new_scale == 'kinematic viscosity':
        val = 1E-6*val # convert to m^2/s
    elif new_scale == 'saybolt universal':
        val = Saybolt_universal_eq(val)
    elif new_scale in viscosity_converters_from_nu:
        val = exp(viscosity_converters_from_nu[new_scale](log(val)))
        if not extrapolate:
            range_check(val, new_scale)
    elif new_scale in viscosity_scales_linear:
        c, tmin = viscosity_scales_linear[new_scale]
        val = val/c # convert from centistokes to seconds
        if not extrapolate:
            range_check_linear(val, c, tmin, new_scale)
    else:
        keys = sorted(set(list(viscosity_scales.keys()) + list(viscosity_scales_linear.keys())))
        raise Exception('Scale "%s" not recognized - allowable values are any of %s.' %(new_scale, keys))
    return float(val)
