# -*- coding: utf-8 -*-
"""
Data and methods for calculating a chemical's permitivity.
"""
__all__ = ('Permittivity',)

import numpy as np
from .utils import CASDataReader
from .._constants import N_A, epsilon_0, k
from ..base import TDependentHandleBuilder, epsilon

read = CASDataReader(__file__, 'Electrolytes')
_CRC_Permittivity = read('Permittivity (Dielectric Constant) of Liquids.tsv')

@epsilon
def IAPWS_Permittivity(T, Vl,
                       # Actual molecular dipole moment of water, in C*m
                       dipole = 6.138E-30,
                       # Actual mean molecular polarizability of water, C^2/J*m^2
                       polarizability = 1.636E-40,
                       # Molecular weight of water, kg/mol
                       MW = 0.018015268, 
                       # Coefficients
                       ih = np.array([1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 10]),
                       jh = np.array([0.25, 1, 2.5, 1.5, 1.5, 2.5, 2, 2, 5, 0.5, 10]),
                       Nh = np.array([0.978224486826, -0.957771379375,
                                      0.237511794148, 0.714692244396,
                                      -0.298217036956, -0.108863472196,
                                      0.949327488264E-1, -.980469816509E-2,
                                      0.165167634970E-4, 0.937359795772E-4, 
                                      -0.12317921872E-9])):
    rhom = 1/Vl(T, 101325)
    rho = MW * rhom
    delta = rho/322.
    tau = 647.096/T
    g = (1 + (Nh*delta**ih*tau**jh).sum() + 0.196096504426E-2*delta*(T/228. - 1)**-1.2)
    
    A = N_A*dipole**2*(rhom)*g/epsilon_0/k/T
    B = N_A*polarizability*(rhom)/3./epsilon_0
    return (1. + A + 5.*B + (9. + 2.*A + 18.*B + A**2 + 10.*A*B + 9.*B**2)**0.5)/(4. - 4.*B)

@epsilon
def CRC(T, A, B, C, D):
    return A + B*T + C*T**2 + D*T**3

@TDependentHandleBuilder
def Permittivity(handle, CAS, Vl):
    add_model = handle.add_model
    if Vl and CAS == '7732-18-5':
        add_model(IAPWS_Permittivity.from_args((Vl,)))
    if CAS in _CRC_Permittivity:
        _, CRC_CONSTANT_T, CRC_permittivity, A, B, C, D, Tmin, Tmax = _CRC_Permittivity[CAS]
        args = tuple(0 if np.isnan(x) else x for x in [A, B, C, D])
        Tmin = 0 if np.isnan(Tmin) else Tmin
        Tmax = 1e6 if np.isnan(Tmax) else Tmax
        add_model(CRC.from_args(args), Tmin, Tmax, name='CRC')
        add_model(CRC_permittivity, Tmin, Tmax, name='CRC_constant')


