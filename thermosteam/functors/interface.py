# -*- coding: utf-8 -*-

from math import log, exp
from ..base import sigma, TDependentHandleBuilder, InterpolatedTDependentModel
from .utils import CASDataReader
from .._constants import N_A, k
from .miscdata import _VDISaturationDict, VDI_tabular_data
from .dippr import DIPPR_EQ106

__all__ = ('REFPROP', 'Somayajulu', 'Jasper', 'Brock_Bird', 'Pitzer', 'Sastri_Rao', 'Zuo_Stenby',
           'Hakim_Steinberg_Stiel', 'Miqueu', 'Aleem', 'Mersmann_Kind_surface_tension'
)

read = CASDataReader(__file__, 'Interface')
_Mulero_Cachadina = read('MuleroCachadinaParameters.tsv')
_Jasper_Lange = read('Jasper-Lange.tsv')
_Somayajulu = read('Somayajulu.tsv')
_Somayajulu_2 = read('SomayajuluRevised.tsv')
_VDI_PPDS_11 = read('VDI PPDS surface tensions.tsv')


### Regressed coefficient-based functions

@sigma
def REFPROP(T, Tc, sigma0, n0, sigma1=0, n1=0, sigma2=0, n2=0):
    Tr = T/Tc
    invTr = 1. - Tr
    return sigma0*(invTr)**n0 + sigma1*(invTr)**n1 + sigma2*(invTr)**n2

@sigma
def Somayajulu(T, Tc, A, B, C):
    X = (Tc-T)/Tc
    return (A*X**1.25 + B*X**2.25 + C*X**3.25)/1000.

@sigma
def Jasper(T, a, b):
    return (a - b*(T-273.15))/1000


### CSP methods

@sigma
def Brock_Bird(T, Tb, Tc, Pc):
    Tbr = Tb/Tc
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    Q = 0.1196*(1 + Tbr*log(Pc/1.01325)/(1-Tbr))-0.279
    sigma = (Pc)**(2/3.)*Tc**(1/3.)*Q*(1-Tr)**(11/9.)
    return sigma/1000  # convert to N/m

@sigma
def Pitzer(T, Tc, Pc, omega):
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = Pc**(2/3.0)*Tc**(1/3.0)*(1.86+1.18*omega)/19.05 * (
        (3.75+0.91*omega)/(0.291-0.08*omega))**(2/3.0)*(1-Tr)**(11/9.0)
    return sigma/1000.  # N/m, please

@sigma
def Sastri_Rao(T, Tb, Tc, Pc, chemicaltype=None):
    if chemicaltype == 'alcohol':
        k, x, y, z, m = 2.28, 0.25, 0.175, 0, 0.8
    elif chemicaltype == 'acid':
        k, x, y, z, m = 0.125, 0.50, -1.5, 1.85, 11/9.0
    else:
        k, x, y, z, m = 0.158, 0.50, -1.5, 1.85, 11/9.0
    Tr = T/Tc
    Tbr = Tb/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = k*Pc**x*Tb**y*Tc**z*((1 - Tr)/(1 - Tbr))**m 
    return sigma/1000.  # N/m

def ST_r(ST, Tc, Pc):
    return log(1. + ST/(Tc**(1/3.0)*Pc**(2/3.0)))

@sigma
def Zuo_Stenby(T, Tc, Pc, omega):
    Tc_1, Pc_1, omega_1 = 190.56, 4599000.0/1E5, 0.012
    Tc_2, Pc_2, omega_2 = 568.7, 2490000.0/1E5, 0.4
    Pc = Pc/1E5
    ST_1 = 40.520*(1 - T/Tc)**1.287  # Methane
    ST_2 = 52.095*(1 - T/Tc)**1.21548  # n-octane
    ST_r_1, ST_r_2 = ST_r(ST_1, Tc_1, Pc_1), ST_r(ST_2, Tc_2, Pc_2)
    sigma_r = ST_r_1 + (omega-omega_1)/(omega_2 - omega_1)*(ST_r_2-ST_r_1)
    sigma = Tc**(1/3.0)*Pc**(2/3.0)*(exp(sigma_r)-1)
    return sigma/1000  # N/m, please

@sigma
def Hakim_Steinberg_Stiel(T, Tc, Pc, omega, StielPolar=0):
    Q = (0.1574 + 0.359*omega - 1.769*StielPolar - 13.69*StielPolar**2
        - 0.510*omega**2 + 1.298*StielPolar*omega)
    m = (1.210 + 0.5385*omega - 14.61*StielPolar - 32.07*StielPolar**2
        - 1.656*omega**2 + 22.03*StielPolar*omega)
    Tr = T/Tc
    Pc = Pc/101325.
    sigma = Pc**(2/3.)*Tc**(1/3.)*Q*((1 - Tr)/0.4)**m
    sigma = sigma/1000.  # convert to N/m
    return sigma

@sigma
def Miqueu(T, Tc, Vc, omega):
    Vc = Vc*1E6
    t = 1.-T/Tc
    return k*Tc*(N_A/Vc)**(2/3.)*(4.35 + 4.14*omega)*t**1.26*(1+0.19*t**0.5 - 0.25*t)*10000
    
@sigma 
def Aleem(T, MW, Tb, rhol, Hvap_Tb, Cpl):
    MW = MW/1000. # Use kg/mol for consistency with the other units
    Cpl = Cpl(T)
    sphericity = 1. - 0.0047*MW + 6.8E-6*MW*MW
    return sphericity*MW**(1/3.)/(6.*N_A**(1/3.))*rhol**(2/3.)*(Hvap_Tb + Cpl*(Tb-T))

@sigma 
def Mersmann_Kind_surface_tension(T, Tm, Tb, Tc, Pc, n_associated=1):
    Tr = T/Tc
    sigma_star = ((Tb - Tm)/Tm)**(1/3.)*(6.25*(1. - Tr) + 31.3*(1. - Tr)**(4/3.))
    sigma = sigma_star*(k*Tc)**(1/3.)*(Tm/Tc)*Pc**(2/3.)*n_associated**(-1/3.)
    return sigma

@TDependentHandleBuilder
def SurfaceTension(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, StielPolar, Hvap_Tb, rhol, Cpl_Tb):
    if CAS in _Mulero_Cachadina:
        _, sigma0, n0, sigma1, n1, sigma2, n2, Tc, Tmin, Tmax = _Mulero_Cachadina[CAS]
        STREFPROP_coeffs = (Tc, sigma0, n0, sigma1, n1, sigma2, n2)
        handle.model(REFPROP.from_args(STREFPROP_coeffs), Tmin, Tmax)
    if CAS in _Somayajulu_2:
        _, Tt, Tc, A, B, C = _Somayajulu_2[CAS]
        SOMAYAJULU2_coeffs = (Tc, A, B, C)
        Tmin = Tt; Tmax = Tc
        handle.model(Somayajulu.from_args(SOMAYAJULU2_coeffs), Tmin, Tmax)
    elif CAS in _Somayajulu:
        _, Tt, Tc, A, B, C = _Somayajulu[CAS]
        SOMAYAJULU_coeffs = (Tc, A, B, C)
        Tmin = Tt; Tmax = Tc
        handle.model(Somayajulu.from_args(SOMAYAJULU_coeffs), Tmin, Tmax)
    if CAS in _VDISaturationDict:
        Ts, Ys = VDI_tabular_data(CAS, 'sigma')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        handle.model(InterpolatedTDependentModel(Ts, Ys), Tmin, Tmax)
    if CAS in _Jasper_Lange:
        _, a, b, Tmin, Tmax= _Jasper_Lange[CAS]
        JASPER_coeffs = (a, b)
        handle.model(Jasper.from_args(JASPER_coeffs))
    data = (Tc, Vc, omega)
    if all(data):
        handle.model(Miqueu.from_args(data), 0.0, Tc)
    data = (Tb, Tc, Pc)
    if all(data):
        handle.model(Brock_Bird.from_args(data), 0.0, Tc)
        handle.model(Sastri_Rao.from_args(data), 0.0, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        handle.model(Pitzer.from_args(data), 0.0, Tc)
        handle.model(Zuo_Stenby.from_args(data), 0.0, Tc)
    if CAS in _VDI_PPDS_11:
        _, Tm, Tc, A, B, C, D, E = _VDI_PPDS_11[CAS]
        VDI_PPDS_coeffs = (Tc, A, B, C, D, E)
        handle.model(DIPPR_EQ106.from_args(VDI_PPDS_coeffs))
    data = (MW, Tb, rhol, Hvap_Tb, Cpl_Tb)
    if all(data):
        Tmax = Tb + Hvap_Tb/Cpl_Tb
        # This method will ruin solve_prop as it is typically valids
        # well above Tc. If Tc is available, limit it to that.
        if Tc: Tmax = min(Tc, Tmax)
        handle.model(Aleem.from_args(data))
