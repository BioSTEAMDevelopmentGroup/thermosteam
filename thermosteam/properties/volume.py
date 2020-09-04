# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the permittivity module from the chemicals's library:
# https://github.com/CalebBell/chemicals
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
import numpy as np
from .row_data import RowData
from ..utils import forward
from chemicals import volume as vol
import sys
module = sys.modules[__name__]
sys.modules[__name__] = vol
del sys
from scipy.interpolate import interp1d
from math import log, exp
from .. import functional as fn
from .._constants import R
from .virial import BVirial_Pitzer_Curl, BVirial_Abbott, BVirial_Tsonopoulos, BVirial_Tsonopoulos_extended
from .dippr import DIPPR_EQ105
# from .electrochem import _Laliberte_Density_ParametersDict, Laliberte_Density
from ..base import (functor, InterpolatedTDependentModel, 
                    TPDependentModel, TPDependentHandleBuilder,
                    PhaseTPHandleBuilder, ThermoModelHandle)
from .data import (VDI_saturation_dict,
                   VDI_tabular_data,
                   V_data_COSTALD,
                   V_data_SNM0,
                   V_data_Perry_l,
                   V_data_VDI_PPDS_2,
                   V_data_CRC_inorg_l,
                   V_data_CRC_inorg_l_const,
                   V_data_CRC_inorg_s_const,
                   V_data_CRC_virial)

__all__ = ('volume_handle',
           'Yen_Woods_saturation',
           'Rackett',
           'Yamada_Gunn', 
           'Townsend_Hales', 
           'Bhirud_Normal', 
           'Costald',
           'Campbell_Thodos', 
           'SNM0', 
           'CRC_Inorganic',
           'VDI_PPDS',
           'Costald_Compressed', 
           'ideal_gas',
           'Tsonopoulos_extended',
           'Tsonopoulos', 
           'Abbott', 
           'Pitzer_Curl', 
           'CRCVirial',
           'Goodman')

# %% Liquids

Yen_Woods_saturation = functor(vol.Yen_Woods_saturation, 'V.l')
Rackett = functor(vol.Rackett, 'V.l')
Yamada_Gunn = functor(vol.Yamada_Gunn, 'V.l')
Townsend_Hales = functor(vol.Townsend_Hales, 'V.l')
Bhirud_Normal = functor(vol.Bhirud_Normal, 'V.l')
COSTALD = functor(vol.COSTALD, 'V.l')
Campbell_Thodos = functor(vol.Campbell_Thodos, 'V.l')
SNM0 = functor(vol.SNM0, 'V.l')
CRC_inorganic = functor(vol.CRC_inorganic, 'V.l')
VDI_PPDS = functor (vol.VDI_PPDS , 'V.l')

@functor(var='V.l')
def COSTALD_Compressed(T, P, Psat, Tc, Pc, omega, V):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that 
    estimates the liquid molar volume (V.l; in m^3/mol) of a chemical using
    the Costald Compressed method, as described in [11]_.
    
    Parameters
    ----------
    Psat : float
        Saturated vapor pressure [Pa].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    V : float
        Molar volume [m^3/mol].
    
    Notes
    -----
    The molar volume of a liquid is given by:

    .. math::
        V = V_s\left( 1 - C \ln \frac{B + P}{B + P^{sat}}\right)

        \frac{B}{P_c} = -1 + a\tau^{1/3} + b\tau^{2/3} + d\tau + e\tau^{4/3}

        e = \exp(f + g\omega_{SRK} + h \omega_{SRK}^2)

        C = j + k \omega_{SRK}

    Original equation was in terms of density, but it is converted here.

    The example is from DIPPR, and exactly correct.
    This is DIPPR Procedure 4C: Method for Estimating the Density of Pure
    Organic Liquids under Pressure.

    Examples
    --------
    >>> f = COSTALD_compressed(9.8E7, 85857.9, 466.7, 3640000.0, 0.281, 0.000105047)
    >>> f(303.)
    9.287482879788506e-05
    
    """
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
    tau = 1. - T/Tc if T < Tc else 0.
    B = Pc*(-1 + a*tau**(1/3.) + b*tau**(2/3.) + d*tau + e*tau**(4/3.))
    if isinstance(Psat, ThermoModelHandle): Psat = Psat.at_T(T)
    if isinstance(V, ThermoModelHandle): V = V.at_T(T)
    return V*(1 - C*log((B + P)/(B + Psat)))

@TPDependentHandleBuilder('V.l')
def volume_liquid_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc,
                         omega, Psat, eos, dipole, has_hydroxyl):
    Tmin = 50
    Tmax = 1000
    all_ = all
    add_model = handle.add_model
    if CAS in V_data_VDI_PPDS_2:
        _, MW, Tc_, rhoc, A, B, C, D = V_data_VDI_PPDS_2[CAS]
        data = (Tc, rhoc, A, B, C, D, MW)
        add_model(VDI_PPDS.from_args(data), 0., Tc_)
    if CAS in V_data_Perry_l:
        _, C1, C2, C3, C4, Tmin, Tmax = V_data_Perry_l[CAS]
        data = (C1, C2, C3, C4, True)
        add_model(DIPPR_EQ105.from_args(data), Tmin, Tmax)
    if CAS in VDI_saturation_dict:
        Ts, Vls = VDI_tabular_data(CAS, 'Volume (l)')
        model = InterpolatedTDependentModel(Ts, Vls,
                                            Tmin=Ts[0], Tmax=Ts[-1])
        add_model(model)
    data = (Tb, Tc, Pc, MW, dipole, has_hydroxyl)
    if all_([i is not None for i in data]):
        add_model(Campbell_Thodos.from_args(data), 0, Tc)
    if CAS in V_data_CRC_inorg_l:
        _, MW, rho, k, Tm, Tmax = V_data_CRC_inorg_l[CAS]
        data = (MW, rho, k, Tm)
        add_model(CRC_inorganic.from_args(data), Tm, Tmax)
    data = (Tc, Vc, Zc)
    if all_(data):
        add_model(Yen_Woods_saturation.from_args(data))
    data = (Tc, Pc, Zc)
    if all_(data):
        add_model(Rackett.from_args(data), 0, Tc)
    data = (Tc, Pc, omega)
    if all_(data):
        add_model(Yamada_Gunn.from_args(data), 0, Tc)
        add_model(Bhirud_Normal.from_args(data), 0, Tc)
    data = (Tc, Vc, omega)
    if all_(data):
        add_model(Townsend_Hales.from_args(data), 0, Tc)
        if CAS in V_data_SNM0:
            SNM0_delta_SRK = float(V_data_SNM0.at[CAS, 'delta_SRK'])
            data = (Tc, Vc, omega, SNM0_delta_SRK)
            add_model(SNM0.from_args(data))
    if CAS in V_data_CRC_inorg_l_const:
        Vl = float(V_data_CRC_inorg_l_const.at[CAS, 'Vm'])
        add_model(Vl, Tmin, Tmax, name="CRC inorganic liquid constant")
    if Tc and Pc and CAS in V_data_COSTALD:
        Zc_ = V_data_COSTALD.at[CAS, 'Z_RA']
        if not np.isnan(Zc_): Zc_ = float(Zc_)
        data = (Tc, Pc, Zc_)
        add_model(Rackett.from_args(data), Tmin, Tmax)
        # Roughly data at STP; not guaranteed however; not used for Trange
    data = (Tc, Vc, omega)
    if all_(data) and CAS in V_data_COSTALD:
        add_model(COSTALD.from_args(data), 0, Tc)
    data = (Tc, Pc, omega)
    if all_(data):
        data = (Psat, Tc, Pc, omega, handle)
        add_model(COSTALD_Compressed.from_args(data), 50, 500)
        

# %% Gases

def ideal_gas(T, P): return R*T/P

ideal_gas_model = TPDependentModel(ideal_gas,
                                   0, 10e6,
                                   0, 10e12,
                                   var='V.g')

@functor(var='V.g')
def Tsonopoulos_extended(T, P, Tc, Pc, omega, a=0, b=0,
                         species_type='', dipole=0, order=0):
    return ideal_gas(T, P) + BVirial_Tsonopoulos_extended(T, Tc, Pc, omega, a, b,
                                                          species_type, dipole, order)
@functor(var='V.g') 
def Tsonopoulos(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Tsonopoulos(T, Tc, Pc, omega)

@functor(var='V.g')
def Abbott(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Abbott(T, Tc, Pc, omega)

@functor(var='V.g')
def Pitzer_Curl(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Pitzer_Curl(T, Tc, Pc, omega)
    
@functor(var='V.g')
def CRCVirial(T, P, a1, a2, a3, a4, a5):
    t = 298.15/T - 1.
    return ideal_gas(T, P) + (a1 + a2*t + a3*t**2 + a4*t**3 + a5*t**4)/1e6

@TPDependentHandleBuilder('V.g')
def volume_gas_handle(handle, CAS, Tc, Pc, omega, eos):
    add_model = handle.add_model
    # no point in getting Tmin, Tmax
    if all((Tc, Pc, omega)):
        data = (Tc, Pc, omega)
        add_model(Tsonopoulos_extended.from_args(data))
        add_model(Tsonopoulos.from_args(data))
        add_model(Abbott.from_args(data))
        add_model(Pitzer_Curl.from_args(data))
    if CAS in V_data_CRC_virial:
        _, *data = V_data_CRC_virial[CAS]
        add_model(CRCVirial.from_args(data))
    add_model(ideal_gas_model)


# %% Solids

@functor(var='V.s')
def Goodman(T, Tt, V_l):
    r"""
    Create a functor of temperature (T; in K) that estimates the solid molar
    volume (V.s; in m^3/mol) of a chemical using the Goodman method, as 
    described in [12]_.
    
    Parameters
    ----------
    Tt : float
        Triple point temperature [K].
    V_l : float
        Liquid molar volume [-].
    
    Notes
    -----
    Calculates solid density at T using the simple relationship
    by a member of the DIPPR.

    The molar volume of a solid is given by:

    .. math::
        \frac{1}{V_m} = \left( 1.28 - 0.16 \frac{T}{T_t}\right)
        \frac{1}{{Vm}_L(T_t)}

    Works to the next solid transition temperature or to approximately 0.3Tt.

    Examples
    --------
    >>> f = Goodman(353.43, 7.6326)
    >>> f(281.46)
    8.797191839062899

    """
    return (1.28 - 0.16*(T/Tt))*V_l

@TPDependentHandleBuilder('V.s')
def volume_solid_handle(handle, CAS):
    if CAS in V_data_CRC_inorg_s_const:
        CRC_INORG_S_Vm = float(V_data_CRC_inorg_s_const.at[CAS, 'Vm'])
        handle.add_model(CRC_INORG_S_Vm, 0, 1e6, 0, 1e12)


volume_handle = PhaseTPHandleBuilder('V',
                                       volume_solid_handle, 
                                       volume_liquid_handle,
                                       volume_gas_handle)

