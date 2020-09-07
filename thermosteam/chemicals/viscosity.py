# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# A significant portion of this module originates from:
# Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
# 
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/thermo/blob/master/LICENSE.txt for details.
from ..functional import horner
from ..base import (InterpolatedTDependentModel, 
                    TPDependentHandleBuilder, 
                    TDependentModel, 
                    PhaseTPHandleBuilder,
                    functor)
from .data import (VDI_saturation_dict,
                   lookup_VDI_tabular_data,
                   mu_data_Dutt_Prasad,
                   mu_data_VN3,
                   mu_data_VN2,
                   mu_data_Perrys_8E_2_313,
                   mu_data_Perrys_8E_2_312,
                   mu_data_VDI_PPDS_7,
                   mu_data_VDI_PPDS_8,
)
from .thermal_conductivity import IAPWS_rho_hook
from chemicals import viscosity
from chemicals.dippr import EQ101, EQ102


mul = 'mu.l'
mu_IAPWS = functor(viscosity.mu_IAPWS, mul)
mu_IAPWS.functor.hook = IAPWS_rho_hook
Viswanath_Natarajan_2 = functor(viscosity.Viswanath_Natarajan_2, mul)
Viswanath_Natarajan_3 = functor(viscosity.Viswanath_Natarajan_3, mul)
Letsou_Stiel = functor(viscosity.Letsou_Stiel, mul)
Przedziecki_Sridhar = functor(viscosity.Przedziecki_Sridhar, mul)
PPDS9 = functor(viscosity.PPDS9, mul)
Lucas = functor(viscosity.Lucas, mul)

def Przedziecki_Sridhar_hook(f, T, kwargs):
    Vm = kwargs['Vm']
    if callable(Vm):
        kwargs = kwargs.copy()
        kwargs['Vm'] = Vm.at_T(T)
    return f(T, **kwargs)
Przedziecki_Sridhar.functor.hook = Przedziecki_Sridhar_hook

def Lucas_hook(f, T, P, kwargs):
    kwargs = kwargs.copy()
    Psat = kwargs['Psat']
    mu_l = kwargs['mu_l']
    if callable(Psat):
        kwargs['Psat'] = Psat(T)
    if callable(mu_l):
        kwargs['mu_l'] = mu_l.at_T(T)
    return kwargs
Lucas.hook = Lucas_hook

@TPDependentHandleBuilder('mu.l')
def viscosity_liquid_handle(handle, CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, Vl):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'Mu (l)')
        model = InterpolatedTDependentModel(Ts, Ys, Ts[0], Ts[-1], 
                                            name='VDI-interpolated')
        add_model(model)
    if CAS in mu_data_Dutt_Prasad:
        A, B, C, Tmin, Tmax = mu_data_Dutt_Prasad[CAS]
        data = (A, B, C)
        add_model(Viswanath_Natarajan_3.functor.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VN3:
        A, B, C, Tmin, Tmax = mu_data_VN3[CAS]
        data = (A, B, C)
        add_model(Viswanath_Natarajan_3.functor.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VN2:
        A, B, Tmin, Tmax = mu_data_VN2[CAS]
        data = (A, B)
        add_model(Viswanath_Natarajan_2.functor.from_args(data), Tmin ,Tmax)
    if CAS in mu_data_Perrys_8E_2_313:
        C1, C2, C3, C4, C5, Tmin, Tmax = mu_data_Perrys_8E_2_313[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(EQ101.functor.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VDI_PPDS_7:
        add_model(PPDS9.functor.from_args(mu_data_VDI_PPDS_7[CAS]))
    data = (MW, Tc, Pc, omega)
    if all(data):
        add_model(Letsou_Stiel.functor.from_args(data), Tc/4, Tc)
    data = (MW, Tm, Tc, Pc, Vc, omega, Vl)
    if all(data):
        add_model(Przedziecki_Sridhar.functor.from_args(data), Tm, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for mu_l in handle.models:
            if isinstance(mu_l, TDependentModel): break
        data = (Tc, Pc, omega, Psat, mu_l)
        add_model(Lucas.functor.from_args(data), Tm, Tc,
                  name='Lucas')
viscosity.viscosity_liquid_handle = viscosity_liquid_handle

mug = 'mu.g'
Yoon_Thodos = functor(viscosity.Yoon_Thodos, mug)
Stiel_Thodos = functor(viscosity.Stiel_Thodos, mug)
Lucas_gas = functor(viscosity.Lucas_gas, mug)
viscosity_gas_Gharagheizi = functor(viscosity.viscosity_gas_Gharagheizi, mug)

@TPDependentHandleBuilder('mu.g')
def viscosity_gas_handle(handle, CAS, MW, Tc, Pc, Zc, dipole):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'Mu (g)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        model = InterpolatedTDependentModel(Ts, Ys, Tmin, Tmax,
                                            name='VDI-interpolated')
        add_model(model)
    if CAS in mu_data_Perrys_8E_2_312:
        C1, C2, C3, C4, Tmin, Tmax = mu_data_Perrys_8E_2_312[CAS]
        data = (C1, C2, C3, C4)
        add_model(EQ102.functor.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VDI_PPDS_8:
        data = mu_data_VDI_PPDS_8[CAS].tolist()
        data.reverse()
        add_model(horner.functor.from_kwargs({'coeffs':data}),
                  name='VDI-PPDS')
    # data = (Tc, Pc, Zc, MW)
    # if all(data):
    #     Tmin = 0; Tmax = 1e3
    #     add_model(Lucas_gas.from_args(data), Tmin, Tmax,
    #               name='Lucas')
    data = (Tc, Pc, MW)
    if all(data):
        Tmin = 0; Tmax = 5e3
        add_model(viscosity_gas_Gharagheizi.functor.from_args(data), Tmin, Tmax,
                  name='Gharagheizi')
        add_model(Yoon_Thodos.functor.from_args(data), Tmin, Tmax)
        add_model(Stiel_Thodos.functor.from_args(data), Tmin, Tmax)
        # Intelligently set limit
        # GHARAGHEIZI turns nonsensical at ~15 K, YOON_THODOS fine to 0 K,
        # same as STIEL_THODOS
viscosity.viscosity_gas_handle = viscosity_gas_handle

viscosity.viscosity_handle = PhaseTPHandleBuilder('mu', None,
                                                  viscosity_liquid_handle,
                                                  viscosity_gas_handle)
