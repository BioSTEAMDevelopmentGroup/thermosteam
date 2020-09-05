# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the interface module from the chemicals's library:
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
from chemicals import interface
from ..base import functor, TDependentHandleBuilder, InterpolatedTDependentModel
from .data import (
    sigma_data_Mulero_Cachadina,
    sigma_data_Jasper_Lange,
    sigma_data_Somayajulu,
    sigma_data_Somayajulu2,
    sigma_data_VDI_PPDS_11,
    VDI_saturation_dict,
    lookup_VDI_tabular_data,
)
from .dippr import DIPPR_EQ106

interface.__all__.extend([
    'surface_tension_handle',
])


### Regressed coefficient-based functions

REFPROP_sigma = functor(interface.REFPROP_sigma, 'sigma')
Somayajulu = functor(interface.Somayajulu, 'sigma')
Jasper = functor(interface.Jasper, 'sigma')
Brock_Bird = functor(interface.Brock_Bird, 'sigma')
Pitzer_sigma = functor(interface.Pitzer_sigma, 'sigma')
Sastri_Rao = functor(interface.Sastri_Rao, 'sigma')
Zuo_Stenby = functor(interface.Zuo_Stenby, 'sigma')
Hakim_Steinberg_Stiel = functor(interface.Hakim_Steinberg_Stiel, 'sigma')
Miqueu = functor(interface.Miqueu, 'sigma')
Mersmann_Kind_sigma = functor(interface.Mersmann_Kind_sigma, 'sigma')

@TDependentHandleBuilder('sigma')
def surface_tension_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, StielPolar):
    add_model = handle.add_model
    if CAS in sigma_data_Mulero_Cachadina:
        _, sigma0, n0, sigma1, n1, sigma2, n2, Tc, Tmin, Tmax = sigma_data_Mulero_Cachadina[CAS]
        STREFPROP_sigma_coeffs = (Tc, sigma0, n0, sigma1, n1, sigma2, n2)
        add_model(REFPROP_sigma.functor.from_args(STREFPROP_sigma_coeffs), Tmin, Tmax)
    if CAS in sigma_data_Somayajulu2:
        _, Tt, Tc, a, b, c = sigma_data_Somayajulu2[CAS]
        SOMAYAJULU2_coeffs = (Tc, a, b, c)
        Tmin = Tt; Tmax = Tc
        add_model(Somayajulu.functor.from_args(SOMAYAJULU2_coeffs), Tmin, Tmax)
    elif CAS in sigma_data_Somayajulu:
        _, Tt, Tc, a, b, c = sigma_data_Somayajulu[CAS]
        SOMAYAJULU_coeffs = (Tc, a, b, c)
        Tmin = Tt; Tmax = Tc
        add_model(Somayajulu.functor.from_args(SOMAYAJULU_coeffs), Tmin, Tmax)
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'sigma')
        Tmin = Ts[0]
        *Ts, Tmax = Ts
        Ys = Ys[:-1]
        add_model(InterpolatedTDependentModel(Ts, Ys), Tmin, Tmax)
    if CAS in sigma_data_Jasper_Lange:
        _, a, b, Tmin, Tmax= sigma_data_Jasper_Lange[CAS]
        JASPER_coeffs = (a, b)
        add_model(Jasper.functor.from_args(JASPER_coeffs))
    data = (Tc, Vc, omega)
    if all(data):
        add_model(Miqueu.functor.from_args(data), 0.0, Tc)
    data = (Tb, Tc, Pc)
    if all(data):
        add_model(Brock_Bird.functor.from_args(data), 0.0, Tc)
        add_model(Sastri_Rao.functor.from_args(data), 0.0, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        add_model(Pitzer_sigma.functor.from_args(data), 0.0, Tc)
        add_model(Zuo_Stenby.functor.from_args(data), 0.0, Tc)
    if CAS in sigma_data_VDI_PPDS_11:
        _, Tm, Tc, a, b, c, d, e = sigma_data_VDI_PPDS_11[CAS]
        VDI_PPDS_coeffs = (Tc, a, b, c, d, e)
        add_model(DIPPR_EQ106.functor.from_args(VDI_PPDS_coeffs))
interface.surface_tension_handle = surface_tension_handle