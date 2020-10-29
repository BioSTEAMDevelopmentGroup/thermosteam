# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the thermal_conductivity module from the chemicals's library:
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
import flexsolve as flx
from chemicals import thermal_conductivity as tc
from scipy.interpolate import interp2d
from ..base import (InterpolatedTDependentModel, 
                    ThermoModelHandle,
                    TPDependentHandleBuilder, 
                    PhaseTPHandleBuilder, 
                    functor)
from ..constants import R
from math import log, exp
from ..functional import horner
from .data import (
    VDI_saturation_dict,
    lookup_VDI_tabular_data,
    k_data_Perrys_8E_2_315,  
    k_data_VDI_PPDS_9,
)
from chemicals.dippr import EQ100

kl = 'kappa.l'
k_IAPWS = functor(tc.k_IAPWS, kl)
Sheffy_Johnson = functor(tc.Sheffy_Johnson, kl)
Sato_Riedel = functor(tc.Sato_Riedel, kl)
Lakshmi_Prasad = functor(tc.Lakshmi_Prasad, kl)
Gharagheizi_liquid = functor(tc.Gharagheizi_liquid, kl)
Nicola = functor(tc.Nicola, kl)
Bahadori_liquid = functor(tc.Bahadori_liquid, kl)
kl_Mersmann_Kind = functor(tc.kl_Mersmann_Kind, kl)
DIPPR9G = functor(tc.DIPPR9G, kl)
Missenard = functor(tc.Missenard, kl)

def IAPWS_rho_hook(self, T, kwargs):
    if 'Vl' in kwargs:
        kwargs = kwargs.copy()
        Vl = kwargs.pop('Vl')
        kwargs['rho'] = 0.01801528 / Vl.at_T(T)
    return self.function(T, **kwargs)
k_IAPWS.functor.hook = IAPWS_rho_hook

def Tmax_Lakshmi_Prasad(MW):
    """
    Returns the maximum temperature at which the Lakshmi Prasad method is
    valid.
    """
    T_max = flx.aitken_secant(Lakshmi_Prasad, 298.15, args=(MW,),
                              checkroot=False)
    return T_max - 10 # As an extra precaution

def kl_hook(self, T, P, kwargs):
    kl = kwargs['kl']
    if isinstance(kl, ThermoModelHandle): 
        kl = kl.at_T(T)
        kwargs = kwargs.copy()
        kwargs['kl'] = kl
    return self.function(T, P, **kwargs)
DIPPR9G.functor.hook = kl_hook
Missenard.functor.hook = kl_hook

@TPDependentHandleBuilder('kappa.l')
def thermal_conductivity_liquid_handle(handle, CAS, MW, Tm, Tb, Tc, Pc, omega, Vl):
    add_model = handle.add_model
    if CAS == '7732-18-5':
        add_model(k_IAPWS.functor(Vl=Vl))
    if all((Tc, Pc)):
        data = (Tc, Pc, handle)
        add_model(DIPPR9G.functor.from_args(data))
        add_model(Missenard.functor.from_args(data))
    if CAS in k_data_Perrys_8E_2_315:
        C1, C2, C3, C4, C5, Tmin, Tmax = k_data_Perrys_8E_2_315[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(EQ100.functor.from_args(data), Tmin, Tmax)
    if CAS in k_data_VDI_PPDS_9:
        A, B, C, D, E = k_data_VDI_PPDS_9[CAS]
        add_model(horner.functor.from_kwargs({'coeffs':(E, D, C, B, A)}))
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'K (l)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        add_model(InterpolatedTDependentModel(Ts, Ys, Tmin=Tmin, Tmax=Tmax))
    data = (MW, Tb, Pc, omega)
    if all(data):
        add_model(Gharagheizi_liquid.functor.from_args(data), Tb, Tc)
    data = (MW, Tc, Pc, omega)
    if all(data):
        add_model(Nicola.functor.from_args(data))
    data = (MW, Tb, Tc)
    if all(data):
        add_model(Sato_Riedel.functor.from_args(data))
    data = (MW, Tm)
    if all(data):
        # Works down to 0, has a nice limit at T = Tm+793.65 from Sympy
        add_model(Sheffy_Johnson.functor.from_args(data), 0, 793.65)
    data = (MW,)
    if MW:
        Tmax = Tmax_Lakshmi_Prasad(MW)
        add_model(Lakshmi_Prasad.functor.from_args(data), 0., Tmax)
        add_model(Bahadori_liquid.functor.from_args(data))

### Thermal Conductivity of Gases

kg = 'kappa.g'
Chung = functor(tc.Chung, kg)
Eli_Hanley = functor(tc.Eli_Hanley, kg)
Gharagheizi_gas = functor(tc.Gharagheizi_gas, kg)
Bahadori_gas = functor(tc.Bahadori_gas, kg)
Stiel_Thodos_dense = functor(tc.Stiel_Thodos_dense, kg)
Eli_Hanley_dense = functor(tc.Eli_Hanley_dense, kg)

@functor(var=kg)
def Eucken_2(T, MW, Cn, mu):
    return tc.Eucken(MW, Cn(T) - R, mu.at_T(T))
tc.Eucken_2 = Eucken_2

@functor(var=kg)
def Eucken_modified_2(T, MW, Cn, mu):
    return tc.Eucken_modified(MW, Cn(T) - R, mu.at_T(T))
tc.Eucken_modified_2 = Eucken_modified_2

@functor(var=kg)
def Chung_dense_2(T, P, MW, Tc, Vc, omega, Cn, Vm, mu, dipole, association=0.0):
    return tc.Chung_dense(T, MW, Tc, Vc, omega, Cn(T) - R, Vm(T, P), mu(T, P), dipole, association)
tc.Chung_dense_2 = Chung_dense_2

def kg_hook_T(f, T, kwargs):
    kwargs = kwargs.copy()
    if 'Cn' in kwargs:
        Cn = kwargs.pop('Cn')
        if callable(Cn): Cn = Cn(T)
        kwargs['Cvm'] = Cn - R
    if 'mu' in kwargs:
        mu = kwargs['mu']
        if callable(mu):
            kwargs['mu'] = mu.at_T(T)
    return f(T, **kwargs)
Chung.functor.hook = kg_hook_T
Eli_Hanley.functor.hook = kg_hook_T

def kg_hook_TP(f, T, P, kwargs):
    kwargs = kwargs.copy()
    if 'Cn' in kwargs:
        Cn = kwargs.pop('Cn')
        if callable(Cn): Cn = Cn(T)
        kwargs['Cvm'] = Cn - R
    for var in ('Vm', 'kg', 'mu'):
        if var in kwargs:
            obj = kwargs[var]
            if callable(obj):
                kwargs[var] = obj.at_T(T)
    return f(T, **kwargs)
Stiel_Thodos_dense.functor.hook = kg_hook_TP
Eli_Hanley_dense.functor.hook = kg_hook_TP

@TPDependentHandleBuilder('kappa.g')
def thermal_conductivity_gas_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, Vg, Cn, mug):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'K (g)')
        add_model(InterpolatedTDependentModel(Ts, Ys))
    # if CAS in k_data_VDI_PPDS_10:
    #     _,  *data = k_data_VDI_PPDS_10[CAS].tolist()
    #     data.reverse()
    #     add_model(horner_polynomial.from_kwargs({'coeffs': data}))
    data = (MW, Tb, Pc, omega)
    if all(data):
        add_model(Gharagheizi_gas.functor.from_args(data))   
    data = (MW, Tc, omega, Cn, mug)
    if all(data):
        add_model(Chung.functor.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cn)
    if all(data):
        add_model(Eli_Hanley.functor.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cn, Vg)
    if all(data):
        add_model(Eli_Hanley_dense.functor.from_args(data))
    data = (MW, Tc, Vc, omega, Cn, Vg, mug, dipole)
    if all(data):
        add_model(Chung_dense_2.functor.from_args(data))
    data = (MW, Tc, Pc, Vc, Zc, Vg, handle)
    if all(data):
        add_model(Stiel_Thodos_dense.functor.from_args(data))
    data = (MW, Cn, mug)
    if all(data):
        add_model(Eucken_modified_2.functor.from_args(data))
        add_model(Eucken_2.functor.from_args(data))
    # TODO: Fix propblem with values
    # if CAS in k_data_Perrys2_314:
    #     _, *data, Tmin, Tmax = k_data_Perrys2_314[CAS]
    #     add_model(DIPPR9B_linear(data), Tmin, Tmax)

tc.thermal_conductivity_handle = PhaseTPHandleBuilder('kappa', None,
                                                      thermal_conductivity_liquid_handle,
                                                      thermal_conductivity_gas_handle)
