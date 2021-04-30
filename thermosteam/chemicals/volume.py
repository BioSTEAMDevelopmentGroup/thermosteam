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
from ..utils import forward
from chemicals import volume as vol
from chemicals.virial import (
    BVirial_Pitzer_Curl, 
    BVirial_Abbott, 
    BVirial_Tsonopoulos, 
    BVirial_Tsonopoulos_extended
)
from chemicals.dippr import EQ105
# from .electrochem import _Laliberte_Density_ParametersDict, Laliberte_Density
from ..base import (functor, InterpolatedTDependentModel, 
                    TPDependentModel, TPDependentHandleBuilder,
                    PhaseTPHandleBuilder)
from .data import (
    VDI_saturation_dict,
    lookup_VDI_tabular_data,    
    rho_data_COSTALD, 
    rho_data_SNM0,
    rho_data_Perry_8E_105_l, 
    rho_data_VDI_PPDS_2, 
    rho_data_CRC_inorg_l,
    rho_data_CRC_inorg_l_const,
    rho_data_CRC_inorg_s_const, 
    rho_data_CRC_virial
)

vol.__all__.extend([
    'volume_handle',
    'volume_solid_handle',
    'volume_liquid_handle',
    'volume_gas_handle',
])



# %% Liquids

Yen_Woods_saturation = functor(vol.Yen_Woods_saturation, 'V.l')
Rackett = functor(vol.Rackett, 'V.l')
Yamada_Gunn = functor(vol.Yamada_Gunn, 'V.l')
Townsend_Hales = functor(vol.Townsend_Hales, 'V.l')
Bhirud_normal = functor(vol.Bhirud_normal, 'V.l')
COSTALD = functor(vol.COSTALD, 'V.l')
Campbell_Thodos = functor(vol.Campbell_Thodos, 'V.l')
SNM0 = functor(vol.SNM0, 'V.l')
CRC_inorganic = functor(vol.CRC_inorganic, 'V.l')
volume_VDI_PPDS = functor(vol.volume_VDI_PPDS , 'V.l')
COSTALD_compressed = functor(vol.COSTALD_compressed , 'V.l')

def EQ105_hook(self, T, kwargs):
    if kwargs.get('molar_volume'):
        del kwargs['molar_volume']
        V_inv = self.function(T, **kwargs)
        value = 1. / V_inv
        kwargs['molar_volume'] = True
    else:
        value = self.function(T, **kwargs)
    return value
EQ105.functor.hook = EQ105_hook

def COSTALD_compressed_hook(self, T, P, kwargs):
    kwargs = kwargs.copy()
    Psat = kwargs['Psat']
    Vs = kwargs['Vs']
    if callable(Psat): kwargs['Psat'] = Psat(T)
    if callable(Vs): kwargs['Vs'] = Vs(T, P)
    return self.function(T, P, **kwargs)
COSTALD_compressed.functor.hook = COSTALD_compressed_hook

@TPDependentHandleBuilder('V.l')
def volume_liquid_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc,
                         omega, Psat, eos, dipole, has_hydroxyl):
    all_ = all
    add_model = handle.add_model
    if CAS in rho_data_VDI_PPDS_2:
        MW, Tc_, rhoc, A, B, C, D = rho_data_VDI_PPDS_2[CAS]
        data = (Tc, rhoc, A, B, C, D, MW)
        add_model(volume_VDI_PPDS.functor.from_args(data), 0., Tc_)
    if CAS in rho_data_Perry_8E_105_l:
        C1, C2, C3, C4, Tmin, Tmax = rho_data_Perry_8E_105_l[CAS]
        data = (C1, C2, C3, C4, 0)
        f = EQ105.functor.from_args(data)
        f.molar_volume = True
        add_model(f, Tmin, Tmax)
    if CAS in VDI_saturation_dict:
        Ts, Vls = lookup_VDI_tabular_data(CAS, 'Volume (l)')
        model = InterpolatedTDependentModel(Ts, Vls,
                                            Tmin=Ts[0], Tmax=Ts[-1])
        add_model(model)
    data = (Tb, Tc, Pc, MW, dipole, has_hydroxyl)
    if all_([i is not None for i in data]):
        add_model(Campbell_Thodos.functor.from_args(data), 0, Tc)
    if CAS in rho_data_CRC_inorg_l:
        MW, rho, k, Tm, Tmax = rho_data_CRC_inorg_l[CAS]
        data = (MW, rho, k, Tm)
        add_model(CRC_inorganic.functor.from_args(data), Tm, Tmax)
    data = (Tc, Vc, Zc)
    if all_(data):
        add_model(Yen_Woods_saturation.functor.from_args(data), 0, Tc)
    data = (Tc, Pc, Zc)
    if all_(data):
        add_model(Rackett.functor.from_args(data), 0, Tc)
    data = (Tc, Pc, omega)
    if all_(data):
        add_model(Yamada_Gunn.functor.from_args(data), 0, Tc)
        add_model(Bhirud_normal.functor.from_args(data), 0, Tc)
    data = (Tc, Vc, omega)
    if all_(data):
        add_model(Townsend_Hales.functor.from_args(data), 0, Tc)
        if CAS in rho_data_SNM0:
            SNM0_delta_SRK = float(rho_data_SNM0.get(CAS, 'delta_SRK'))
            data = (Tc, Vc, omega, SNM0_delta_SRK)
            add_model(SNM0.functor.from_args(data), 0, Tc)
    if CAS in rho_data_CRC_inorg_l_const:
        Vl = float(rho_data_CRC_inorg_l_const.get(CAS, 'Vm'))
        add_model(Vl, 0., Tc, name="CRC inorganic liquid constant")
    if Tc and Pc and CAS in rho_data_COSTALD:
        Zc_ = rho_data_COSTALD.get(CAS, 'Z_RA')
        if not np.isnan(Zc_): Zc_ = float(Zc_)
        data = (Tc, Pc, Zc_)
        add_model(Rackett.functor.from_args(data), 0., Tc)
        # Roughly data at STP; not guaranteed however; not used for Trange
    data = (Tc, Vc, omega)
    if all_(data) and CAS in rho_data_COSTALD:
        add_model(COSTALD.functor.from_args(data), 0, Tc)
    data = (Tc, Pc, omega)
    if all_(data):
        data = (Psat, Tc, Pc, omega, handle)
        add_model(COSTALD_compressed.functor.from_args(data), 50, 500)
vol.volume_liquid_handle = volume_liquid_handle

# %% Gases

ideal_gas = vol.ideal_gas

ideal_gas_model = TPDependentModel(vol.ideal_gas,
                                   0, 10e6,
                                   0, 10e12,
                                   var='V.g')

@forward(vol)
@functor(var='V.g')
def Tsonopoulos_extended(T, P, Tc, Pc, omega, a=0, b=0,
                         species_type='', dipole=0, order=0):
    return ideal_gas(T, P) + BVirial_Tsonopoulos_extended(T, Tc, Pc, omega, a, b,
                                                          species_type, dipole, order)

@forward(vol)
@functor(var='V.g') 
def Tsonopoulos(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Tsonopoulos(T, Tc, Pc, omega)

@forward(vol)
@functor(var='V.g')
def Abbott(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Abbott(T, Tc, Pc, omega)

@forward(vol)
@functor(var='V.g')
def Pitzer_Curl(T, P, Tc, Pc, omega):
    return ideal_gas(T, P) + BVirial_Pitzer_Curl(T, Tc, Pc, omega)
    
@forward(vol)
@functor(var='V.g')
def CRCVirial(T, P, a1, a2, a3, a4, a5):
    t = 298.15/T - 1.
    return ideal_gas(T, P) + (a1 + t*(a2 + t*(a3 + t*(a4 + a5*t))))/1e6

@TPDependentHandleBuilder('V.g')
def volume_gas_handle(handle, CAS, Tc, Pc, omega, eos):
    add_model = handle.add_model
    # no point in getting Tmin, Tmax
    if all((Tc, Pc, omega)):
        data = (Tc, Pc, omega)
        add_model(Tsonopoulos_extended.functor.from_args(data))
        add_model(Tsonopoulos.functor.from_args(data))
        add_model(Abbott.functor.from_args(data))
        add_model(Pitzer_Curl.functor.from_args(data))
    if CAS in rho_data_CRC_virial:
        add_model(CRCVirial.functor.from_args(rho_data_CRC_virial[CAS]))
    add_model(ideal_gas_model)
vol.volume_gas_handle = volume_gas_handle

# %% Solids

Goodman = functor(vol.Goodman, 'V.s')

# TODO: Make a data table for new additions (not just for common sugars).
# Source: PubChem
sugar_solid_densities = {
    '50-99-7': 0.0001125975, 
    '57-48-7': 0.0001063495,
    '3458-28-4': 0.00011698,
    '25990-60-7': 9.84459e-05,
    '59-23-4': 0.000120104,   
}

@TPDependentHandleBuilder('V.s')
def volume_solid_handle(handle, CAS):
    if CAS in rho_data_CRC_inorg_s_const:
        CRC_INORG_S_Vm = float(rho_data_CRC_inorg_s_const.get(CAS, 'Vm'))
        handle.add_model(CRC_INORG_S_Vm, 0, 1e6, 0, 1e12)
    elif CAS in sugar_solid_densities: 
        handle.add_model(sugar_solid_densities[CAS], 0, 1e6, 0, 1e12)
vol.volume_solid_handle = volume_solid_handle

vol.volume_handle = PhaseTPHandleBuilder('V',
                                     volume_solid_handle, 
                                     volume_liquid_handle,
                                     volume_gas_handle)

