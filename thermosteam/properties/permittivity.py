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
from .data import (
    permittivity_data_CRC,
)
from ..base import TDependentHandleBuilder, functor
from chemicals import permittivity

permittivity_CRC = functor(permittivity.permittivity_CRC, 'epsilon')
permittivity_IAPWS = functor(permittivity.permittivity_IAPWS, 'epsilon')

@permittivity_IAPWS.functor.set_hook
def hook(f, T, kwargs):
    kwargs = kwargs.copy()
    rho = kwargs['rho']
    if callable(rho): # Assume its a liquid molar volume handle
        kwargs['rho'] = 0.018015268 / rho(T)
    return f(T, **kwargs)

@TDependentHandleBuilder('epsilon')
def permitivity_handle(handle, CAS, Vl):
    add_model = handle.add_model
    if Vl and CAS == '7732-18-5':
        add_model(permittivity_IAPWS.functor.from_args((Vl,)))
    if CAS in permittivity_data_CRC:
        CRC_CONSTANT_T, CRC_permittivity, *coeffs, Tmin, Tmax = permittivity_data_CRC[CAS]
        args = [0 if np.isnan(x) else x for x in coeffs]
        Tmin = 0 if np.isnan(Tmin) else Tmin
        Tmax = 1e6 if np.isnan(Tmax) else Tmax
        add_model(permittivity_CRC.functor.from_args(args), Tmin, Tmax, name='CRC')
        add_model(CRC_permittivity, Tmin, Tmax, name='CRC_constant')
permittivity.permitivity_handle = permitivity_handle
