# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the vapor_pressure module from the chemicals's library:
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
import sys
from chemicals import vapor_pressure as vp
module = sys.modules[__name__]
sys.modules[__name__] = vp
del sys
from .row_data import RowData
from ..base import functor, TDependentHandleBuilder
from .dippr import DIPPR_EQ101
import numpy as np

Psat_data_WagnerMcGarry = RowData(vp.Psat_data_WagnerMcGarry.index, vp.Psat_values_WagnerMcGarry)
Psat_data_WagnerPoling = RowData(vp.Psat_data_WagnerPoling.index, vp.Psat_values_WagnerPoling)
Psat_data_AntoinePoling = RowData(vp.Psat_data_AntoinePoling.index, vp.Psat_values_AntoinePoling)
Psat_data_AntoineExtended = RowData(vp.Psat_data_AntoineExtended.index, vp.Psat_values_AntoineExtended)
Psat_data_Perrys2_8 = RowData(vp.Psat_data_Perrys2_8.index, vp.Psat_values_Perrys2_8)
Psat_data_VDI_PPDS_3 = RowData(vp.Psat_data_VDI_PPDS_3.index, vp.Psat_values_VDI_PPDS_3)

# %% Vapor pressure

vp.__all__.extend([
    'vapor_pressure_handle',
])

Antoine = functor(vp.Antoine, var='Psat')
TRC_Antoine_extended = functor(vp.TRC_Antoine_extended, var='Psat')
Wagner_original = functor(vp.Wagner_original, var='Psat')
Wagner = functor(vp.Wagner, var='Psat')
boiling_critical_relation = functor(vp.boiling_critical_relation , var='Psat')
Lee_Kesler = functor(vp.Lee_Kesler, var='Psat')
Ambrose_Walton = functor(vp.Ambrose_Walton, var='Psat')
Sanjari = functor(vp.Sanjari, var='Psat')
Edalat = functor(vp.Edalat, var='Psat')

@TDependentHandleBuilder('Psat')
def vapor_pressure_handle(handle, CAS, Tb, Tc, Pc, omega):
    add_model = handle.add_model
    if CAS in Psat_data_WagnerMcGarry:
        a, b, c, d, Pc, Tc, Tmin = Psat_data_WagnerMcGarry[CAS]
        Tmax = Tc
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner_original.functor.from_args(data), Tmin, Tmax)
    elif CAS in Psat_data_WagnerPoling:
        a, b, c, d, Tc, Pc, Tmin, Tmax = Psat_data_WagnerPoling[CAS]
        # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
        if np.isnan(Tmin): Tmin = Tmax * 0.1
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner_original.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_AntoineExtended:
        a, b, c, Tc, to, n, e, f, Tmin, Tmax = Psat_data_AntoineExtended[CAS]
        data = (Tc, to, a, b, c, n, e, f)
        add_model(TRC_Antoine_extended.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_AntoinePoling:
        a, b, c, Tmin, Tmax = Psat_data_AntoinePoling[CAS]
        data = (a, b, c)
        add_model(Antoine.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_Perrys2_8:
        C1, C2, C3, C4, C5, Tmin, Tmax = Psat_data_Perrys2_8[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(DIPPR_EQ101.functor.from_args(data), Tmin, Tmax)
    if CAS in Psat_data_VDI_PPDS_3:
        Tm, Tc, Pc, a, b, c, d = Psat_data_VDI_PPDS_3[CAS]
        data = (Tc, Pc, a, b, c, d)
        add_model(Wagner.functor.from_args(data), 0., Tc,)
    data = (Tb, Tc, Pc)
    if all(data):
        add_model(boiling_critical_relation.functor.from_args(data), 0., Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for f in (Lee_Kesler, Ambrose_Walton, Sanjari, Edalat):
            add_model(f.functor.from_args(data), 0., Tc)
vp.vapor_pressure_handle = vapor_pressure_handle