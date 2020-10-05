# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the phase_change module from the chemicals's library:
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
from chemicals import phase_change as pc
import numpy as np
from ..base import InterpolatedTDependentModel, TDependentHandleBuilder, functor
from .. import functional as fn
from chemicals.dippr import EQ106
from .data import (phase_change_data_Perrys2_150,
                   phase_change_data_VDI_PPDS_4,
                   VDI_saturation_dict,
                   phase_change_data_Alibakhshi_Cs,
                   lookup_VDI_tabular_data,
                   Hvap_data_CRC,
                   Hvap_data_Gharagheizi,
)

### Enthalpy of Vaporization at T

Clapeyron = functor(pc.Clapeyron, 'Hvap')
Pitzer = functor(pc.Pitzer, 'Hvap')
SMK = functor(pc.SMK, 'Hvap')
MK = functor(pc.MK, 'Hvap')
Velasco = functor(pc.Velasco, 'Hvap')
Watson = functor(pc.Watson, 'Hvap')
Alibakhshi = functor(pc.Alibakhshi, 'Hvap')
PPDS12 = functor(pc.PPDS12, 'Hvap')

def Clapeyron_hook(self, T, kwargs):
    kwargs = kwargs.copy()
    Psat = kwargs['Psat']
    if callable(Psat): kwargs['Psat'] = Psat = Psat(T)
    if 'V' in kwargs: 
        # Use molar volume to compute dZ if possible
        V = kwargs.pop('V')
        kwargs['dZ'] = fn.Z(T, Psat, V.g(T, Psat) - V.l(T, Psat))
    return self.function(T, **kwargs)
Clapeyron.functor.hook = Clapeyron_hook

@TDependentHandleBuilder('Hvap')
def heat_of_vaporization_handle(handle, CAS, Tb, Tc, Pc, omega, 
                                similarity_variable, Psat, V):
    # if has_CoolProp and self.CASRN in coolprop_dict:
    #     methods.append(COOLPROP)
    #     self.CP_f = coolprop_fluids[self.CASRN]
    #     Tmins.append(self.CP_f.Tt); Tmaxs.append(self.CP_f.Tc)
    add_model = handle.add_model
    if CAS in phase_change_data_Perrys2_150:
        Tc, C1, C2, C3, C4, Tmin, Tmax = phase_change_data_Perrys2_150[CAS]
        data = (Tc, C1, C2, C3, C4)
        add_model(EQ106.functor.from_args(data), Tmin, Tmax)
    if CAS in phase_change_data_VDI_PPDS_4:
        Tc, A, B, C, D, E = phase_change_data_VDI_PPDS_4[CAS]
        add_model(PPDS12.functor.from_args(data=(Tc, A, B, C, D, E)), 0, Tc)
    if all((Tc, Pc)):
        model = Clapeyron.functor.from_args(data=(Tc, Pc, None, Psat))
        model.V = V
        add_model(model, 0, Tc)
    data = (Tc, omega)
    if all(data):
        for f in (MK, SMK, Velasco, Pitzer):
            add_model(f.functor.from_args(data), 0, Tc)
    if CAS in VDI_saturation_dict:
        Ts, Hvaps = lookup_VDI_tabular_data(CAS, 'Hvap')
        add_model(InterpolatedTDependentModel(Ts, Hvaps, Ts[0], Ts[-1]))
    if Tc:
        if CAS in phase_change_data_Alibakhshi_Cs:
            C = float(phase_change_data_Alibakhshi_Cs.get(CAS, 'C'))
            add_model(Alibakhshi.functor.from_args(data=(Tc, C)), 0, Tc)
        if CAS in Hvap_data_CRC:
            Hvap = float(Hvap_data_CRC.get(CAS, 'HvapTb'))
            if not np.isnan(Hvap):
                Tb = float(Hvap_data_CRC.get(CAS, 'Tb'))
                data = dict(Hvap_ref=Hvap, T_ref=Tb, Tc=Tc, exponent=0.38)
                add_model(Watson.functor.from_kwargs(data), 0, Tc)
            Hvap = float(Hvap_data_CRC.get(CAS, 'Hvap298'))
            if not np.isnan(Hvap):
                data = dict(Hvap_ref=Hvap, T_ref=298., Tc=Tc, exponent=0.38)
                add_model(Watson.functor.from_kwargs(data), 0, Tc)
        if CAS in Hvap_data_Gharagheizi:
            Hvap = float(Hvap_data_Gharagheizi.get(CAS, 'Hvap298'))
            data = dict(Hvap_ref=Hvap, T_ref=298., Tc=Tc, exponent=0.38)
            add_model(Watson.functor.from_kwargs(data), 0, Tc)
    data = (Tb, Tc, Pc)
    if all(data) and Tc > Tb:
        for f in (pc.Riedel, pc.Chen, pc.Vetere, pc.Liu):
            add_model(f(*data), 0, Tc)
pc.heat_of_vaporization_handle = heat_of_vaporization_handle