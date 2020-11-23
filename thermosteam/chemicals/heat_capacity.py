# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the heat_capacity module from the chemicals's library:
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
from chemicals import heat_capacity as hc
from math import log
import numpy as np
from ..utils import forward
from ..base import (InterpolatedTDependentModel,
                    TDependentHandleBuilder, 
                    PhaseTHandleBuilder, 
                    thermo_model, 
                    functor)
from .data import (
    VDI_saturation_dict, 
    lookup_VDI_tabular_data,
    Cp_data_Poling,
    TRC_gas_data,
    CRC_standard_data,
    Cp_dict_PerryI,
    zabransky_dict_sat_s,
    zabransky_dict_sat_p,
    zabransky_dict_const_s,
    zabransky_dict_const_p,
    zabransky_dict_iso_s,
    zabransky_dict_iso_p,
)
from ..constants import calorie

hc.__all__.extend([
    'heat_capacity_handle',
    
    'Lastovka_Shaw_definite_integral',
    'Lastovka_Shaw_definite_integral_over_T',

    'TRCCp_definite_integral', 
    'TRCCp_definite_integral_over_T',

    'Poling_definite_integral', 
    'Poling_definite_integral_over_T',

    'Zabransky_cubic_definite_integral', 
    'Zabransky_cubic_definite_integral_over_T',

    'Zabransky_quasi_polynomial_definite_integral', 
    'Zabransky_quasi_polynomial_definite_integral_over_T',
 
    'Dadgostar_Shaw_definite_integral', 
    'Dadgostar_Shaw_definite_integral_over_T',

    'Lastovka_solid_definite_integral', 
    'Lastovka_solid_definite_integral_over_T',
    
    'Perry_151_definite_integral', 
    'Perry_151_definite_integral_over_T'
])


# %% Utilities

def CnHS(FCn, FH, FS, data):
    fCn = FCn.from_args(data)
    return {'evaluate': fCn,
            'integrate_by_T': FH.from_other(fCn),
            'integrate_by_T_over_T': FS.from_other(fCn)}

def CnHSModel(FCn, FH, FS, data=None, Tmin=None, Tmax=None, name=None):
    funcs = CnHS(FCn, FH, FS, data)
    return thermo_model(Tmin=Tmin, Tmax=Tmax, name=name, var='Cn', **funcs)

class ZabranskyModelBuilder:
    __slots__ = ('name', 'casdata', 'funcs', 'many')
    
    def __init__(self, name, casdata, funcs, many=False):
        self.name = name
        self.casdata = casdata
        self.funcs = funcs
        self.many = many
    
    def add_model(self, CAS, models):
        if CAS in self.casdata:
            if self.many:
                models.extend([self.build_model(i) for i in self.casdata[CAS]])
            else:
                models.append(self.build_model(self.casdata[CAS]))
    
    def build_model(self, data):
        try:
            coeffs = (data.Tc, *data.coeffs)
        except:
            coeffs = data.coeffs
        return CnHSModel(*self.funcs, coeffs, data.Tmin, data.Tmax, self.name)


# %% Heat Capacities Gas

Lastovka_Shaw = functor(hc.Lastovka_Shaw, 'Cn.g')

@forward(hc)
@functor(var='H.g')
def Lastovka_Shaw_definite_integral(Ta, Tb, MW, similarity_variable, cyclic_aliphatic=False):
    term = hc.Lastovka_Shaw_term_A(similarity_variable, cyclic_aliphatic)
    return (hc.Lastovka_Shaw_integral(Tb, similarity_variable, cyclic_aliphatic, MW, term)
            - hc.Lastovka_Shaw_integral(Ta, similarity_variable, cyclic_aliphatic, MW, term))

@forward(hc)    
@functor(var='S.g')
def Lastovka_Shaw_definite_integral_over_T(Ta, Tb, MW, similarity_variable,
                                  cyclic_aliphatic=False):
    term = hc.Lastovka_Shaw_term_A(similarity_variable, cyclic_aliphatic)
    return (hc.Lastovka_Shaw_integral_over_T(Tb, similarity_variable, cyclic_aliphatic, MW, term)
            - hc.Lastovka_Shaw_integral_over_T(Ta, similarity_variable, cyclic_aliphatic, MW, term))

TRCCp = functor(hc.TRCCp, 'Cn.g')

@forward(hc)
@functor(var='H.g')
def TRCCp_definite_integral(Ta, Tb, a0, a1, a2, a3, a4, a5, a6, a7):
    return (hc.TRCCp_integral(Tb, a0, a1, a2, a3, a4, a5, a6, a7)
            - hc.TRCCp_integral(Ta, a0, a1, a2, a3, a4, a5, a6, a7))

@forward(hc)
@functor(var='S.g')
def TRCCp_definite_integral_over_T(Ta, Tb, a0, a1, a2, a3, a4, a5, a6, a7):
    return (hc.TRCCp_integral_over_T(Tb, a0, a1, a2, a3, a4, a5, a6, a7)
            - hc.TRCCp_integral_over_T(Ta, a0, a1, a2, a3, a4, a5, a6, a7))

Poling = functor(hc.Poling, 'Cn.g')

@forward(hc)
@functor(var='H.g')
def Poling_definite_integral(Ta, Tb, a, b, c, d, e):
    return (hc.Poling_integral(Tb, a, b, c, d, e)
            - hc.Poling_integral(Ta, a, b, c, d, e))

@forward(hc)    
@functor(var='S.g')
def Poling_definite_integral_over_T(Ta, Tb, a, b, c, d, e):
    return (hc.Poling_integral_over_T(Tb, a, b, c, d, e)
            - hc.Poling_integral_over_T(Ta, a, b, c, d, e))

# Heat capacity gas methods

TRCCp_functors = (TRCCp.functor,
                  TRCCp_definite_integral.functor,
                  TRCCp_definite_integral_over_T.functor)
Poling_functors = (Poling.functor,
                   Poling_definite_integral.functor, 
                   Poling_definite_integral_over_T.functor)

@TDependentHandleBuilder('Cn.g')
def heat_capacity_gas_handle(handle, CAS, MW, similarity_variable, cyclic_aliphatic):
    add_model = handle.add_model
    if CAS in TRC_gas_data:
        Tmin, Tmax, a0, a1, a2, a3, a4, a5, a6, a7, _, _, _ = TRC_gas_data[CAS]
        funcs = CnHS(*TRCCp_functors, (a0, a1, a2, a3, a4, a5, a6, a7))
        add_model(Tmin=Tmin, Tmax=Tmax, name=hc.TRCIG, **funcs)
    if CAS in Cp_data_Poling:
        Tmin, Tmax, a, b, c, d, e, Cn_g, _ = Cp_data_Poling[CAS]
        if not np.isnan(a):
            funcs = CnHS(*Poling_functors, (a, b, c, d, e))
            add_model(Tmin=Tmin, Tmax=Tmax, **funcs, name=hc.POLING)
        if not np.isnan(Cn_g):
            add_model(Cn_g, Tmin, Tmax, name=hc.POLING_CONST)
    if CAS in CRC_standard_data:
        Cn_g = CRC_standard_data[CAS][-1]
        if not np.isnan(Cn_g):
            add_model(Cn_g, name=hc.CRCSTD)
    if MW and similarity_variable:
        data = (similarity_variable, cyclic_aliphatic, MW)
        add_model(Lastovka_Shaw.functor.from_args(data), name=hc.LASTOVKA_SHAW)
    if CAS in VDI_saturation_dict:
        # NOTE: VDI data is for the saturation curve, i.e. at increasing
        # pressure; it is normally substantially higher than the ideal gas
        # value
        Ts, Cn_gs = lookup_VDI_tabular_data(CAS, 'Cp (g)')
        add_model(InterpolatedTDependentModel(Ts, Cn_gs, Tmin=Ts[0], Tmax=Ts[-1], name=hc.VDI_TABULAR))
hc.heat_capacity_gas_handle = heat_capacity_gas_handle
    
### Heat capacities of liquids

Rowlinson_Poling = functor(hc.Rowlinson_Poling, 'Cn.l')

def Rowlinson_Poling_hook(self, T, kwargs):
    Cpgm = kwargs['Cpgm']
    if callable(Cpgm): 
        kwargs = kwargs.copy()
        kwargs['Cpgm'] = Cpgm(T)
    return self.function(T, **kwargs)
Rowlinson_Poling.functor.hook = Rowlinson_Poling_hook

# # This method is less accurate than Rowlinson Poling; so we don't include it
# @forward(hc)
# @functor(var='Cn.l')
# def Rowlinson_Bondi_2(T, Tc, omega, Cpgm):
#     return hc.Rowlinson_Bondi(T, Tc, omega, Cpgm(T) if callable(Cpgm) else Cpgm)

Dadgostar_Shaw = functor(hc.Dadgostar_Shaw, 'Cn.l')

@forward(hc)
@functor(var='H.l')
def Dadgostar_Shaw_definite_integral(Ta, Tb, similarity_variable, MW):
    terms = hc.Dadgostar_Shaw_terms(similarity_variable)
    return (hc.Dadgostar_Shaw_integral(Tb, similarity_variable, MW, terms)
               - hc.Dadgostar_Shaw_integral(Ta, similarity_variable, MW, terms))
hc.Dadgostar_Shaw_definite_integral = Dadgostar_Shaw_definite_integral

@forward(hc)
@functor(var='S.l')
def Dadgostar_Shaw_definite_integral_over_T(Ta, Tb, similarity_variable, MW):
    terms = hc.Dadgostar_Shaw_terms(similarity_variable)
    return (hc.Dadgostar_Shaw_integral_over_T(Tb, similarity_variable, MW, terms)
               - hc.Dadgostar_Shaw_integral_over_T(Ta, similarity_variable, MW, terms))
hc.Dadgostar_Shaw_definite_integral_over_T = Dadgostar_Shaw_definite_integral_over_T

Zabransky_quasi_polynomial = functor(hc.Zabransky_quasi_polynomial, 'Cn.l')
 
@forward(hc)
@functor(var='H.l')
def Zabransky_quasi_polynomial_definite_integral(Ta, Tb, Tc, a1, a2, a3, a4, a5, a6):
    return (hc.Zabransky_quasi_polynomial_integral(Tb, Tc, a1, a2, a3, a4, a5, a6)
            - hc.Zabransky_quasi_polynomial_integral(Ta, Tc, a1, a2, a3, a4, a5, a6))
hc.Zabransky_quasi_polynomial_definite_integral = Zabransky_quasi_polynomial_definite_integral

@forward(hc)
@functor(var='S.l')
def Zabransky_quasi_polynomial_definite_integral_over_T(Ta, Tb, Tc, a1, a2, a3, a4, a5, a6):
    return (hc.Zabransky_quasi_polynomial_integral_over_T(Tb, Tc, a1, a2, a3, a4, a5, a6)
            - hc.Zabransky_quasi_polynomial_integral_over_T(Ta, Tc, a1, a2, a3, a4, a5, a6))
hc.Zabransky_quasi_polynomial_definite_integral_over_T = Zabransky_quasi_polynomial_definite_integral_over_T

Zabransky_cubic = functor(hc.Zabransky_cubic, 'Cn.l')

@forward(hc)
@functor(var='H.l')
def Zabransky_cubic_definite_integral(Ta, Tb, a1, a2, a3, a4):
    return (hc.Zabransky_cubic_integral(Tb, a1, a2, a3, a4)
            - hc.Zabransky_cubic_integral(Ta, a1, a2, a3, a4))

@forward(hc)
@functor(var='S.l')
def Zabransky_cubic_definite_integral_over_T(Ta, Tb, a1, a2, a3, a4):
    return (hc.Zabransky_cubic_integral_over_T(Tb, a1, a2, a3, a4)
            - hc.Zabransky_cubic_integral_over_T(Ta, a1, a2, a3, a4))
    
# Heat capacity liquid methods:
Zabransky_cubic_functors = (Zabransky_cubic.functor,
                            Zabransky_cubic_definite_integral.functor,
                            Zabransky_cubic_definite_integral_over_T.functor)
Zabransky_quasi_polynomial_functors = (Zabransky_quasi_polynomial.functor,
                                       Zabransky_quasi_polynomial_definite_integral.functor, 
                                       Zabransky_quasi_polynomial_definite_integral_over_T.functor)
Dadgostar_Shaw_functors = (Dadgostar_Shaw.functor,
                           Dadgostar_Shaw_definite_integral.functor,
                           Dadgostar_Shaw_definite_integral_over_T.functor)

zabransky_model_data = ((hc.ZABRANSKY_SPLINE,
                         zabransky_dict_const_s,
                         Zabransky_cubic_functors),
                        (hc.ZABRANSKY_QUASIPOLYNOMIAL,
                         zabransky_dict_const_p,
                         Zabransky_quasi_polynomial_functors),
                        (hc.ZABRANSKY_SPLINE_C,
                         zabransky_dict_iso_s, 
                         Zabransky_cubic_functors),
                        (hc.ZABRANSKY_QUASIPOLYNOMIAL_C, 
                         zabransky_dict_iso_p,
                         Zabransky_quasi_polynomial_functors),
                        (hc.ZABRANSKY_SPLINE_SAT,
                         zabransky_dict_sat_s,
                         Zabransky_cubic_functors),
                        (hc.ZABRANSKY_QUASIPOLYNOMIAL_SAT, 
                         zabransky_dict_sat_p,
                         Zabransky_quasi_polynomial_functors))

zabransky_model_builders = [ZabranskyModelBuilder(*i) for i in zabransky_model_data]
zabransky_model_builders[0].many = True
zabransky_model_builders[2].many = True
zabransky_model_builders[4].many = True

@TDependentHandleBuilder('Cn.l')
def heat_capacity_liquid_handle(handle, CAS, Tb, Tc, omega, MW, similarity_variable, Cn_g):
    for i in zabransky_model_builders: i.add_model(CAS, handle.models)        
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        # NOTE: VDI data is for the saturation curve, i.e. at increasing
        # pressure; it is normally substantially higher than the ideal gas
        # value
        Ts, Cn_ls = lookup_VDI_tabular_data(CAS, 'Cp (l)')
        add_model(InterpolatedTDependentModel(Ts, Cn_ls, Ts[0], Ts[-1], name=hc.VDI_TABULAR))
    if Tc and omega and Cn_g:
        args = (Tc, omega, Cn_g, 200, Tc)
        add_model(Rowlinson_Poling.functor.from_args(args),Tmin=0, Tmax=Tc, name=hc.ROWLINSON_POLING)
    # Other
    if similarity_variable and MW:
        add_model(
            CnHSModel(*Dadgostar_Shaw_functors,
                      data=(similarity_variable, MW),
                      name=hc.DADGOSTAR_SHAW,
                      Tmin=0, Tmax=1e5)
        )
    # Constant models
    if CAS in Cp_data_Poling:
        Tmin, Tmax, a, b, c, d, e, Cn_g, Cn_l = Cp_data_Poling[CAS]
        if not np.isnan(Cn_g):
            if Tmax > 1e5: Tmax = 1e5
            add_model(Cn_l, Tmin, Tmax, name=hc.POLING_CONST)
    if CAS in CRC_standard_data:
        Cn_l = CRC_standard_data[CAS][-5]
        if not np.isnan(Cn_l):
            add_model(Cn_l, 0, Tc, name=hc.CRCSTD)
hc.heat_capacity_liquid_handle = heat_capacity_liquid_handle

# %% Heat Capacity Solid

Lastovka_solid = functor(hc.Lastovka_solid, 'Cn.s')

@forward(hc)
@functor(var='H.s')
def Lastovka_solid_definite_integral(Ta, Tb, similarity_variable, MW):
    return (hc.Lastovka_solid_integral(Tb, similarity_variable, MW)
            - hc.Lastovka_solid_integral(Ta, similarity_variable, MW))

@forward(hc)
@functor(var='S.s')
def Lastovka_solid_definite_integral_over_T(Ta, Tb, similarity_variable, MW):
    return (hc.Lastovka_solid_integral_over_T(Tb, similarity_variable, MW)
            - hc.Lastovka_solid_integral_over_T(Ta, similarity_variable, MW))
    
Perry_151 = functor(hc.Perry_151, 'Cn.s')

@forward(hc)
@functor(var='H.s')
def Perry_151_definite_integral(Ta, Tb, a, b, c, d):
    H1 = (a*Ta + 0.5*b*Ta**2 - d/Ta + c*Ta**3/3.)
    H2 = (a*Tb + 0.5*b*Tb**2 - d/Tb + c*Tb**3/3.)
    return (H2 - H1)*calorie

@forward(hc)
@functor(var='S.s')
def Perry_151_definite_integral_over_T(Ta, Tb, a, b, c, d):
    S1 = a*log(Ta) + b*Ta - d/(2.*Ta**2) + 0.5*c*Ta**2
    S2 = a*log(Tb) + b*Tb - d/(2.*Tb**2) + 0.5*c*Tb**2
    return (S2 - S1)*calorie

Lastovka_solid_functors = (Lastovka_solid.functor,
                           Lastovka_solid_definite_integral.functor,
                           Lastovka_solid_definite_integral_over_T.functor)
Perry_151_functors = (Perry_151.functor,
                      Perry_151_definite_integral.functor, 
                      Perry_151_definite_integral_over_T.functor)

# Heat capacity solid methods

@TDependentHandleBuilder('Cn.s')
def heat_capacity_solid_handle(handle, CAS, similarity_variable, MW):
    Tmin = 0
    Tmax = 2000
    add_model = handle.add_model
    if CAS in Cp_dict_PerryI:
        vals = Cp_dict_PerryI[CAS]
        if 'c' in vals:
            c = vals['c']
            Tmin = c['Tmin']
            Tmax = c['Tmax']
            data = (c['Const'], c['Lin'], c['Quad'], c['Quadinv'])
            add_model(CnHSModel(*Perry_151_functors, data), Tmin, Tmax,
                      name=hc.PERRY151)
    if CAS in CRC_standard_data:
        Cnc = CRC_standard_data[CAS][3]
        if not np.isnan(Cnc):
            add_model(float(Cnc), 200, 350)
    if similarity_variable and MW:
        data = (similarity_variable, MW)
        add_model(CnHSModel(*Lastovka_solid_functors, data), Tmin, Tmax,
                  name=hc.LASTOVKA_S)
hc.heat_capacity_solid_handle = heat_capacity_solid_handle

hc.heat_capacity_handle = PhaseTHandleBuilder(
    'Cn', heat_capacity_solid_handle, heat_capacity_liquid_handle, heat_capacity_gas_handle
)
