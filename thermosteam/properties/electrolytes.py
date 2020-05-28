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
"""
Data and functions for calculating heat capacity,
density, and viscosity of aqueous electrolyte solutions as given by [1]_.

References
----------
.. [1] Laliberte, Marc. "A Model for Calculating the Heat Capacity of
   Aqueous Solutions, with Updated Density and Viscosity Data." Journal of
   Chemical & Engineering Data 54, no. 6 (June 11, 2009): 1725-60.
   doi:10.1021/je8008123

"""
from math import exp
from scipy.interpolate import interp1d
from .data import (Laliberte_Density_ParametersDict,
                   Laliberte_Viscosity_ParametersDict,
                   Laliberte_Heat_Capacity_ParametersDict)

### Laliberty Viscosity Functions

def Laliberte_water_viscosity(T):
    r"""
    Return the viscosity of a water (Pa*s) at arbitrary temperatures (K) 
    using the form proposed by [1]_. 

    .. math::
        \mu_w = \frac{T - 27.15}{(0.05594T-25.27581)t + 2867.723}

    Notes
    -----
    Original source or pure water viscosity is not cited.
    No temperature range is given for this equation.

    """
    return (T - 27.15)/((55.94*T-25275.81)*T + 2867723)


def Laliberte_partial_viscosity(T, w_w, v1, v2, v3, v4, v5, v6):
    r'''
    Return the viscosity of a solute using the form proposed by [1]_
    
    .. math::
        \mu_i = \frac{\exp\left( \frac{v_1(1-w_w)^{v_2}+v_3}{v_4 t +1}\right)}
            {v_5(1-w_w)^{v_6}+1}
    
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    v1-v6 : floats
        Function fit parameters
    
    Returns
    -------
    mu_i : float
        Solute partial viscosity, Pa*s
    
    Notes
    -----
    Temperature range check is outside of this function.
    Check is performed using NaCl at 5 degC from the first value in [1]_'s spreadsheet.
    
    Examples
    --------
    >>> d =  Laliberte_Viscosity_ParametersDict['7647-14-5']
    >>> Laliberte_viscosity_i(273.15+5, 1-0.005810, d["V1"], d["V2"], d["V3"], d["V4"], d["V5"], d["V6"] )
    0.004254025533308794
    
    '''
    t = T-273.15
    mu_i = exp((v1*(1-w_w)**v2 + v3)/(v4*t+1))/(v5*(1-w_w)**v6 + 1)
    return mu_i/1000.

### Laliberty Density Functions

def Laliberte_water_density(T):
    r"""
    Return the density of water using the form proposed by [1]_.
    No parameters are needed, just a temperature.

    .. math::
        \rho_w = \frac{\left\{\left([(-2.8054253\times 10^{-10}\cdot t +
        1.0556302\times 10^{-7})t - 4.6170461\times 10^{-5}]t
        -0.0079870401\right)t + 16.945176   \right\}t + 999.83952}
        {1 + 0.01687985\cdot t}

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    rho_w : float
        Water density, [kg/m^3]

    Notes
    -----
    Original source not cited
    No temperature range is used.

    Examples
    --------
    >>> Laliberte_density_w(298.15)
    997.0448954179155
    >>> Laliberte_density_w(273.15 + 50)
    988.0362916114763

    """
    t = T-273.15
    rho_w = (((((-2.8054253E-10*t + 1.0556302E-7)*t - 4.6170461E-5)*t - 0.0079870401)*t + 16.945176)*t + 999.83952) / (1 + 0.01687985*t)
    return rho_w


def Laliberte_partial_density(T, w_w, c0, c1, c2, c3, c4):
    r"""
    Return the density of a solute using the form proposed by Laliberte [1]_.
    
    .. math::
        \rho_{app,i} = \frac{(c_0[1-w_w]+c_1)\exp(10^{-6}[t+c_4]^2)}
        {(1-w_w) + c_2 + c_3 t}
    
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    c0-c4 : floats
        Function fit parameters
    
    Returns
    -------
    rho_i : float
        Solute partial density, [kg/m^3]
    
    Notes
    -----
    Temperature range check is TODO
    
    Examples
    --------
    >>> d = Laliberte_Density_ParametersDict['7647-14-5']
    >>> Laliberte_partial_density(273.15+0, 1-0.0037838838, d["C0"], d["C1"], d["C2"], d["C3"], d["C4"])
    3761.8917585699983
    
    """
    t = T - 273.15
    return ((c0*(1 - w_w)+c1)*exp(1E-6*(t + c4)**2))/((1 - w_w) + c2 + c3*t)


### Laliberty Heat Capacity Functions

_T_array = [-15, -10, -5, 0, 5, 10, 15, 20, 25,
            30, 35, 40, 45, 50, 55, 60, 65, 70,
            75, 80, 85, 90, 95, 100, 105, 110,
            115, 120, 125, 130, 135, 140]
_Cp_array = [4294.03, 4256.88, 4233.58, 4219.44,
             4204.95, 4195.45, 4189.1, 4184.8, 
             4181.9, 4180.02, 4178.95, 4178.86, 
             4178.77, 4179.56, 4180.89, 4182.77, 
             4185.17, 4188.1, 4191.55, 4195.52, 
             4200.01, 4205.02, 4210.57, 4216.64, 
             4223.23, 4230.36, 4238.07, 4246.37, 
             4255.28, 4264.84, 4275.08, 4286.04]
Laliberte_heat_capacity_w_interp = interp1d(_T_array, _Cp_array, kind='cubic')
del _T_array, _Cp_array

def Laliberte_water_heat_capacity(T):
    r"""
    Return the heat capacity of water using the interpolation proposed by [1]_.

    .. math::
        Cp_w = Cp_1 + (Cp_2-Cp_1) \left( \frac{t-t_1}{t_2-t_1}\right)
        + \frac{(Cp_3 - 2Cp_2 + Cp_1)}{2}\left( \frac{t-t_1}{t_2-t_1}\right)
        \left( \frac{t-t_1}{t_2-t_1}-1\right)

    Parameters
    ----------
    T : float
        Temperature of fluid [K]

    Returns
    -------
    Cp_w : float
        Water heat capacity, [J/kg/K]

    Notes
    -----
    Units are Kelvin and J/kg/K.
    Original source not cited
    No temperature range is used.
    The original equation is not used, but rather a cubic scipy interpolation routine.

    Examples
    --------
    >>> Laliberte_heat_capacity_w(273.15+3.56)
    4208.878020261102

    """
    return float(Laliberte_heat_capacity_w_interp(T - 273.15))


def Laliberte_partial_heat_capacity(T, w_w, a1, a2, a3, a4, a5, a6):
    r'''
    Return the heat capacity of a solute using the form proposed by [1]_
    
    .. math::
        Cp_i = a_1 e^\alpha + a_5(1-w_w)^{a_6}
        \alpha = a_2 t + a_3 \exp(0.01t) + a_4(1-w_w)
    
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    w_w : float
        Weight fraction of water in the solution
    a1-a6 : floats
        Function fit parameters
    
    Returns
    -------
    Cp_i : float
        Solute partial heat capacity, [J/kg/K]
    
    Notes
    -----
    Units are Kelvin and J/kg/K.
    Temperature range check is TODO
    
    Examples
    --------
    >>> d = Laliberte_Heat_Capacity_ParametersDict['7647-14-5']
    >>> Laliberte_heat_capacity_i(1.5+273.15, 1-0.00398447, d["A1"], d["A2"], d["A3"], d["A4"], d["A5"], d["A6"])
    -2930.7353945880477
    
    '''
    t = T - 273.15
    alpha = a2*t + a3*exp(0.01*t) + a4*(1. - w_w)
    return 1000. * (a1*exp(alpha) + a5*(1. - w_w)**a6)
