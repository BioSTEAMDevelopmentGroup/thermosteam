# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:04:44 2019

@author: yoelr
"""
from ..base import Psat, TDependentHandleBuilder
from .dippr import DIPPR_EQ101
from .utils import CASDataReader
from math import log, exp
import numpy as np

# %% Data

read = CASDataReader(__file__, 'Vapor Pressure')
_WagnerMcGarry = read("Wagner Original McGarry.tsv")
_Wagner = read("Wagner Collection Poling.tsv")
_Antoine = read("Antoine Collection Poling.tsv")
_AntoineExtended = read("Antoine Extended Collection Poling.tsv")
_Perrys2_8 = read("Table 2-8 Vapor Pressure of Inorganic and Organic Liquids.tsv")
_VDI_PPDS_3 = read("VDI PPDS Boiling temperatures at different pressures.tsv")

# %% Vapor pressure

@Psat(ref='[1]_',
      math=r"\log_{\text{10}} P^{\text{sat}} = A - \frac{B}{T+C}")
def Antoine(T, a, b, c):
    """
    {Header}

    {Math}

    {Parameters}

    Examples
    --------
    Methane, coefficients from [2]_, at 100 K:
    
    >>> antoine = Antoine(a=8.7687, b=395.744, c=-6.469)
    Functor: Antoine(T, P=None) -> Psat [Pa]
     a: 8.7687
     b: 395.74
     c: -6.469
    >>> antoine(100)
    34478.367349639906
    
    Tetrafluoromethane, coefficients from [2]_, at 180 K
    
    >>> antoine = Antoine(a=8.95894, b=510.595, c=-15.95)
    >>> antoine
    Functor: Antoine(T, P=None) -> Psat [Pa]
     a: 8.9589
     b: 510.6
     c: -15.95
    >>> antoine(180)
    702271.0518579542
    
    Oxygen at 94.91 K, with coefficients from [3]_ in units of °C, mmHg, log10,
    showing the conversion of coefficients A (mmHg to Pa) and C (°C to K)
    
    >>> antoine = Antoine(6.83706+2.1249, 339.2095, 268.70-273.15)
    >>> Antoine
    Functor: Antoine(T, P=None) -> Psat [Pa]
     a: 8.962
     b: 339.21
     c: -4.45
    >>> antoine(94.91)
    162978.88655572367

    References
    ----------
    .. [1] Antoine, C. 1888. Tensions des Vapeurs: Nouvelle Relation Entre les 
       Tensions et les Tempé. Compt.Rend. 107:681-684.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [3] Yaws, Carl L. The Yaws Handbook of Vapor Pressure: Antoine 
       Coefficients. 1 edition. Houston, Tex: Gulf Publishing Company, 2007.
    """
    return 10.0**(a - b / (T + c))

@Psat(ref='[1]_',
      math=(r"\log_{10} P^{sat} = A - \frac{B}{T + C} + 0.43429x^n + Ex^8 + Fx^{12}",
           r"x = \max \left(\frac{T-t_o-273.15}{T_c}, 0 \right)"))
def TRC_Extended_Antoine(T, Tc, to, a, b, c, n, E, F):
    """
    {Header}
    Parameters are chemical dependent, and said to be from the 
    Thermodynamics Research Center (TRC) at Texas A&M. Coefficients for various
    chemicals can be found in [1]_.

    {Math}

    {Parameters}

    Examples
    --------
    Tetrafluoromethane, coefficients from [1]_, at 180 K:
    
    >>> TRC_Antoine_extended(180.0, 227.51, -120., 8.95894, 510.595, -15.95, 
    ... 2.41377, -93.74, 7425.9) 
    706317.0898414153

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    """
    x = max((T - to - 273.15) / Tc, 0.0)
    return 10.0**(a - b / (T + c) + 0.43429 * x**n + E * x**8 + F * x**12)

@Psat(ref='[1]_',
      math=(r"\ln P^{sat}= \ln P_c + \frac{a\tau + b \tau^{1.5} + c\tau^3 + d\tau^6} {T_r}",
            r"\tau = 1 - \frac{T}{T_c}"))
def Wagner_McGraw(T, a, b, c, d, Tc, Pc):
    """
    {Header}

    {Math}

    {Parameters}

    Notes
    -----
    Warning: Pc is often treated as adjustable constant.

    Examples
    --------
    Methane, coefficients from [2]_, at 100 K.

    >>> Wagner_McGraw(100.0, a=-6.00435, b=1.1885, 
    ... c=-0.834082, d=-1.22833, Tc=190.53, Pc=4596420.)
    34520.44601450496

    References
    ----------
    .. [1] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    .. [2] McGarry, Jack. "Correlation and Prediction of the Vapor Pressures of
       Pure Liquids over Large Pressure Ranges." Industrial & Engineering
       Chemistry Process Design and Development 22, no. 2 (April 1, 1983):
       313-22. doi:10.1021/i200021a023.
    """
    Tr = T / Tc
    tau = 1.0 - Tr
    return Pc * exp((a * tau + b * tau**1.5 + c * tau**3 + d * tau**6) / Tr)

@Psat(ref='[1]_',
      math=(r"\ln P^{sat}= \ln P_c + \frac{a\tau + b \tau^{1.5} + c\tau^{2.5} + d\tau^5} {T_r}",
            r"\tau = 1 - \frac{T}{T_c}"))
def Wagner(T, Tc, Pc, a, b, c, d):
    """
    {Header}

    {Math}
    
    {Parameters}

    Notes
    -----
    Warning: Pc is often treated as adjustable constant.

    Examples
    --------
    Methane, coefficients from [2]_, at 100 K.

    >>> Wagner(100., 190.551, 4599200, -6.02242, 1.26652, -0.5707, -1.366)
    34415.00476263708

    References
    ----------
    .. [1] Wagner, W. "New Vapour Pressure Measurements for Argon and Nitrogen and
       a New Method for Establishing Rational Vapour Pressure Equations."
       Cryogenics 13, no. 8 (August 1973): 470-82. doi:10.1016/0011-2275(73)90003-9
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    """
    Tr = T / Tc
    τ = 1.0 - Tr
    return Pc * exp((a*τ + b*τ**1.5 + c*τ**2.5 + d*τ**5) / Tr)

@Psat(ref="[1]_",
      math=(r"\ln P^{sat}_r = h\left( 1 - \frac{1}{T_r} \right)",
            r"h = T_{br} \frac{\ln(P_c/101325)}{1-T_{br}}"))
def Boiling_Critical_Relation(T, Tc, Pc, Tbr, h):
    r"""
    {Header}

    {Math}

    {Parameters}
    
    Notes
    -----
    Units are Pa. Formulation makes intuitive sense; a logarithmic form of
    interpolation.

    Examples
    --------
    Example as in [1]_ for ethylbenzene

    >>> Boiling_Critical_Relation(347.2, 409.3, 617.1, 36E5)
    15209.467273093938

    References
    ----------
    .. [1] Reid, Robert C..; Prausnitz, John M.;; Poling, Bruce E.
       The Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    """
    return exp(h * (1 - Tc / T)) * Pc

@Boiling_Critical_Relation.wrapper
def Boiling_Critical_Relation(Tb, Tc, Pc):
    Tbr = Tb / Tc
    h = Tbr * log(Pc / 101325.0) / (1 - Tbr)
    return {'Tc':Tc, 'Pc':Pc, 'Tbr': Tbr, 'h':h}

@Psat(ref='[1]_',
      math=(r"\ln P^{sat}_r = f^{(0)} + \omega f^{(1)}",
            r"f^{(0)} = 5.92714-\frac{6.09648}{T_r}-1.28862\ln T_r + 0.169347T_r^6",
            r"f^{(1)} = 15.2518-\frac{15.6875}{T_r} - 13.4721 \ln T_r + 0.43577T_r^6"))
def Lee_Kesler(T, Tc, Pc, ω):
    r"""
    {Header}

    {Math}

    {Parameters}
    
    Notes
    -----
    This equation appears in [1]_ in expanded form.
    The reduced pressure form of the equation ensures predicted vapor pressure 
    cannot surpass the critical pressure.

    Examples
    --------
    Example from [2]_; ethylbenzene at 347.2 K.

    >>> Lee_Kesler(347.2, 617.1, 36E5, 0.299)
    13078.694162949312

    References
    ----------
    .. [1] Lee, Byung Ik, and Michael G. Kesler. "A Generalized Thermodynamic
       Correlation Based on Three-Parameter Corresponding States." AIChE Journal
       21, no. 3 (1975): 510-527. doi:10.1002/aic.690210313.
    .. [2] Reid, Robert C..; Prausnitz, John M.;; Poling, Bruce E.
       The Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
    """
    Tr = T / Tc
    Tra = Tr**6
    logTr = log(Tr)
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * logTr + 0.169347 * Tra
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * logTr + 0.43577 * Tra
    return exp(f0 + ω * f1) * Pc

@Psat(ref="[1]_",
      math=(r"\ln P_r=f^{(0)}+\omega f^{(1)}+\omega^2f^{(2)}",
            r"f^{(0)}=\frac{-5.97616\tau + 1.29874\tau^{1.5}- 0.60394\tau^{2.5}-1.06841\tau^5}{T_r}",
            r"f^{(1)}=\frac{-5.03365\tau + 1.11505\tau^{1.5}- 5.41217\tau^{2.5}-7.46628\tau^5}{T_r}",
            r"f^{(2)}=\frac{-0.64771\tau + 2.41539\tau^{1.5}- 4.26979\tau^{2.5}+3.25259\tau^5}{T_r}",
            r"\tau = 1-T_{r}"))
def Ambrose_Walton(T, Tc, Pc, ω):
    r"""
    {Header}

    {Math}

    {Parameters}

    Notes
    -----
    Somewhat more accurate than the :obj:`Lee_Kesler` formulation.

    Examples
    --------
    Example from [2]_; ethylbenzene at 347.25 K.

    >>> Ambrose_Walton(347.25, 617.15, 36.09E5, 0.304)
    13278.878504306222

    References
    ----------
    .. [1] Ambrose, D., and J. Walton. "Vapour Pressures up to Their Critical
       Temperatures of Normal Alkanes and 1-Alkanols." Pure and Applied
       Chemistry 61, no. 8 (1989): 1395-1403. doi:10.1351/pac198961081395.
    .. [2] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    """
    Tr = T / Tc
    τ = 1 - Tr
    τa = τ**1.5
    τb = τ**2.5
    τc = τ**5
    f0 = -5.97616 * τ + 1.29874 * τa - 0.60394 * τb - 1.06841 * τc
    f1 = -5.03365 * τ + 1.11505 * τa - 5.41217 * τb - 7.46628 * τc
    f2 = -0.64771 * τ + 2.41539 * τa - 4.26979 * τb + 3.25259 * τc
    return Pc * exp((f0 + f1 * ω + f2 * ω**2) / Tr)

@Psat(ref="[1]_",
      math=(r"P^{sat} = P_c\exp(f^{(0)} + \omega f^{(1)} + \omega^2 f^{(2)})",
            r"f^{(0)} = a_1 + \frac{a_2}{T_r} + a_3\ln T_r + a_4 T_r^{1.9}",
            r"f^{(1)} = a_5 + \frac{a_6}{T_r} + a_7\ln T_r + a_8 T_r^{1.9}",
            "rf^{(2)} = a_9 + \frac{a_{10}}{T_r} + a_{11}\ln T_r + a_{12} T_r^{1.9}"))
def Sanjari(T, Tc, Pc, ω):
    r"""
    {Header}

    {Math}

    {Parameters}
    
    Notes
    -----
    Although developed for refrigerants, this model should have some general predictive ability.
    
    a[1-12] are as follows:
    6.83377, -5.76051, 0.90654, -1.16906,
    5.32034, -28.1460, -58.0352, 23.57466,
    18.19967, 16.33839, 65.6995, -35.9739.

    For a claimed fluid not included in the regression, R128, the claimed AARD
    was 0.428%. A re-calculation using 200 data points from 125.45 K to
    343.90225 K evenly spaced by 1.09775 K as generated by NIST Webbook April
    2016 produced an AARD of 0.644%. It is likely that the author's regression
    used more precision in its coefficients than was shown here. Nevertheless,
    the function is reproduced as shown in [1]_.

    For Tc=808 K, Pc=1100000 Pa, omega=1.1571, this function actually declines
    after 770 K.

    Examples
    --------
    >>> Sanjari(347.2, 617.1, 36E5, 0.299)
    13651.916109552498

    References
    ----------
    .. [1] Sanjari, Ehsan, Mehrdad Honarmand, Hamidreza Badihi, and Ali
       Ghaheri. "An Accurate Generalized Model for Predict Vapor Pressure of
       Refrigerants." International Journal of Refrigeration 36, no. 4
       (June 2013): 1327-32. doi:10.1016/j.ijrefrig.2013.01.007.
    """
    Tr = T / Tc
    logTr = log(Tr)
    Ta = Tr**1.9
    f0 = 6.83377 + -5.76051 / Tr + 0.90654 * logTr + -1.16906 * Ta
    f1 = 5.32034 + -28.1460 / Tr + -58.0352 * logTr + 23.57466 * Ta
    f2 = 18.19967 + 16.33839 / Tr + 65.6995 * logTr + -35.9739 * Ta
    return Pc * exp(f0 + f1 * ω + f2 * ω**2)

@Psat(ref='[1]_',
      math=(r"\ln(P^{sat}/P_c) = \frac{a\tau + b\tau^{1.5} + c\tau^3 + d\tau^6}{1-\tau}",
            r"a = -6.1559 - 4.0855\omega",
            r"b = 1.5737 - 1.0540\omega - 4.4365\times 10^{-3} d",
            r"c = -0.8747 - 7.8874\omega",
            r"d = \frac{1}{-0.4893 - 0.9912\omega + 3.1551\omega^2}",
            r"\tau = 1 - \frac{T}{T_c})"))
def Edalat(T, Tc, Pc, ω):
    r"""
    {Header}

    {Math}
        
    {Parameters}

    Notes
    -----
    Claimed to have a higher accuracy than the Lee-Kesler CSP relationship.
    [1]_ found an average error of 6.06% on 94 compounds and 1106 data points.
    
    Examples
    --------
    >>> Edalat(347.2, 617.1, 36E5, 0.299)
    13461.273080743307

    References
    ----------
    .. [1] Edalat, M., R. B. Bozar-Jomehri, and G. A. Mansoori. "Generalized 
       Equation Predicts Vapor Pressure of Hydrocarbons." Oil and Gas Journal; 
       91:5 (February 1, 1993).
    """
    τ = 1.0 - T / Tc
    a = -6.1559 - 4.0855 * ω
    c = -0.8747 - 7.8874 * ω
    d = 1.0 / (-0.4893 - 0.9912 * ω + 3.1551 * ω**2)
    b = 1.5737 - 1.0540 * ω - 4.4365E-3 * d
    lnPr = (a * τ + b * τ**1.5 + c * τ**3.0 + d * τ**6.0) / (1.0 - τ)
    return exp(lnPr) * Pc

@TDependentHandleBuilder
def VaporPressure(handle, CAS, Tb, Tc, Pc, omega):
    if CAS in _WagnerMcGarry:
        _, a, b, c, d, Pc, Tc, Tmin = _WagnerMcGarry[CAS]
        Tmax = Tc
        data = (a, b, c, d, Tc, Pc)
        handle.model(Wagner_McGraw.from_args(data), Tmin, Tmax)
    if CAS in _Wagner:
        _, a, b, c, d, Tc, Pc, Tmin, Tmax = _Wagner[CAS]
        # Some Tmin values are missing; Arbitrary choice of 0.1 lower limit
        if np.isnan(Tmin): Tmin = Tmax * 0.1
        data = (a, b, c, d, Tc, Pc)
        handle.model(Wagner_McGraw.from_args(data), Tmin, Tmax)
    if CAS in _AntoineExtended:
        _, a, b, c, Tc, to, n, E, F, Tmin, Tmax = _AntoineExtended[CAS]
        data = (a, b, c, Tc, to, n, E, F)
        handle.model(TRC_Extended_Antoine.from_args(data), Tmin, Tmax)
    if CAS in _Antoine:
        _, a, b, c, Tmin, Tmax = _Antoine[CAS]
        data = (a, b, c)
        handle.model(Antoine.from_args(data), Tmin, Tmax)
    if CAS in _Perrys2_8:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = _Perrys2_8[CAS]
        data = (C1, C2, C3, C4, C5)
        handle.model(DIPPR_EQ101.from_args(data), Tmin, Tmax,)
    if CAS in _VDI_PPDS_3:
        _, Tm, Tc, Pc, a, b, c, d = _VDI_PPDS_3[CAS]
        data = (Tc, Pc, a, b, c, d)
        handle.model(Wagner.from_args(data), 0., Tc,)
    data = (Tb, Tc, Pc)
    if all(data):
        handle.model(Boiling_Critical_Relation.from_args(data), 0., Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for f in (Lee_Kesler, Ambrose_Walton, Sanjari, Edalat):
            handle.model(f.from_args(data), 0., Tc)
    