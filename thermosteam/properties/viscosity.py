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
Data and methods for calculating the viscosity of a chemical.

References
----------
.. [1] Viswanath, Dabir S., and G. Natarajan. Databook On The Viscosity Of
       Liquids. New York: Taylor & Francis, 1989
.. [2] Letsou, Athena, and Leonard I. Stiel. "Viscosity of Saturated
       Nonpolar Liquids at Elevated Pressures." AIChE Journal 19, no. 2 (1973):
       409-11. doi:10.1002/aic.690190241.
.. [3] Przedziecki, J. W., and T. Sridhar. "Prediction of Liquid
       Viscosities." AIChE Journal 31, no. 2 (February 1, 1985): 333-35.
       doi:10.1002/aic.690310225.
.. [4] Lucas, Klaus. "Ein Einfaches Verfahren Zur Berechnung Der
       Viskositat von Gasen Und Gasgemischen." Chemie Ingenieur Technik 46, no. 4
       (February 1, 1974): 157-157. doi:10.1002/cite.330460413.
.. [5] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
       Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
.. [6] Yoon, Poong, and George Thodos. "Viscosity of Nonpolar Gaseous
       Mixtures at Normal Pressures." AIChE Journal 16, no. 2 (1970): 300-304.
       doi:10.1002/aic.690160225.
.. [7] Stiel, Leonard I., and George Thodos. "The Viscosity of Nonpolar
       Gases at Normal Pressures." AIChE Journal 7, no. 4 (1961): 611-15.
       doi:10.1002/aic.690070416.
.. [8] Gharagheizi, Farhad, Ali Eslamimanesh, Mehdi Sattari, Amir H.
       Mohammadi, and Dominique Richon. "Corresponding States Method for
       Determination of the Viscosity of Gases at Atmospheric Pressure."
       Industrial & Engineering Chemistry Research 51, no. 7
       (February 22, 2012): 3179-85. doi:10.1021/ie202591f.

"""
from math import exp
from ..functional import horner_polynomial
from ..base import (InterpolatedTDependentModel, 
                    TPDependentHandleBuilder, 
                    TDependentModel, 
                    PhaseTPHandleBuilder,
                    functor)
from .data import (VDI_saturation_dict,
                   VDI_tabular_data,
                   mu_data_Dutt_Prasad,
                   mu_data_VN3,
                   mu_data_VN2,
                   mu_data_Perrys2_313,
                   mu_data_Perrys2_312,
                   mu_data_VDI_PPDS_7,
                   mu_data_VDI_PPDS_8,
)
# from .electrochem import _Laliberte_Viscosity_ParametersDict, Laliberte_viscosity
from .dippr import DIPPR_EQ101, DIPPR_EQ102

__all__= ('viscosity_gas_handle',
          'Viswanath_Natarajan2', 
          'Viswanath_Natarajan3', 
          'Letsou_Stiel', 
          'Przedziecki_Sridhar',
          'VDI', 
          'Lucas_liquid',
          'Lucas_gas',
          'Yoon_Thodos',
          'Stiel_Thodos',
          'Gharagheizi_gas_viscosity',
)


@functor(var='mu.l')
def Viswanath_Natarajan2(T, a, b):
    """
    Create a functor of temperature (T; in K) that estimates the liquid 
    hydraulic viscosity (mu.l; in Pa*s) of a chemical using the
    Viswanath Natarajan method #2, as described in [1]_.
    
    Parameters
    ----------
    a,b : float
        Regressed coefficients.
    
    Notes
    -----
    This method is known to produce values 10 times too low.
    The author's data must have an error. I have adjusted it to fix this.
    
    >>> # DDBST has 0.0004580 as a value at this temperature
    >>> f = ViswanathNatarajan2(-5.9719, 1007.0)
    >>> f(348.15)
    0.00045983686956829517
    
    """
    return exp(a + b/T) / 100.

@functor(var='mu.l')
def Viswanath_Natarajan3(T, a, b, c):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    hydraulic viscosity (mu.l; in Pa*s) of a chemical using the Viswanath
    Natarajan method #3, as described in [1]_.
    
    Parameters
    ----------
    a,b,c : float
        Regressed coefficients.
        
    Notes
    -----
    Calculates the viscosity of a liquid using the 3-term Antoine form
    representation developed in [1]_. Requires input coefficients. The `A`
    coefficient is assumed to yield coefficients in centipoise, as all 
    coefficients found so far have been.
    
    .. math::
        \log_{10} \mu = a + b/(T + c)
    
    No other source for these coefficients has been found.
    
    Examples
    --------
    >>> ViswanathNatarajan3(298.15, -2.7173, -1071.18, -129.51)
    0.0006129806445142112
    
    """
    return 10**(a + b/(c - T))/1000.

@functor(var='mu.l')
def Letsou_Stiel(T, MW, Tc, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    hydraulic viscosity (mu.l; in Pa*s) of a chemical using the Letsou Stiel 
    method, as described in [2]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
        
    Notes
    -----
    Calculates the viscosity of a liquid using an emperical model
    developed in [2]_. However. the fitting parameters for tabulated values
    in the original article are found in ChemSep.
    
    .. math::
        \xi = \frac{2173.424 T_c^{1/6}}{\sqrt{MW} P_c^{2/3}}
        
        \xi^{(0)} = (1.5174 - 2.135T_r + 0.75T_r^2)\cdot 10^{-5}
        
        \xi^{(1)} = (4.2552 - 7.674 T_r + 3.4 T_r^2)\cdot 10^{-5}
        
        \mu = (\xi^{(0)} + \omega \xi^{(1)})/\xi
    
    The form of this equation is a polynomial fit to tabulated data.
    The fitting was performed by the DIPPR. This is DIPPR Procedure 8G: Method
    for the viscosity of pure, nonhydrocarbon liquids at high temperatures
    internal units are SI standard. [2]_'s units were different.
    DIPPR test value for ethanol is used.
    Average error 34%. Range of applicability is 0.76 < Tr < 0.98.
    
    Examples
    --------
    >>> f = Letsou_Stiel(46.07, 516.25, 6.383E6, 0.6371)
    >>> f(400.)
    0.0002036150875308151
    
    """
    Tr = T/Tc
    xi0 = (1.5174-2.135*Tr + 0.75*Tr**2)*1E-5
    xi1 = (4.2552-7.674*Tr + 3.4*Tr**2)*1E-5
    xi = 2173.424*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    return (xi0 + omega*xi1)/xi

@functor(var='mu.l')
def Przedziecki_Sridhar(T, Tm, Tc, Pc, Vc, V, omega, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    hydraulic viscosity (mu.l; in Pa*s) of a chemical using the Przedziecki 
    Sridhar method, as described in [3]_.
    
    Parameters
    ----------
    Tm : float
        Melting point temperature [K].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    Vc : float
        Critical point volume [m^3/mol].
    V : float
        Molar volume [m^3/mol].
    omega : float
        Acentric factor [-].
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Calculates the viscosity of a liquid using an emperical formula
    developed in [1]_.
    
    .. math::
        \mu=\frac{V_o}{E(V-V_o)}
        
        E=-1.12+\frac{V_c}{12.94+0.10MW-0.23P_c+0.0424T_{m}-11.58(T_{m}/T_c)}
        
        V_o = 0.0085\omega T_c-2.02+\frac{V}{0.342(T_m/T_c)+0.894}
    
    A test by Reid (1983) is used, but only mostly correct.
    This function is not recommended. Its use has been removed from the Liquid Viscosity function.
    Internal units are bar and mL/mol.
    TODO: Test again with data from 5th ed table.
    
    Examples
    --------
    >>> f = Przedziecki_Sridhar(178., 591.8, 41E5, 316E-6, 95E-6, .263, 92.14)
    >>> f(383.)
    0.0002198147995603383
    
    """
    Pc = Pc/1E5  # Pa to atm
    V, Vc = V*1E6, Vc*1E6  # m^3/mol to mL/mol
    Tr = T/Tc
    Gamma = 0.29607 - 0.09045*Tr - 0.04842*Tr**2
    VrT = 0.33593-0.33953*Tr + 1.51941*Tr**2 - 2.02512*Tr**3 + 1.11422*Tr**4
    V = VrT*(1-omega*Gamma)*Vc

    Vo = 0.0085*omega*Tc - 2.02 + V/(0.342*(Tm/Tc) + 0.894)  # checked
    E = -1.12 + Vc/(12.94 + 0.1*MW - 0.23*Pc + 0.0424*Tm - 11.58*(Tm/Tc))
    return Vo/(E*(V-Vo))/1000.

@functor(var='mu.l')
def VDI(T, a, b, c, d, e):
    """
    Create a functor of temperature (T; in K) that estimates the liquid 
    hydraulic viscosity (mu.l; in Pa*s) of a chemical using the VDI-PPDS, 
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.

    """
    term = (c - T)/(T-d)
    if term < 0:
        term1 = -((T - c)/(T-d))**(1/3.)
    else:
        term1 = term**(1/3.)
    term2 = term*term1
    return e*exp(a*term1 + b*term2)

@functor(var='mu.l')
def Lucas_liquid(T, P, Tc, Pc, omega, Psat, mu_l):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that
    estimates the liquid hydraulic viscosity (mu.l; in Pa*s) of a chemical
    using the Lucas liquid method, as described in [4]_ [5]_.
    
    Parameters
    ----------
    T : float
        Temperature of fluid [K]
    P : float
        Pressure of fluid [Pa]
    Tc: float
        Critical point of fluid [K]
    Pc : float
        Critical pressure of the fluid [Pa]
    omega : float
        Acentric factor of compound
    Psat : function(T)
        Saturation pressure of the fluid [Pa]
    mu_l : function(T, P)
        Viscosity of liquid at 1 atm or saturation, [Pa*s]
    
    Notes
    -----
    This equation is entirely dimensionless; all dimensions cancel.
    The example is from Reid (1987); all results agree.
    Above several thousand bar, this equation does not represent true behavior.
    If Psat is larger than P, the fluid may not be liquid; dPr is set to 0.
    
    Examples
    --------
    >>> f = Lucas_liquid(500E5, 572.2, 34.7E5, 0.236, 0, 0.00068) # methylcyclohexane
    >>> f(300.)
    0.0010683738499316518
    
    """
    Tr = T/Tc
    C = -0.07921+2.1616*Tr - 13.4040*Tr**2 + 44.1706*Tr**3 - 84.8291*Tr**4 \
        + 96.1209*Tr**5-59.8127*Tr**6+15.6719*Tr**7
    D = 0.3257/((1.0039-Tr**2.573)**0.2906) - 0.2086
    A = 0.9991 - 4.674E-4/(1.0523*Tr**-0.03877 - 1.0513)
    dPr = (P-Psat(T))/Pc
    if dPr < 0: dPr = 0
    return (1. + D*(dPr/2.118)**A)/(1. + C*omega*dPr)*mu_l(T)

@TPDependentHandleBuilder('mu.l')
def viscosity_liquid_handle(handle, CAS, MW, Tm, Tc, Pc, Vc, omega, Psat, Vl):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = VDI_tabular_data(CAS, 'Mu (l)')
        model = InterpolatedTDependentModel(Ts, Ys, Ts[0], Ts[-1], 
                                            name='VDI-interpolated')
        add_model(model)
    if CAS in mu_data_Dutt_Prasad:
        _, A, B, C, Tmin, Tmax = mu_data_Dutt_Prasad[CAS]
        data = (A, B, C)
        add_model(Viswanath_Natarajan3.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VN3:
        _, _, A, B, C, Tmin, Tmax = mu_data_VN3[CAS]
        data = (A, B, C)
        add_model(Viswanath_Natarajan3.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VN2:
        _, _, A, B, Tmin, Tmax = mu_data_VN2[CAS]
        data = (A, B)
        add_model(Viswanath_Natarajan2.from_args(data), Tmin ,Tmax)
    if CAS in mu_data_Perrys2_313:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = mu_data_Perrys2_313[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(DIPPR_EQ101.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VDI_PPDS_7:
        coef = mu_data_VDI_PPDS_7[CAS][2:]
        add_model(VDI.from_args(coef))
    data = (MW, Tc, Pc, omega)
    if all(data):
        add_model(Letsou_Stiel.from_args(data), Tc/4, Tc)
    data = (MW, Tm, Tc, Pc, Vc, omega, Vl)
    if all(data):
        add_model(Przedziecki_Sridhar.from_args(data), Tm, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        for mu_l in handle.models:
            if isinstance(mu_l, TDependentModel): break
        data = (Tc, Pc, omega, Psat, mu_l)
        add_model(Lucas_liquid.from_args(data), Tm, Tc,
                  name='Lucas')


### Viscosity of Gases - low pressure
@functor(var='mu.g')
def Yoon_Thodos(T, Tc, Pc, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas hydraulic
    viscosity (mu.g; in Pa*s) of a chemical using the Yoon Thodos method, as
    described in [6]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Calculates the viscosity of a gas using an emperical formula
    developed in [6]_.
    
    .. math::
        \eta \xi \times 10^8 = 46.10 T_r^{0.618} - 20.40 \exp(-0.449T_r) + 1
        9.40\exp(-4.058T_r)+1
        
        \xi = 2173.424 T_c^{1/6} MW^{-1/2} P_c^{-2/3}
        
    This equation has been tested. The equation uses SI units only internally.
    The constant 2173.424 is an adjustment factor for units.
    Average deviation within 3% for most compounds.
    Greatest accuracy with dipole moments close to 0.
    Hydrogen and helium have different coefficients, not implemented.
    This is DIPPR Procedure 8B: Method for the Viscosity of Pure,
    non hydrocarbon, nonpolar gases at low pressures
    
    Examples
    --------
    >>> f = Yoon_Thodos(556.35, 4.5596E6, 153.8)
    >>> f(300.)
    1.0194885727776819e-05
    
    """
    Tr = T/Tc
    xi = 2173.4241*Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    a = 46.1
    b = 0.618
    c = 20.4
    d = -0.449
    e = 19.4
    f = -4.058
    return (1. + a*Tr**b - c * exp(d*Tr) + e*exp(f*Tr))/(1E8*xi)

@functor(var='mu.g')
def Stiel_Thodos(T, Tc, Pc, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas hydraulic 
    viscosity (mu.g; in Pa*s) of a chemical using the Stiel Thodos method, as 
    described in [7]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Calculates the viscosity of a gas using an emperical formula
    developed in [7]_.
    
    Untested.
    Claimed applicability from 0.2 to 5 atm.
    Developed with data from 52 nonpolar, and 53 polar gases.
    internal units are poise and atm.
    Seems to give reasonable results.
    
    Examples
    --------
    >>> f = Stiel_Thodos(556.35, 4.5596E6, 153.8) #CCl4
    >>> f(300.)
    1.0408926223608723e-05
    
    """
    Pc = Pc/101325.
    Tr = T/Tc
    xi = Tc**(1/6.)/(MW**0.5*Pc**(2/3.))
    if Tr > 1.5:
        mu_g = 17.78E-5*(4.58*Tr-1.67)**.625/xi
    else:
        mu_g = 34E-5*Tr**0.94/xi
    return mu_g/1000.

_lucas_Q_dict = {'7440-59-7': 1.38, '1333-74-0': 0.76, '7782-39-0': 0.52}

@functor(var='mu.g')
def Lucas_gas(T, Tc, Pc, Zc, MW, Q, dipole=0):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas hydraulic 
    viscosity (mu.g; in Pa*s) of a chemical using the Lucas gas method, as 
    described in [5]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    Zc : float
        Critical compressibility [-].
    MW : float
        Molecular weight [g/mol].
    dipole : float
        Dipole momment [Debye].
    Q : float
        Regressed coefficient.
    
    Notes
    -----
    Estimates the viscosity of a gas using an emperical
    formula developed in several sources, but as discussed in [5]_ as the
    original sources are in German or merely personal communications with the
    authors of [5]_:
    
    .. math::
        \eta  = \left[0.807T_r^{0.618}-0.357\exp(-0.449T_r) + 0.340\exp(-4.058
        T_r) + 0.018\right]F_p^\circ F_Q^\circ /\xi
        
        F_p^\circ = 1, 0 \le \mu_{r} < 0.022
        
        F_p^\circ = 1+30.55(0.292-Z_c)^{1.72}, 0.022 \le \mu_{r} < 0.075
        
        F_p^\circ = 1+30.55(0.292-Z_c)^{1.72}|0.96+0.1(T_r-0.7)| 0.075 < \mu_{r}
        
        F_Q^\circ = 1.22Q^{0.15}\left\{ 1+0.00385[(T_r-12)^2]^{1/M}\text{sign}
        (T_r-12)\right\}
        
        \mu_r = 52.46 \frac{\mu^2 P_c}{T_c^2}
        
        \xi = 0.176\left(\frac{T_c}{MW^3 P_c^4}\right)^{1/6}
    
    The example is from [5]_; all results agree.
    Viscosity is calculated in micropoise, and converted to SI internally (1E-7).
    Q for He = 1.38; Q for H2 = 0.76; Q for D2 = 0.52.
    
    Examples
    --------
    >>> f = Lucas_gas(Tc=512.6, Pc=80.9E5, Zc=0.224, MW=32.042, dipole=1.7)
    >>> f(T=550)
    1.7822676912698928e-05
    
    """
    Tr = T/Tc
    xi = 0.176*(Tc/MW**3/(Pc/1E5)**4)**(1/6.)  # bar arrording to example in Poling
    if dipole is None:
        dipole = 0
    dipoler = 52.46*dipole**2*(Pc/1E5)/Tc**2  # bar arrording to example in Poling
    if dipoler < 0.022:
        Fp = 1
    elif 0.022 <= dipoler < 0.075:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72
    else:
        Fp = 1 + 30.55*(0.292 - Zc)**1.72*abs(0.96 + 0.1*(Tr-0.7))
    if Q:
        if Tr - 12 > 0:
            value = 1
        else:
            value = -1
        FQ = 1.22*Q**0.15*(1 + 0.00385*((Tr-12)**2)**(1./MW)*value)
    else:
        FQ = 1
    eta = (0.807*Tr**0.618 - 0.357*exp(-0.449*Tr) + 0.340*exp(-4.058*Tr) + 0.018)*Fp*FQ/xi
    return eta/1E7

@functor(var='mu.g')
def Gharagheizi_gas_viscosity(T, Tc, Pc, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas hydraulic 
    viscosity (mu.g; in Pa*s) of a chemical using the Gharagheizi method, as 
    described in [8]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Calculates the viscosity of a gas using an emperical formula
    developed in [8]_.
    
    .. math::
        \mu = 10^{-7} | 10^{-5} P_cT_r + \left(0.091-\frac{0.477}{M}\right)T +
        M \left(10^{-5}P_c-\frac{8M^2}{T^2}\right)
        
        \left(\frac{10.7639}{T_c}-\frac{4.1929}{T}\right)|
    
    Example is first point in supporting information of article, for methane.
    This is the prefered function for gas viscosity.
    7% average relative deviation. Deviation should never be above 30%.
    Developed with the DIPPR database. It is believed theoretically predicted values
    are included in the correlation.
    
    Examples
    --------
    >>> f = Gharagheizi_gas_viscosity(190.564, 45.99E5, 16.04246)
    >>> f(120.)
    5.215761625399613e-06
    
    """
    Tr = T/Tc
    mu_g = 1E-5*Pc*Tr + (0.091 - 0.477/MW)*T + MW*(1E-5*Pc - 8*MW**2/T**2)*(10.7639/Tc - 4.1929/T)
    return 1E-7 * abs(mu_g)


@TPDependentHandleBuilder('mu.g')
def viscosity_gas_handle(handle, CAS, MW, Tc, Pc, Zc, dipole):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = VDI_tabular_data(CAS, 'Mu (g)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        model = InterpolatedTDependentModel(Ts, Ys, Tmin, Tmax,
                                            name='VDI-interpolated')
        add_model(model)
    if CAS in mu_data_Perrys2_312:
        _, C1, C2, C3, C4, Tmin, Tmax = mu_data_Perrys2_312[CAS]
        data = (C1, C2, C3, C4)
        add_model(DIPPR_EQ102.from_args(data), Tmin, Tmax)
    if CAS in mu_data_VDI_PPDS_8:
        data = mu_data_VDI_PPDS_8[CAS].tolist()[1:]
        data.reverse()
        add_model(horner_polynomial.from_kwargs({'coeffs':data}),
                  name='VDI-PPDS')
    # data = (Tc, Pc, Zc, MW)
    # if all(data):
    #     Tmin = 0; Tmax = 1e3
    #     add_model(Lucas_gas.from_args(data), Tmin, Tmax,
    #               name='Lucas')
    data = (Tc, Pc, MW)
    if all(data):
        Tmin = 0; Tmax = 5e3
        add_model(Gharagheizi_gas_viscosity.from_args(data), Tmin, Tmax,
                  name='Gharagheizi')
        add_model(Yoon_Thodos.from_args(data), Tmin, Tmax)
        add_model(Stiel_Thodos.from_args(data), Tmin, Tmax)
        # Intelligently set limit
        # GHARAGHEIZI turns nonsensical at ~15 K, YOON_THODOS fine to 0 K,
        # same as STIEL_THODOS

viscosity_handle = PhaseTPHandleBuilder('mu', None,
                                        viscosity_liquid_handle,
                                        viscosity_gas_handle)
