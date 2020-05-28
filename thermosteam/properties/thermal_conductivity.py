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
All data and methods for estimating a chemical's thermal conductivity.

References
----------
.. [1] Scheffy, W. J., and E. F. Johnson. "Thermal Conductivities of
    Liquids at High Temperatures." Journal of Chemical & Engineering Data
    6, no. 2 (April 1, 1961): 245-49. doi:10.1021/je60010a019
.. [2] Riedel, L.: Chem. Ing. Tech., 21, 349 (1949); 23: 59, 321, 465 (1951)
.. [3] Maejima, T., private communication, 1973
.. [4] Properties of Gases and Liquids", 3rd Ed., McGraw-Hill, 1977
.. [5] Lakshmi, D. S., and D. H. L. Prasad. "A Rapid Estimation Method for
    Thermal Conductivity of Pure Liquids." The Chemical Engineering Journal
    48, no. 3 (April 1992): 211-14. doi:10.1016/0300-9467(92)80037-B
.. [6] Gharagheizi, Farhad, Poorandokht Ilani-Kashkouli, Mehdi Sattari,
    Amir H. Mohammadi, Deresh Ramjugernath, and Dominique Richon.
    "Development of a General Model for Determination of Thermal
    Conductivity of Liquid Chemical Compounds at Atmospheric Pressure."
    AIChE Journal 59, no. 5 (May 1, 2013): 1702-8. doi:10.1002/aic.13938
.. [7] Di Nicola, Giovanni, Eleonora Ciarrocchi, Gianluca Coccia, and
    Mariano Pierantozzi. "Correlations of Thermal Conductivity for
    Liquid Refrigerants at Atmospheric Pressure or near Saturation."
    International Journal of Refrigeration. 2014.
    doi:10.1016/j.ijrefrig.2014.06.003
.. [8] Bahadori, Alireza, and Saeid Mokhatab. "Estimating Thermal
    Conductivity of Hydrocarbons." Chemical Engineering 115, no. 13
    (December 2008): 52-54
.. [9] Mersmann, Alfons, and Matthias Kind. "Prediction of Mechanical and 
   Thermal Properties of Pure Liquids, of Critical Data, and of Vapor 
   Pressure." Industrial & Engineering Chemistry Research, January 31, 
   2017. https://doi.org/10.1021/acs.iecr.6b04323.
.. [10] Missenard, F. A., Thermal Conductivity of Organic Liquids of a
       Series or a Group of Liquids , Rev. Gen.Thermodyn., 101 649 (1970).
.. [11] Danner, Ronald P, and Design Institute for Physical Property Data.
   Manual for Predicting Chemical Process Design Data. New York, N.Y, 1982.
.. [12] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
    New York: McGraw-Hill Professional, 2000.
.. [13] Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E.
    Properties of Gases and Liquids. McGraw-Hill Companies, 1987.
.. [14] Chung, Ting Horng, Lloyd L. Lee, and Kenneth E. Starling.
    "Applications of Kinetic Gas Theories and Multiparameter Correlation for
    Prediction of Dilute Gas Viscosity and Thermal Conductivity."
    Industrial & Engineering Chemistry Fundamentals 23, no. 1
    (February 1, 1984): 8-13. doi:10.1021/i100013a002
.. [15] Ely, James F., and H. J. M. Hanley. "Prediction of Transport
    Properties. 2. Thermal Conductivity of Pure Fluids and Mixtures."
    Industrial & Engineering Chemistry Fundamentals 22, no. 1 (February 1,
    1983): 90-97. doi:10.1021/i100009a016.
.. [16] Bahadori, Alireza, and Saeid Mokhatab. "Estimating Thermal
    Conductivity of Hydrocarbons." Chemical Engineering 115, no. 13
    (December 2008): 52-54
.. [17] Stiel, Leonard I., and George Thodos. "The Thermal Conductivity of
    Nonpolar Substances in the Dense Gaseous and Liquid Regions." AIChE
    Journal 10, no. 1 (January 1, 1964): 26-30. doi:10.1002/aic.690100114.
.. [18] Chung, Ting Horng, Mohammad Ajlan, Lloyd L. Lee, and Kenneth E.
    Starling. "Generalized Multiparameter Correlation for Nonpolar and Polar
    Fluid Transport Properties." Industrial & Engineering Chemistry Research
    27, no. 4 (April 1, 1988): 671-79. doi:10.1021/ie00076a024.
   
"""
import numpy as np
import flexsolve as flx
from scipy.interpolate import interp2d
from ..base import (InterpolatedTDependentModel, 
                    ThermoModelHandle,
                    TPDependentHandleBuilder, 
                    PhaseTPHandleBuilder, 
                    functor)
from .._constants import R, N_A, k
from math import log, exp
from ..functional import horner_polynomial
from .data import (VDI_saturation_dict,
                   VDI_tabular_data,
                   # kappa_data_Perrys2_314,
                   kappa_data_Perrys2_315,
                   kappa_data_VDI_PPDS_9,
                   # kappa_data_VDI_PPDS_10,
)
from .dippr import DIPPR_EQ100

__all__ = ('thermal_conductivity_handle',
           'Sheffy_Johnson', 
           'Sato_Riedel', 
           'Lakshmi_Prasad', 
           'Gharagheizi_liquid',
           'Nicola', 
           'Bahadori_liquid', 
           'Mersmann_Kind_liquid',
           'DIPPR9G', 
           'Missenard', 
           'Eucken', 
           'Eucken_modified',
           'DIPPR9B_linear',
           'DIPPR9B_monoatomic', 
           'DIPPR9B_nonlinear', 
           'Chung', 
           'Eli_Hanley',
           'Gharagheizi_gas', 
           'Bahadori_gas', 
           'Stiel_Thodos_dense',
           'Eli_Hanley_dense', 
           'Chung_dense',
)


### Purely CSP Methods - Liquids

@functor(var='kappa.l')
def Sheffy_Johnson(T, MW, Tm):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the Sheffy 
    Johnson method, as described in [1]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tm : float
        Melting point temperature [K].
    
    Notes
    -----
    Calculates the thermal conductivity of a liquid as a function of
    temperature using the Sheffy-Johnson (1961) method. Requires
    temperature, molecular weight, and melting point:
    
    .. math::
        k = 1.951 \frac{1-0.00126(T-T_m)}{T_m^{0.216}MW^{0.3}}
    
    The origin of this equation has been challenging to trace. It is
    presently unknown, and untested.
    
    Examples
    --------
    >>> f = Sheffy_Johnson(47, 280)
    >>> f(300)
    0.17740150413112196
    
    """
    return 1.951*(1 - 0.00126*(T - Tm))/(Tm**0.216*MW**0.3)

@functor(var='kappa.l')
def Sato_Riedel(T, MW, Tb, Tc):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the Sato 
    Riedel method, as described in [4]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tb : float
        Boiling point temperature [K].
    Tc : float
        Critical point temperature [K].
    
    Notes
    -----
    Calculates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of Sato-Riedel [2]_, [3]_, published in
    Reid [4]_. Requires temperature, molecular weight, and boiling and critical
    temperatures.
    
    .. math::
        k = \frac{1.1053}{\sqrt{MW}}\frac{3+20(1-T_r)^{2/3}}
        {3+20(1-T_{br})^{2/3}}
        
    This equation has a complicated history. It is proposed by Reid [3]_.
    Limited accuracy should be expected. Uncheecked.
    
    Examples
    --------
    >>> f = Sato_Riedel(47, 390, 520)
    >>> f(300)
    0.21037692461337687
    
    """
    Tr = T/Tc
    Tbr = Tb/Tc
    return 1.1053*(3. + 20.*(1 - Tr)**(2/3.))*MW**-0.5/(3. + 20.*(1 - Tbr)**(2/3.))

@functor(var='kappa.l')
def Lakshmi_Prasad(T, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the 
    Lakshmi Prasad method, as described in [5]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Estimates thermal conductivity of pure liquids as a function of
    temperature using a reference fluid approach. Low accuracy but quick.
    Developed using several organic fluids.
    
    .. math::
        \lambda = 0.0655-0.0005T + \frac{1.3855-0.00197T}{M^{0.5}}
        
    This equation returns negative numbers at high T sometimes.
    This equation is one of those implemented by DDBST.
    If this results in a negative thermal conductivity, no value is returned.
    
    Examples
    --------
    >>> f = Lakshmi_Prasad(100)
    >>> f(273.15)
    0.013664450000000009
    
    """
    return 0.0655 - 0.0005*T + (1.3855 - 0.00197*T)*MW**-0.5

def Tmax_Lakshmi_Prasad(MW):
    """
    Returns the maximum temperature at which the Lakshmi Prasad method is
    valid.
    """
    T_max = flx.aitken_secant(Lakshmi_Prasad.function, 298.15, args=(MW,))
    return T_max - 10 # As an extra precaution

@functor(var='kappa.l')
def Gharagheizi_liquid(T, MW, Tb, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the 
    Gharagheizi liquid method, as described in [6]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tb : float
        Boiling point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    
    Notes
    -----
    Estimates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of Gharagheizi [6]_. A  convoluted
    method claiming high-accuracy and using only statistically significant
    variable following analalysis.
    Requires temperature, molecular weight, boiling temperature and critical
    pressure and acentric factor.
    
    .. math::
        &k = 10^{-4}\left[10\omega + 2P_c-2T+4+1.908(T_b+\frac{1.009B^2}{MW^2})
        +\frac{3.9287MW^4}{B^4}+\frac{A}{B^8}\right]
        
        &A = 3.8588MW^8(1.0045B+6.5152MW-8.9756)
        
        &B = 16.0407MW+2T_b-27.9074
    
    Pressure is internally converted into bar, as used in the original equation.
    This equation was derived with 19000 points representing 1640 unique compounds.
    
    Examples
    --------
    >>> f = Gharagheizi_liquid(40, 350, 1E6, 0.27)
    >>> f(300)
    0.2171113029534838
    
    """
    Pc = Pc/1E5
    B = 16.0407*MW + 2.*Tb - 27.9074
    A = 3.8588*MW**8*(1.0045*B + 6.5152*MW - 8.9756)
    return 1E-4*(10.*omega + 2.*Pc - 2.*T + 4. + 1.908*(Tb + 1.009*B*B/(MW*MW))
        + 3.9287*MW**4*B**-4 + A*B**-8)


@functor(var='kappa.l')
def Nicola(T, MW, Tc, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the Nicola 
    method, as described in [7]_.
    
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
    Estimates the thermal conductivity of a liquid as a function of
    temperature using the CSP method of [1]_. A statistically derived
    equation using any correlated terms.
    Requires temperature, molecular weight, critical temperature and pressure,
    and acentric factor.
    
    .. math::
        \frac{\lambda}{0.5147 W/m/K} = -0.2537T_r+\frac{0.0017Pc}{\text{bar}}
        +0.1501 \omega + \left(\frac{1}{MW}\right)^{-0.2999}
    
    A statistical correlation. A revision of an original correlation.
    
    Examples
    --------
    >>> f = Nicola(142.3, 611.7, 2110000.0, 0.49)
    >>> f(300)
    0.10863821554584034
    
    """
    Tr = T/Tc
    Pc = Pc/1E5
    return 0.5147*(-0.2537*Tr + 0.0017*Pc + 0.1501*omega + (1./MW)**0.2999)

@functor(var='kappa.l')
def Bahadori_liquid(T, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the Bahadori
    liquid method, as described in [8]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Estimates the thermal conductivity of parafin liquid hydrocarbons.
    Fits their data well, and is useful as only MW is required.
    X is the Molecular weight, and Y the temperature.
    
    .. math::
        K = a + bY + CY^2 + dY^3
        
        a = A_1 + B_1 X + C_1 X^2 + D_1 X^3
        
        b = A_2 + B_2 X + C_2 X^2 + D_2 X^3
        
        c = A_3 + B_3 X + C_3 X^2 + D_3 X^3
        
        d = A_4 + B_4 X + C_4 X^2 + D_4 X^3
    
    The accuracy of this equation has not been reviewed.
    
    Examples
    --------
    Data point from [9]_.
    >>> f = Bahadori_liquid(170)
    >>> f(273.15)
    0.14274278108272603
    
    """
    A = [-6.48326E-2, 2.715015E-3, -1.08580E-5, 9.853917E-9]
    B = [1.565612E-2, -1.55833E-4, 5.051114E-7, -4.68030E-10]
    C = [-1.80304E-4, 1.758693E-6, -5.55224E-9, 5.201365E-12]
    D = [5.880443E-7, -5.65898E-9, 1.764384E-11, -1.65944E-14]
    X = MW
    Y = T
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3

@functor(var='kappa.l')
def Mersmann_Kind_liquid(T, MW, Tc, Vc, atoms):
    r"""
    Create a functor of temperature (T; in K) that estimates the liquid 
    thermal conductivity (kappa.l; in W/m/K) of a chemical using the Mersmann
    Kind method, as described in [9]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Vc : float
        Critical point volume [m^3/mol].
    atoms : Dict[str, int]
        Atom-count pairs [-].
    
    Notes
    -----
    Estimates the thermal conductivity of organic liquid substances
    according to the method of [9]_.
    
    .. math::
        \lambda^* = \frac{\lambda\cdot V_c^{2/3}\cdot T_c\cdot \text{MW}^{0.5}}
        {(k\cdot T_c)^{1.5}\cdot N_A^{7/6}}
        
        \lambda^* = \frac{2}{3}\left(n_a + 40\sqrt{1-T_r}\right)
        
    
    In the equation, all quantities must be in SI units but N_A is in a kmol
    basis and Vc is in units of (m^3/kmol); this is converted internally.
    
    Examples
    --------
    Dodecane at 400 K:
        
    >>> f = Mersmann_Kind_liquid(170.33484, 658.0, 
    ... 0.000754, {'C': 12, 'H': 26})
    >>> f(400)
    0.08952713798442789
    
    """
    na = sum(atoms.values())
    lambda_star = 2/3.*(na + 40.*(1. - T/Tc)**0.5)
    Vc = Vc*1000 # m^3/mol to m^3/kmol
    N_A2 = N_A*1000 # Their avogadro's constant is per kmol
    kl = lambda_star*(k*Tc)**1.5*N_A2**(7/6.)*Vc**(-2/3.)/Tc*MW**-0.5
    return kl

### Thermal Conductivity of Dense Liquids

@functor(var='kappa.l')
def DIPPR9G(T, P, Tc, Pc, k_l):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that 
    estimates the liquid thermal conductivity (kappa.l; in W/m/K) of a 
    chemical using the DIPPR9G method, as described in [11]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    k_l : float
        Regressed coefficient.
    
    Notes
    -----
    Adjusts for pressure. the thermal conductivity of a liquid using an
    emperical formula based on [10]_, but as given in [11]_.
    
    .. math::
        k = k^* \left[ 0.98 + 0.0079 P_r T_r^{1.4} + 0.63 T_r^{1.2}
        \left( \frac{P_r}{30 + P_r}\right)\right]
    
    This equation is entrely dimensionless; all dimensions cancel.
    The original source has not been reviewed.
    This is DIPPR Procedure 9G: Method for the Thermal Conductivity of Pure
    Nonhydrocarbon Liquids at High Pressures
    
    Examples
    --------
    From [11]_, for butyl acetate.
    
    >>> f = DIPPR9G(3.92E7, 579.15, 3.212E6, 7.085E-2)
    >>> f(515.05)
    0.0864419738671184
    
    """
    Tr = T/Tc
    Pr = P/Pc
    if isinstance(k_l, ThermoModelHandle): k_l = k_l.at_T(T)
    return k_l*(0.98 + 0.0079*Pr*Tr**1.4 + 0.63*Tr**1.2*(Pr/(30. + Pr)))


Trs_Missenard = [0.8, 0.7, 0.6, 0.5]
Prs_Missenard = [1, 5, 10, 50, 100, 200]
Qs_Missenard = np.array([[0.036, 0.038, 0.038, 0.038, 0.038, 0.038],
                         [0.018, 0.025, 0.027, 0.031, 0.032, 0.032],
                         [0.015, 0.020, 0.022, 0.024, 0.025, 0.025],
                         [0.012, 0.0165, 0.017, 0.019, 0.020, 0.020]])
Qfunc_Missenard = interp2d(Prs_Missenard, Trs_Missenard, Qs_Missenard)

@functor(var='kappa.l')
def Missenard(T, P, Tc, Pc, k_l):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) 
    that estimates the liquid thermal conductivity (kappa.l; in W/m/K)
    of a chemical using the Missenard method, as described in [12]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    k_l : float
        Regressed coefficient.
        
    Notes
    -----
    Adjusts for pressure. the thermal conductivity of a liquid using an
    emperical formula based on [10]_, but as given in [12]_.
    
    .. math::
        \frac{k}{k^*} = 1 + Q P_r^{0.7}
    
    This equation is entirely dimensionless; all dimensions cancel.
    An interpolation routine is used here from tabulated values of Q.
    The original source has not been reviewed.
    
    Examples
    --------
    Example from [12]_, toluene; matches.
    
    >>> f = Missenard(6330E5, 591.8, 41E5, 0.129)
    >>> f(304.)
    0.2198375777069657
    
    """
    Tr = T/Tc
    Pr = P/Pc
    Q = float(Qfunc_Missenard(Pr, Tr))
    if isinstance(k_l, ThermoModelHandle): k_l = k_l.at_T(T)
    return k_l*(1. + Q*Pr**0.7)

@TPDependentHandleBuilder('kappa.l')
def thermal_conductivity_liquid_handle(handle, CAS, MW, Tm, Tb, Tc, Pc, omega):
    add_model = handle.add_model
    if all((Tc, Pc)):
        data = (Tc, Pc, handle)
        add_model(DIPPR9G.from_args(data))
        add_model(Missenard.from_args(data))
    if CAS in kappa_data_Perrys2_315:
        _, C1, C2, C3, C4, C5, Tmin, Tmax = kappa_data_Perrys2_315[CAS]
        data = (C1, C2, C3, C4, C5)
        add_model(DIPPR_EQ100.from_args(data), Tmin, Tmax)
    if CAS in kappa_data_VDI_PPDS_9:
        _,  A, B, C, D, E = kappa_data_VDI_PPDS_9[CAS]
        add_model(horner_polynomial.from_kwargs({'coeffs':(E, D, C, B, A)}))
    if CAS in VDI_saturation_dict:
        Ts, Ys = VDI_tabular_data(CAS, 'K (l)')
        Tmin = Ts[0]
        Tmax = Ts[-1]
        add_model(InterpolatedTDependentModel(Ts, Ys, Tmin=Tmin, Tmax=Tmax))
    data = (MW, Tb, Pc, omega)
    if all(data):
        add_model(Gharagheizi_liquid.from_args(data), Tb, Tc)
    data = (MW, Tc, Pc, omega)
    if all(data):
        add_model(Nicola.from_args(data))
    data = (MW, Tb, Tc)
    if all(data):
        add_model(Sato_Riedel.from_args(data))
    data = (MW, Tm)
    if all(data):
        # Works down to 0, has a nice limit at T = Tm+793.65 from Sympy
        add_model(Sheffy_Johnson.from_args(data), 0, 793.65)
    data = (MW,)
    if MW:
        Tmax = Tmax_Lakshmi_Prasad(MW)
        add_model(Lakshmi_Prasad.from_args(data), 0., Tmax)
        add_model(Bahadori_liquid.from_args(data))

### Thermal Conductivity of Gases

@functor(var='kappa.g')
def Eucken(T, MW, Cn, mu_g):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas thermal 
    conductivity (kappa.g; in W/m/K) of a chemical using the Eucken method, 
    as described in [13]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Cn : float
        Molar heat capacity [J/mol/K].
    mu_g : float
        Gas hydrolic viscosity [-].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Eucken [13]_.
    
    .. math::
        C_v = C_n - R
        
        \frac{\lambda M}{\eta C_v} = 1 + \frac{9/4}{C_v/R}
    
    Temperature dependence is introduced via heat capacity and viscosity.
    A theoretical equation. No original author located.
    MW internally converted to kg/g-mol.
    
    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [13]_.
    
    >>> f = Eucken(MW=72.151, Cn=135.9, mu=8.77E-6)
    >>> f(373.15)
    0.018792644287722975
    
    """
    if callable(Cn): Cn = Cn(T)
    if callable(mu_g): mu_g = mu_g.at_T(T)
    Cv = Cn - R
    MW = MW/1000.
    return (1. + 9/4./(Cv/R))*mu_g*Cv/MW

@functor(var='kappa.g')
def Eucken_modified(T, MW, Cn, mu):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas 
    thermal conductivity (kappa.g; in W/m/K) of a chemical using the 
    Eucken modified method, as described in [13]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Cn : float
        Molar heat capacity [J/mol/K].
    mu : float
        Hydrolic viscosity [Pa*s].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas as a function of
    temperature using the Modified CSP method of Eucken [1]_.
    
    .. math::
        \frac{\lambda M}{\eta C_v} = 1.32 + \frac{1.77}{C_v/R}
    
    Temperature dependence is introduced via heat capacity and viscosity.
    A theoretical equation. No original author located.
    MW internally converted to kg/g-mol.
    
    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [1]_.
    
    >>> f = Eucken_modified(MW=72.151, Cvm=135.9, mu=8.77E-6)
    >>> f(373.15)
    0.023593536999201956
    
    """
    if callable(Cn): Cn = Cn(T)
    if callable(mu): mu = mu.at_T(T)
    Cv = Cn - R
    MW = MW/1000.
    return (1.32 + 1.77/(Cv/R))*mu*Cv/MW

@functor(var='kappa.g')
def DIPPR9B_linear(T, MW, Cn, mu, Tc):
    if callable(Cn): Cn = Cn(T)
    if callable(mu): mu = mu.at_T(T)
    Cv = 1000 * (Cn - R) # J/mol/K to J/kmol/K
    Tr = T/Tc
    return mu/MW*(1.30*Cv + 14644 - 2928.80/Tr)

@functor(var='kappa.g')    
def DIPPR9B_monoatomic(T, MW, Cn, mu):
    if callable(Cn): Cn = Cn(T)
    if callable(mu): mu = mu.at_T(T)
    Cv = 1000 * (Cn - R) # J/mol/K to J/kmol/K
    return 2.5*mu*Cv/MW

@functor(var='kappa.g')
def DIPPR9B_nonlinear(T, MW, Cn, mu):
    if callable(Cn): Cn = Cn(T)
    if callable(mu): mu = mu.at_T(T)
    Cv = 1000 * (Cn - R) # J/mol/K to J/kmol/K
    return mu/MW*(1.15*Cv + 16903.36)

@functor(var='kappa.g')
def Chung(T, MW, Tc, omega, Cn, mu):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas thermal
    conductivity (kappa.g; in W/m/K) of a chemical using the Chung method, 
    as described in [14]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    omega : float
        Acentric factor [-].
    Cn : float
        Molar heat capacity [J/mol/K].
    mu : float
        Hydrolic viscosity [Pa*s].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Chung [14]_.
    
    .. math::
        \frac{\lambda M}{\eta C_v} = \frac{3.75 \Psi}{C_v/R}
        
        \Psi = 1 + \alpha \left\{[0.215+0.28288\alpha-1.061\beta+0.26665Z]/
        [0.6366+\beta Z + 1.061 \alpha \beta]\right\}
        
        \alpha = \frac{C_v}{R}-1.5
        
        \beta = 0.7862-0.7109\omega + 1.3168\omega^2
        
        Z=2+10.5T_r^2
    
    MW internally converted to kg/g-mol.
    
    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [13]_.
    
    >>> f = Chung(MW=72.151, Tc=460.4, omega=0.227, Cvm=135.9, mu=8.77E-6)
    >>> f(T=373.15)
    0.023015653729496946
    
    """
    if callable(Cn): Cn = Cn(T)
    if callable(mu): mu = mu.at_T(T)
    Cv = Cn - R 
    MW = MW/1000. # g/mol to kg/mol
    alpha = Cv/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)
                      /(0.6366 + beta*Z + 1.061*alpha*beta))
    return 3.75*psi*R/MW*mu

@functor(var='kappa.g')
def Eli_Hanley(T, MW, Tc, Vc, Zc, omega, Cn):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas thermal 
    conductivity (kappa.g; in W/m/K) of a chemical using the Eli Hanley 
    method, as described in [13]_ [15]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Vc : float
        Critical point volume [m^3/mol].
    Zc : float
        Critical compressibility [-].
    omega : float
        Acentric factor [-].
    Cn : float
        Molar heat capacity [J/mol/K].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas as a function of
    temperature using the reference fluid method of Eli and Hanley [1]_ as
    shown in [13]_.
    
    .. math::
        
        \lambda = \lambda^* + \frac{\eta^*}{MW}(1.32)\left(C_v - \frac{3R}{2}\right)
        
        Tr = \text{min}(Tr, 2)
        
        \theta = 1 + (\omega-0.011)\left(0.56553 - 0.86276\ln Tr - \frac{0.69852}{Tr}\right)
        
        \psi = [1 + (\omega - 0.011)(0.38560 - 1.1617\ln Tr)]\frac{0.288}{Z_c}
        
        f = \frac{T_c}{190.4}\theta
        
        h = \frac{V_c}{9.92E-5}\psi
        
        T_0 = T/f
        
        \eta_0^*(T_0)= \sum_{n=1}^9 C_n T_0^{(n-4)/3}
        
        \theta_0 = 1944 \eta_0
        
        \lambda^* = \lambda_0 H
        
        \eta^* = \eta^*_0 H \frac{MW}{16.04}
        
        H = \left(\frac{16.04}{MW}\right)^{0.5}f^{0.5}/h^{2/3}
    
    Reference fluid is Methane.
    MW internally converted to kg/g-mol.
    
    Examples
    --------
    2-methylbutane at low pressure, 373.15 K. Mathes calculation in [13]_.
    
    >>> f = Eli_Hanley(MW=72.151, Tc=460.4, Vc=3.06E-4, Zc=0.267,
    ... omega=0.227, Cvm=135.9)
    >>> f(T=373.15)
    0.02247951789135337
    
    """
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5, 
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1, 2.037119479E-1]
    if callable(Cn): Cn = Cn(T)
    Cv = Cn - R
    Tr = T/Tc
    if Tr > 2: Tr = 2
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Ci*T0**((i+1. - 4.)/3.) for i, Ci in enumerate(Cs)])
    k0 = 1944*eta0

    H = (16.04/MW)**0.5*f**0.5*h**(-2/3.)
    etas = eta0*H*MW/16.04
    ks = k0*H
    return ks + etas/(MW/1000.)*1.32*(Cv - 1.5*R)

@functor(var='kappa.g')
def Gharagheizi_gas(T, MW, Tb, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas thermal
    conductivity (kappa.g; in W/m/K) of a chemical using the Gharagheizi gas 
    method, as described in [6]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tb : float
        Boiling point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas as a function of
    temperature using the CSP method of Gharagheizi [6]_. A  convoluted
    method claiming high-accuracy and using only statistically significant
    variable following analalysis.
    Requires temperature, molecular weight, boiling temperature and critical
    pressure and acentric factor.
    
    .. math::
        k = 7.9505\times 10^{-4} + 3.989\times 10^{-5} T
        -5.419\times 10^-5 M + 3.989\times 10^{-5} A
       
        A = \frac{\left(2\omega + T - \frac{(2\omega + 3.2825)T}{T_b} + 3.2825\right)}{0.1MP_cT}
        \times (3.9752\omega + 0.1 P_c + 1.9876B + 6.5243)^2
    
    Pressure is internally converted into 10*kPa but author used correlation with
    kPa; overall, errors have been corrected in the presentation of the formula.
    This equation was derived with 15927 points and 1574 compounds.
    Example value from [1]_ is the first point in the supportinf info, for CH4.
    
    Examples
    --------
    >>> f = Gharagheizi_gas(16.04246, 111.66, 4599000.0, 0.0115478000)
    >>> f(580.)
    0.09594861261873211
    
    """
    Pc = Pc/1E4
    B = T + (2.*omega + 2.*T - 2.*T*(2.*omega + 3.2825)/Tb + 3.2825)/(2*omega + T - T*(2*omega+3.2825)/Tb + 3.2825) - T*(2*omega+3.2825)/Tb
    A = (2*omega + T - T*(2*omega + 3.2825)/Tb + 3.2825)/(0.1*MW*Pc*T) * (3.9752*omega + 0.1*Pc + 1.9876*B + 6.5243)**2
    return 7.9505E-4 + 3.989E-5*T - 5.419E-5*MW + 3.989E-5*A

@functor(var='kappa.g')
def Bahadori_gas(T, MW):
    r"""
    Create a functor of temperature (T; in K) that estimates the gas thermal 
    conductivity (kappa.g; in W/m/K) of a chemical using the Bahadori gas 
    method, as described in [16]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    
    Notes
    -----
    Estimates the thermal conductivity of hydrocarbons gases at low P.
    Fits their data well, and is useful as only MW is required.
    Y is the Molecular weight, and X the temperature.
    
    .. math::
        K = a + bY + CY^2 + dY^3
        
        a = A_1 + B_1 X + C_1 X^2 + D_1 X^3
        
        b = A_2 + B_2 X + C_2 X^2 + D_2 X^3
        
        c = A_3 + B_3 X + C_3 X^2 + D_3 X^3
        
        d = A_4 + B_4 X + C_4 X^2 + D_4 X^3
    
    The accuracy of this equation has not been reviewed.
    Examples
    --------
    >>> f = Bahadori_gas(20) # Point from article
    >>> f(313.15)
    0.031968165337873326
    
    """
    A = [4.3931323468E-1, -3.88001122207E-2, 9.28616040136E-4, -6.57828995724E-6]
    B = [-2.9624238519E-3, 2.67956145820E-4, -6.40171884139E-6, 4.48579040207E-8]
    C = [7.54249790107E-6, -6.46636219509E-7, 1.5124510261E-8, -1.0376480449E-10]
    D = [-6.0988433456E-9, 5.20752132076E-10, -1.19425545729E-11, 8.0136464085E-14]
    X, Y = T, MW
    a = A[0] + B[0]*X + C[0]*X**2 + D[0]*X**3
    b = A[1] + B[1]*X + C[1]*X**2 + D[1]*X**3
    c = A[2] + B[2]*X + C[2]*X**2 + D[2]*X**3
    d = A[3] + B[3]*X + C[3]*X**2 + D[3]*X**3
    return a + b*Y + c*Y**2 + d*Y**3


### Thermal Conductivity of dense gases

@functor(var='kappa.g')
def Stiel_Thodos_dense(T,P, MW, Tc, Pc, Vc, Zc, V_g, k_g):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that 
    estimates the gas thermal conductivity (kappa.g; in W/m/K) of a chemical
    using the Stiel Thodos dense method, as described in [17]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    Vc : float
        Critical point volume [m^3/mol].
    Zc : float
        Critical compressibility [-].
    V_g : float
        Gas molar volume [-].
    k_g : float
        Regressed coefficient.
    
    Notes
    -----
    Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using difference method of Stiel and Thodos [1]_
    as shown in [13]_.
    
    if \rho_r < 0.5:
    
    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]
    
    if 0.5 < \rho_r < 2.0:
    
    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]
    
    if 2 < \rho_r < 2.8:
    
    .. math::
        (\lambda-\lambda^\circ)\Gamma Z_c^5=1.22\times 10^{-2} [\exp(0.535 \rho_r)-1]
        \Gamma = 210 \left(\frac{T_cMW^3}{P_c^4}\right)^{1/6}
    
    Pc is internally converted to bar.
    
    Examples
    --------
    >>> f = Stiel_Thodos_dense(MW=44.013, Tc=309.6, Pc=72.4E5,
    ... Vc=97.4E-6, Zc=0.274, Vm=144E-6, kg=2.34E-2)
    >>> f(T=378.15)
    0.041245574404863684
    
    """
    if callable(V_g): V_g = V_g(T, P)
    if isinstance(k_g, ThermoModelHandle): k_g = k_g(T, P)
    gamma = 210*(Tc*MW**3./(Pc/1E5)**4)**(1/6.)
    rhor = Vc/V_g
    if rhor < 0.5:
        term = 1.22E-2*(exp(0.535*rhor) - 1.)
    elif rhor < 2:
        term = 1.14E-2*(exp(0.67*rhor) - 1.069)
    else:
        # Technically only up to 2.8
        term = 2.60E-3*(exp(1.155*rhor) + 2.016)
    diff = term/Zc**5/gamma
    k_g = k_g + diff
    return k_g

@functor(var='kappa.g')
def Eli_Hanley_dense(T, P, MW, Tc, Vc, Zc, omega, Cn, V_g):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that 
    estimates the gas thermal conductivity (kappa.g; in W/m/K) of a chemical 
    using the Eli Hanley dense method, as described in [13]_ [15]_.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Vc : float
        Critical point volume [m^3/mol].
    Zc : float
        Critical compressibility [-].
    omega : float
        Acentric factor [-].
    Cn : float
        Molar heat capacity [J/mol/K].
    V_g : float
        Gas molar volume [-].
    
    Notes
    -----
    Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using the reference fluid method of Eli and
    Hanley [15]_ as shown in [13]_.
    
    .. math::
        Tr = min(Tr, 2)
        
        Vr = min(Vr, 2)
        
        f = \frac{T_c}{190.4}\theta
        
        h = \frac{V_c}{9.92E-5}\psi
        
        T_0 = T/f
        
        \rho_0 = \frac{16.04}{V}h
        
        \theta = 1 + (\omega-0.011)\left(0.09057 - 0.86276\ln Tr + \left(
        0.31664 - \frac{0.46568}{Tr}\right) (V_r - 0.5)\right)
        
        \psi = [1 + (\omega - 0.011)(0.39490(V_r - 1.02355) - 0.93281(V_r -
        0.75464)\ln T_r]\frac{0.288}{Z_c}
        
        \lambda_1 = 1944 \eta_0
        
        \lambda_2 = \left\{b_1 + b_2\left[b_3 - \ln \left(\frac{T_0}{b_4}
        \right)\right]^2\right\}\rho_0
        
        \lambda_3 = \exp\left(a_1 + \frac{a_2}{T_0}\right)\left\{\exp[(a_3 +
        \frac{a_4}{T_0^{1.5}})\rho_0^{0.1} + (\frac{\rho_0}{0.1617} - 1)
        \rho_0^{0.5}(a_5 + \frac{a_6}{T_0} + \frac{a_7}{T_0^2})] - 1\right\}
        
        \lambda^{**} = [\lambda_1 + \lambda_2 + \lambda_3]H
        
        H = \left(\frac{16.04}{MW}\right)^{0.5}f^{0.5}/h^{2/3}
        
        X = \left\{\left[1 - \frac{T}{f}\left(\frac{df}{dT}\right)_v \right]
        \frac{0.288}{Z_c}\right\}^{1.5}
        
        \left(\frac{df}{dT}\right)_v = \frac{T_c}{190.4}\left(\frac{d\theta}
        {d T}\right)_v
        
        \left(\frac{d\theta}{d T}\right)_v = (\omega-0.011)\left[
        \frac{-0.86276}{T} + (V_r-0.5)\frac{0.46568T_c}{T^2}\right]
    
    Reference fluid is Methane.
    MW internally converted to kg/g-mol.
    
    Examples
    --------
    >>> f = Eli_Hanley_dense(MW=42.081, Tc=364.9, Vc=1.81E-4, Zc=0.274,
    ... omega=0.144, Cvm=82.70, Vm=1.721E-4)
    >>> f(T=473.)
    0.06038475936515042
    
    """
    Cs = [2.907741307E6, -3.312874033E6, 1.608101838E6, -4.331904871E5,
          7.062481330E4, -7.116620750E3, 4.325174400E2, -1.445911210E1,
          2.037119479E-1]
    Tr = T/Tc
    if Tr > 2:
        Tr = 2
    if callable(V_g): V_g = V_g(T, P)
    if callable(Cn): Cn = Cn(T)
    Cvm = Cn - R
    Vr = V_g/Vc
    if Vr > 2:
        Vr = 2
    theta = 1 + (omega - 0.011)*(0.09057 - 0.86276*log(Tr) + (0.31664 - 0.46568/Tr)*(Vr-0.5))
    psi = (1 + (omega-0.011)*(0.39490*(Vr-1.02355) - 0.93281*(Vr-0.75464)*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    rho0 = 16.04/(V_g*1E6)*h  # Vm must be in cm^3/mol here.
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    k1 = 1944*eta0
    b1 = -0.25276920E0
    b2 = 0.334328590E0
    b3 = 1.12
    b4 = 0.1680E3
    k2 = (b1 + b2*(b3 - log(T0/b4))**2)/1000.*rho0

    a1 = -7.19771
    a2 = 85.67822
    a3 = 12.47183
    a4 = -984.6252
    a5 = 0.3594685
    a6 = 69.79841
    a7 = -872.8833

    k3 = exp(a1 + a2/T0)*(exp((a3 + a4/T0**1.5)*rho0**0.1 + (rho0/0.1617 - 1)*rho0**0.5*(a5 + a6/T0 + a7/T0**2)) - 1)/1000.

    if T/Tc > 2:
        dtheta = 0
    else:
        dtheta = (omega - 0.011)*(-0.86276/T + (Vr-0.5)*0.46568*Tc/T**2)
    dfdT = Tc/190.4*dtheta
    X = ((1 - T/f*dfdT)*0.288/Zc)**1.5

    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    ks = (k1*X + k2 + k3)*H

    ### Uses calculations similar to those for pure species here
    theta = 1 + (omega - 0.011)*(0.56553 - 0.86276*log(Tr) - 0.69852/Tr)
    psi = (1 + (omega-0.011)*(0.38560 - 1.1617*log(Tr)))*0.288/Zc
    f = Tc/190.4*theta
    h = Vc/9.92E-5*psi
    T0 = T/f
    eta0 = 1E-7*sum([Cs[i]*T0**((i+1-4)/3.) for i in range(len(Cs))])
    H = (16.04/MW)**0.5*f**0.5/h**(2/3.)
    etas = eta0*H*MW/16.04
    k = ks + etas/(MW/1000.)*1.32*(Cvm-3*R/2.)
    return k

@functor(var='kappa.g')
def Chung_dense(T, P, MW, Tc, Vc, omega, Cn, V_g, mu_g, dipole, association=0):
    r"""
    Create a functor of temperature (T; in K) and pressure (P; in Pa) that 
    estimates the gas thermal conductivity (kappa.g; in W/m/K) of a chemical 
    using the Chung dense method.
    
    Parameters
    ----------
    MW : float
        Molecular weight [g/mol].
    Tc : float
        Critical point temperature [K].
    Vc : float
        Critical point volume [m^3/mol].
    omega : float
        Acentric factor [-].
    Cn : float
        Molar heat capacity [J/mol/K].
    V_g : float
        Gas molar volume [-].
    mu_g : float
        Gas hydrolic viscosity [-].
    dipole : float
        Dipole momment [Debye].
    
    Other Parameters
    ----------------
    association : float, optional
        Association factor [-]
        
    Notes
    -----
    Estimates the thermal conductivity of a gas at high pressure as a
    function of temperature using the reference fluid method of
    Chung [18]_ as shown in [12]_.
    
    .. math::
        \lambda = \frac{31.2 \eta^\circ \Psi}{M'}(G_2^{-1} + B_6 y)+qB_7y^2T_r^{1/2}G_2
        
        \Psi = 1 + \alpha \left\{[0.215+0.28288\alpha-1.061\beta+0.26665Z]/
        [0.6366+\beta Z + 1.061 \alpha \beta]\right\}
        
        \alpha = \frac{C_v}{R}-1.5
        
        \beta = 0.7862-0.7109\omega + 1.3168\omega^2
        
        Z = 2+10.5T_r^2
        
        q = 3.586\times 10^{-3} (T_c/M')^{1/2}/V_c^{2/3}
        
        y = \frac{V_c}{6V}
        
        G_1 = \frac{1-0.5y}{(1-y)^3}
        
        G_2 = \frac{(B_1/y)[1-\exp(-B_4y)]+ B_2G_1\exp(B_5y) + B_3G_1}
        {B_1B_4 + B_2 + B_3}
        
        B_i = a_i + b_i \omega + c_i \mu_r^4 + d_i \kappa
    
    MW internally converted to kg/g-mol.
    Vm internally converted to mL/mol.
    [18]_ is not the latest form as presented in [12]_.
    Association factor is assumed 0. Relates to the polarity of the gas.
    Coefficients as follows:
        
        ais = [2.4166E+0, -5.0924E-1, 6.6107E+0, 1.4543E+1, 7.9274E-1, -5.8634E+0, 9.1089E+1]
    
        bis = [7.4824E-1, -1.5094E+0, 5.6207E+0, -8.9139E+0, 8.2019E-1, 1.2801E+1, 1.2811E+2]
    
        cis = [-9.1858E-1, -4.9991E+1, 6.4760E+1, -5.6379E+0, -6.9369E-1, 9.5893E+0, -5.4217E+1]
    
        dis = [1.2172E+2, 6.9983E+1, 2.7039E+1, 7.4344E+1, 6.3173E+0, 6.5529E+1, 5.2381E+2]
    
    Examples
    --------
    >>> f = Chung_dense(MW=42.081, Tc=364.9, Vc=184.6E-6, omega=0.142,
    ... Cvm=82.67, Vm=172.1E-6, mu=134E-7, dipole=0.4)
    >>> f(T=473.)
    0.06160570379787278
    
    """
    if callable(V_g): V_g = V_g(T, P)
    if callable(mu_g): mu_g = mu_g(T, P)
    if callable(Cn): Cn = Cn(T)
    Cvm = Cn - R
    ais = [2.4166E+0, -5.0924E-1, 6.6107E+0, 1.4543E+1, 7.9274E-1, -5.8634E+0, 9.1089E+1]
    bis = [7.4824E-1, -1.5094E+0, 5.6207E+0, -8.9139E+0, 8.2019E-1, 1.2801E+1, 1.2811E+2]
    cis = [-9.1858E-1, -4.9991E+1, 6.4760E+1, -5.6379E+0, -6.9369E-1, 9.5893E+0, -5.4217E+1]
    dis = [1.2172E+2, 6.9983E+1, 2.7039E+1, 7.4344E+1, 6.3173E+0, 6.5529E+1, 5.2381E+2]
    Tr = T/Tc
    mur = 131.3*dipole/(Vc*1E6*Tc)**0.5

    # From Chung Method
    alpha = Cvm/R - 1.5
    beta = 0.7862 - 0.7109*omega + 1.3168*omega**2
    Z = 2 + 10.5*(T/Tc)**2
    psi = 1 + alpha*((0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Z)/(0.6366 + beta*Z + 1.061*alpha*beta))

    y = Vc/(6*V_g)
    B1, B2, B3, B4, B5, B6, B7 = [ais[i] + bis[i]*omega + cis[i]*mur**4 + dis[i]*association for i in range(7)]
    G1 = (1 - 0.5*y)/(1. - y)**3
    G2 = (B1/y*(1 - exp(-B4*y)) + B2*G1*exp(B5*y) + B3*G1)/(B1*B4 + B2 + B3)
    q = 3.586E-3*(Tc/(MW/1000.))**0.5/(Vc*1E6)**(2/3.)
    return 31.2*mu_g*psi/(MW/1000.)*(G2**-1 + B6*y) + q*B7*y**2*Tr**0.5*G2


@TPDependentHandleBuilder('kappa.g')
def thermal_conductivity_gas_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, dipole, Vg, Cn, mug):
    add_model = handle.add_model
    if CAS in VDI_saturation_dict:
        Ts, Ys = VDI_tabular_data(CAS, 'K (g)')
        add_model(InterpolatedTDependentModel(Ts, Ys))
    # if CAS in kappa_data_VDI_PPDS_10:
    #     _,  *data = kappa_data_VDI_PPDS_10[CAS].tolist()
    #     data.reverse()
    #     add_model(horner_polynomial.from_kwargs({'coeffs': data}))
    data = (MW, Cn, mug, Tc)
    if all(data):
        add_model(DIPPR9B_linear.from_args(data))
    data = (MW, Tb, Pc, omega)
    if all(data):
        add_model(Gharagheizi_gas.from_args(data))   
    data = (MW, Tc, omega, Cn, mug)
    if all(data):
        add_model(Chung.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cn)
    if all(data):
        add_model(Eli_Hanley.from_args(data))
    data = (MW, Tc, Vc, Zc, omega, Cn, Vg)
    if all(data):
        add_model(Eli_Hanley_dense.from_args(data))
    data = (MW, Tc, Vc, omega, Cn, Vg, mug, dipole)
    if all(data):
        add_model(Chung_dense.from_args(data))
    data = (MW, Tc, Pc, Vc, Zc, Vg, handle)
    if all(data):
        add_model(Stiel_Thodos_dense.from_args(data))
    data = (MW, Cn, mug)
    if all(data):
        add_model(Eucken_modified.from_args(data))
        add_model(Eucken.from_args(data))
    # TODO: Fix propblem with values
    # if CAS in kappa_data_Perrys2_314:
    #     _, *data, Tmin, Tmax = kappa_data_Perrys2_314[CAS]
    #     add_model(DIPPR9B_linear(data), Tmin, Tmax)

thermal_conductivity_handle = PhaseTPHandleBuilder('kappa', None,
                                                     thermal_conductivity_liquid_handle,
                                                     thermal_conductivity_gas_handle)
