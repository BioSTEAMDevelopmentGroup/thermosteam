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
Property calculation methods from the Design Institute for Physical Properties (DIPPR) [1]_.

References
----------
.. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
    DIPPR/AIChE
.. [2] Aly, Fouad A., and Lloyd L. Lee. "Self-Consistent Equations for
    Calculating the Ideal Gas Heat Capacity, Enthalpy, and Entropy." Fluid
    Phase Equilibria 6, no. 3 (January 1, 1981): 169-79.
    doi:10.1016/0378-3812(81)85002-9.

"""

__all__ = ['DIPPR_EQ100', 'DIPPR_EQ101', 'DIPPR_EQ102',
           'DIPPR_EQ104', 'DIPPR_EQ105', 'DIPPR_EQ106',
           'DIPPR_EQ107', 'DIPPR_EQ114', 'DIPPR_EQ115', 
           'DIPPR_EQ116', 'DIPPR_EQ127']

from ..base.functor import functor
from math import log, exp, sinh, cosh, atan, atanh, sqrt, tanh
from cmath import log as clog
from cmath import sqrt as csqrt
from scipy.special import hyp2f1

@functor
def DIPPR_EQ100(T, a=0., b=0., c=0., d=0., e=0., f=0., g=0.):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ100 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e,f,g : float
        Regressed coefficients.
    
    See also
    --------
    DIPPR_EQ100_derivative_by_T
    DIPPR_EQ100_integral_by_T
    DIPPR_EQ100_integral_by_T_over_T

    Notes
    -----
    DIPPR Equation # 100. Used in calculating the molar heat capacities
    of liquids and solids, liquid thermal conductivity, and solid density.
    All parameters default to zero. As this is a straightforward polynomial,
    no restrictions on parameters apply. Note that high-order polynomials like
    this may need large numbers of decimal places to avoid unnecessary error.

    .. math:: Y = a + bT + cT^2 + dT^3 + eT^4 + fT^5 + gT^6
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. All derivatives and 
    integrals are easily computed with SymPy.
    
    .. math::
        \frac{d Y}{dT} = b + 2 c T + 3 d T^{2} + 4 e T^{3} + 5 f T^{4} 
        + 6 g T^{5}
        
    .. math::
        \int Y dT = a T + \frac{b T^{2}}{2} + \frac{c T^{3}}{3} + \frac{d 
        T^{4}}{4} + \frac{e T^{5}}{5} + \frac{f T^{6}}{6} + \frac{g T^{7}}{7}
        
    .. math::
        \int \frac{Y}{T} dT = a \log{\left (T \right )} + b T + \frac{c T^{2}}
        {2} + \frac{d T^{3}}{3} + \frac{e T^{4}}{4} + \frac{f T^{5}}{5} 
        + \frac{g T^{6}}{6}

    Examples
    --------
    Water liquid heat capacity; DIPPR coefficients normally listed in J/kmol/K:

    >>> EQ100( 276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    >>> EQ100(300)
    75355.81000000003
    
    """
    return a + T*(b + T*(c + T*(d + T*(e + T*(f + g*T)))))
 
@functor   
def DIPPR_EQ100_derivative_by_T(T, a=0., b=0., c=0., d=0., e=0., f=0., g=0.):
    """T-Derivative of DPPR Equation #100."""
    return b + T*(2*c + T*(3*d + T*(4*e + T*(5*f + 6*g*T))))
    
@functor
def DIPPR_EQ100_integral_by_T(T, a=0., b=0., c=0., d=0., e=0., f=0., g=0.):
    """T-Integral of DPPR Equation #100."""
    return T*(a + T*(b/2 + T*(c/3 + T*(d/4 + T*(e/5 + T*(f/6 + g*T/7))))))

@functor
def DIPPR_EQ100_integral_by_T_over_T(T, a=0., b=0., c=0., d=0., e=0., f=0., g=0.):
    """T-Integral of DPPR Equation #100 divided by T."""
    return a*log(T) + T*(b + T*(c/2 + T*(d/3 + T*(e/4 + T*(f/5 + g*T/6)))))

@functor
def DIPPR_EQ101(T, a, b, c, d, e):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ101 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.
    
    Notes
    -----
    DIPPR Equation # 101. Used in calculating vapor pressure, sublimation
    pressure, and liquid viscosity. All 5 parameters are required.
    e is often an integer. As the model is exponential, a sufficiently high
    temperature will cause an OverflowError. A negative temperature 
    (or just low, if fit poorly) may cause a math domain error.
    
    .. math::
        Y = \exp\left(a + \frac{b}{T} + c\cdot \ln T + d \cdot T^e\right)
    
    This function is not integrable for either dT or Y/T dT.
    
    Examples
    --------
    Water vapor pressure; DIPPR coefficients normally listed in Pa:
    
    >>> f = EQ101(73.649, -7258.2, -7.3037, 4.1653E-6, 2)
    >>> f(300)
    3537.44834545549
    
    """
    return exp(a + b/T + c*log(T) + d*T**e)

@functor
def DIPPR_EQ102(T, a, b, c, d):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ102 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.
    
    See also
    --------
    DIPPR_EQ102_derivative_by_T
    DIPPR_EQ102_integral_by_T
    DIPPR_EQ102_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation # 102. Used in calculating vapor viscosity, vapor
    thermal conductivity, and sometimes solid heat capacity. High values of b
    raise an OverflowError. All 4 parameters are required. c and d are often 0.
    
    .. math::
        Y = \frac{a\cdot T^b}{1 + \frac{c}{T} + \frac{d}{T^2}}
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. The first derivative is
    easily computed; the two integrals required Rubi to perform the integration.
    
    .. math::
        \frac{d Y}{dT} = \frac{a b T^{b}}{T \left(\frac{c}{T} + \frac{d}{T^{2}} 
        + 1\right)} + \frac{a T^{b} \left(\frac{c}{T^{2}} + \frac{2 d}{T^{3}}
        \right)}{\left(\frac{c}{T} + \frac{d}{T^{2}} + 1\right)^{2}}
        
    .. math::
        \int Y dT = - \frac{2 a T^{b + 3} \operatorname{hyp2f1}{\left (1,b + 3,
        b + 4,- \frac{2 T}{c - \sqrt{c^{2} - 4 d}} \right )}}{\left(b + 3\right) 
        \left(c + \sqrt{c^{2} - 4 d}\right) \sqrt{c^{2} - 4 d}} + \frac{2 a 
        T^{b + 3} \operatorname{hyp2f1}{\left (1,b + 3,b + 4,- \frac{2 T}{c 
        + \sqrt{c^{2} - 4 d}} \right )}}{\left(b + 3\right) \left(c 
        - \sqrt{c^{2} - 4 d}\right) \sqrt{c^{2} - 4 d}}
        
    .. math::
        \int \frac{Y}{T} dT = - \frac{2 a T^{b + 2} \operatorname{hyp2f1}{\left
        (1,b + 2,b + 3,- \frac{2 T}{c + \sqrt{c^{2} - 4 d}} \right )}}{\left(b 
        + 2\right) \left(c + \sqrt{c^{2} - 4 d}\right) \sqrt{c^{2} - 4 d}}
        + \frac{2 a T^{b + 2} \operatorname{hyp2f1}{\left (1,b + 2,b + 3,
        - \frac{2 T}{c - \sqrt{c^{2} - 4 d}} \right )}}{\left(b + 2\right) 
        \left(c - \sqrt{c^{2} - 4 d}\right) \sqrt{c^{2} - 4 d}}
        
    Examples
    --------
    Water vapor viscosity; DIPPR coefficients normally listed in Pa*s:
    
    >>> f = EQ102(1.7096E-8, 1.1146, 0, 0)
    >>> f(300)
    9.860384711890639e-06
    
    """
    return a*T**b/(1. + c/T + d/(T*T))

def DIPPR_EQ102_derivative_by_T(T, a, b, c, d):
    """T-Derivative of DPPR Equation #102."""
    return (a*b*T**b/(T*(c/T + d/T**2 + 1)) 
            + a*T**b*(c/T**2 + 2*d/T**3)/(c/T + d/T**2 + 1)**2)
    
def DIPPR_EQ102_integral_by_T(T, a, b, c, d):
    """T-Integral of DPPR Equation #102."""
    # imaginary part is 0
    return (2*a*T**(3+b)*hyp2f1(1, 3+b, 4+b, -2*T/(c - csqrt(c*c 
            - 4*d)))/((3+b)*(c - csqrt(c*c-4*d))*csqrt(c*c-4*d))
            -2*a*T**(3+b)*hyp2f1(1, 3+b, 4+b, -2*T/(c + csqrt(c*c - 4*d)))/(
                    (3+b)*(c + csqrt(c*c-4*d))*csqrt(c*c-4*d))).real
    
def DIPPR_EQ102_integral_by_T_over_T(T, a, b, c, d):
    """T-Integral of DPPR Equation #102 divided by T."""
    return (2*a*T**(2+b)*hyp2f1(1, 2+b, 3+b, -2*T/(c - csqrt(c*c - 4*d)))/(
            (2+b)*(c - csqrt(c*c-4*d))*csqrt(c*c-4*d)) - 2*a*T**(2+b)*
        hyp2f1(1, 2+b, 3+b, -2*T/(c + csqrt(c*c - 4*d)))/((2+b)*(c + csqrt(
                            c*c-4*d))*csqrt(c*c-4*d))).real
        
@functor
def DIPPR_EQ104(T, a, b, c, d, e):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ104 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.
    
    See also
    --------
    DIPPR_EQ104_derivative_by_T
    DIPPR_EQ104_integral_by_T
    DIPPR_EQ104_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation #104. Often used in calculating second virial
    coefficients of gases.
    
    .. math::
        Y = a + \frac{b}{T} + \frac{c}{T^3} + \frac{d}{T^8} + \frac{e}{T^9}
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. All expressions can be
    obtained with SymPy readily.
    
    .. math::
        \frac{d Y}{dT} = - \frac{b}{T^{2}} - \frac{3 c}{T^{4}} 
        - \frac{8 d}{T^{9}} - \frac{9 e}{T^{10}}
        
    .. math::
        \int Y dT = a T + b \log{\left (T \right )} - \frac{1}{56 T^{8}} 
        \left(28 c T^{6} + 8 d T + 7 e\right)
        
    .. math::
        \int \frac{Y}{T} dT = a \log{\left (T \right )} - \frac{1}{72 T^{9}} 
        \left(72 b T^{8} + 24 c T^{6} + 9 d T + 8 e\right)
    
    Examples
    --------
    Water second virial coefficient; DIPPR coefficients normally dimensionless:
    
    >>> f = EQ104(0.02222, -26.38, -16750000, -3.894E19, 3.133E21)
    >>> f(300)
    -1.1204179007265156
    
    """
    T2 = T*T
    return a + (b + (c + (d + e/T)/(T2*T2*T))/T2)/T

@functor
def DIPPR_EQ104_derivative_by_T(T, a, b, c, d, e):
    """T-Derivative of DPPR Equation #104."""
    T2 = T*T
    T4 = T2*T2
    return (-b + (-3*c + (-8*d - 9*e/T)/(T4*T))/T2)/T2

@functor
def DIPPR_EQ104_integral_by_T(T, a, b, c, d, e):
    """T-Integral of DPPR Equation #104."""
    return a*T + b*log(T) - (28*c*T**6 + 8*d*T + 7*e)/(56*T**8)

@functor
def DIPPR_EQ104_integral_by_T_over_T(T, a, b, c, d, e):
    """T-Integral of DPPR Equation #104 divided by T."""
    return a*log(T) - (72*b*T**8 + 24*c*T**6 + 9*d*T + 8*e)/(72*T**9)
    

@functor
def DIPPR_EQ105(T, a, b, c, d, vol=False):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ105 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,vol : float
        Regressed coefficients.
    
    Notes
    -----
    DIPPR Equation #105. Often used in calculating liquid molar density.
    c is sometimes the fluid's critical temperature.
    
    .. math::
        Y = \frac{a}{b^{1 + (1-\frac{T}{c})^d}}
    
    This expression can be integrated in terms of the incomplete gamma function
    for dT, but for Y/T dT no integral could be found.
    
    Examples
    --------
    Hexane molar density; DIPPR coefficients normally in kmol/m^3:
        
    >>> f = EQ105(0.70824, 0.26411, 507.6, 0.27537)
    >>> f(300.)
    7.593170096339236
    
    """
    if vol:
        return b**(1. + (1. - T/c)**d)/a
    else:
        return a/b**(1. + (1. - T/c)**d)

@functor
def DIPPR_EQ106(T, Tc, a, b, c=0, d=0, e=0):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ106 method,
    as described in [1]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    a,b,c,d,e : float
        Regressed coefficients.
    
    Notes
    -----
    DIPPR Equation #106. Often used in calculating liquid surface tension,
    and heat of vaporization. Only parameters a and b parameters are required; 
    many fits include no further parameters. Critical temperature is also
    required.
    
    .. math::
        Y = a(1-T_r)^{b + c T_r + d T_r^2 + e T_r^3}
        Tr = \frac{T}{Tc}
    
    The integral could not be found, but the integral over T actually could,
    again in terms of hypergeometric functions.
    
    Examples
    --------
    Water surface tension; DIPPR coefficients normally in Pa*s:
    
    >>> f = EQ106(647.096, 0.17766, 2.567, -3.3377, 1.9699)
    >>> f(300)
    0.07231499373541
    
    """
    Tr = T/Tc
    return a*(1. - Tr)**(b + Tr*(c + Tr*(d + e*Tr)))

@functor
def DIPPR_EQ107(T, a=0, b=0, c=0, d=0, e=0):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ107 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.
    
    See also
    --------
    DIPPR_EQ107_derivative_by_T
    DIPPR_EQ107_integral_by_T
    DIPPR_EQ107_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation #107. Often used in calculating ideal-gas heat capacity.
    Also called the Aly-Lee equation [2]_.
    
    .. math::
        Y = a + b\left[\frac{c/T}{\sinh(c/T)}\right]^2 + d\left[\frac{e/T}{
        \cosh(e/T)}\right]^2
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. The derivative is 
    obtained via SymPy; the integrals from Wolfram Alpha.
    
    .. math::
        \frac{d Y}{dT} = \frac{2 b c^{3} \cosh{\left (\frac{c}{T} \right )}}
        {T^{4} \sinh^{3}{\left (\frac{c}{T} \right )}} - \frac{2 b c^{2}}{T^{3}
        \sinh^{2}{\left (\frac{c}{T} \right )}} + \frac{2 d e^{3} \sinh{\left
        (\frac{e}{T} \right )}}{T^{4} \cosh^{3}{\left (\frac{e}{T} \right )}} 
        - \frac{2 d e^{2}}{T^{3} \cosh^{2}{\left (\frac{e}{T} \right )}}
        
    .. math::
        \int Y dT = a T + \frac{b c}{\tanh{\left (\frac{c}{T} \right )}}
        - d e \tanh{\left (\frac{e}{T} \right )}
        
    .. math::
        \int \frac{Y}{T} dT = a \log{\left (T \right )} + \frac{b c}{T \tanh{
        \left (\frac{c}{T} \right )}} - b \log{\left (\sinh{\left (\frac{c}{T}
        \right )} \right )} - \frac{d e}{T} \tanh{\left (\frac{e}{T} \right )}
        + d \log{\left (\cosh{\left (\frac{e}{T} \right )} \right )}
        
    Examples
    --------
    Water ideal gas molar heat capacity; DIPPR coefficients normally in
    J/kmol/K:
    
    >>> f = EQ107(33363., 26790., 2610.5, 8896., 1169.)
    >>> f(300.)
    33585.90452768923
    
    """
    return a + b*((c/T)/sinh(c/T))**2 + d*((e/T)/cosh(e/T))**2

@functor
def DIPPR_EQ107_derivative_by_T(T, a=0, b=0, c=0, d=0, e=0):
    """T-Derivative of DPPR Equation #107."""
    return (2*b*c**3*cosh(c/T)/(T**4*sinh(c/T)**3) 
            - 2*b*c**2/(T**3*sinh(c/T)**2) 
            + 2*d*e**3*sinh(e/T)/(T**4*cosh(e/T)**3)
            - 2*d*e**2/(T**3*cosh(e/T)**2))
@functor
def DIPPR_EQ107_integral_by_T(T, a=0, b=0, c=0, d=0, e=0):
    """T-Integral of DPPR Equation #107."""
    return a*T + b*c/tanh(c/T) - d*e*tanh(e/T)

@functor
def DIPPR_EQ107_integral_by_T_over_T(T, a=0, b=0, c=0, d=0, e=0):
    """T-Integral of DPPR Equation #107 divided by T."""
    return (a*log(T) + b*c/tanh(c/T)/T - b*log(sinh(c/T)) 
                - d*e*tanh(e/T)/T + d*log(cosh(e/T)))

@functor
def DIPPR_EQ114(T, Tc, a, b, c, d):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ114 method, 
    as described in [1]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    a,b,c,d : float
        Regressed coefficients.
    
    
    See also
    --------
    DIPPR_EQ114_derivative_by_T
    DIPPR_EQ114_integral_by_T
    DIPPR_EQ114_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation #114. Rarely used, normally as an alternate liquid
    heat capacity expression. All 4 parameters are required, as well as
    critical temperature.
    
    .. math::
        Y = \frac{a^2}{\tau} + b - 2AC\tau - AD\tau^2 - \frac{1}{3}c^2\tau^3
        - \frac{1}{2}CD\tau^4 - \frac{1}{5}d^2\tau^5
        
        \tau = 1 - \frac{T}{Tc}
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. All expressions can be
    obtained with SymPy readily.
    
    .. math::
        \frac{d Y}{dT} = \frac{a^{2}}{T_{c} \left(- \frac{T}{T_{c}} 
        + 1\right)^{2}} + \frac{2 a}{T_{c}} c + \frac{2 a}{T_{c}} d \left(
        - \frac{T}{T_{c}} + 1\right) + \frac{c^{2}}{T_{c}} \left(
        - \frac{T}{T_{c}} + 1\right)^{2} + \frac{2 c}{T_{c}} d \left(
        - \frac{T}{T_{c}} + 1\right)^{3} + \frac{d^{2}}{T_{c}} \left(
        - \frac{T}{T_{c}} + 1\right)^{4}
        
    .. math::
        \int Y dT = - a^{2} T_{c} \log{\left (T - T_{c} \right )} + \frac{d^{2}
        T^{6}}{30 T_{c}^{5}} - \frac{T^{5}}{10 T_{c}^{4}} \left(c d + 2 d^{2}
        \right) + \frac{T^{4}}{12 T_{c}^{3}} \left(c^{2} + 6 c d + 6 d^{2}
        \right) - \frac{T^{3}}{3 T_{c}^{2}} \left(a d + c^{2} + 3 c d 
        + 2 d^{2}\right) + \frac{T^{2}}{2 T_{c}} \left(2 a c + 2 a d + c^{2} 
        + 2 c d + d^{2}\right) + T \left(- 2 a c - a d + b - \frac{c^{2}}{3} 
        - \frac{c d}{2} - \frac{d^{2}}{5}\right)
        
    .. math::
        \int \frac{Y}{T} dT = - a^{2} \log{\left (T + \frac{- 60 a^{2} T_{c}
        + 60 a c T_{c} + 30 a d T_{c} - 30 b T_{c} + 10 c^{2} T_{c}
        + 15 c d T_{c} + 6 d^{2} T_{c}}{60 a^{2} - 60 a c - 30 a d + 30 b 
        - 10 c^{2} - 15 c d - 6 d^{2}} \right )} + \frac{d^{2} T^{5}}
        {25 T_{c}^{5}} - \frac{T^{4}}{8 T_{c}^{4}} \left(c d + 2 d^{2}
        \right) + \frac{T^{3}}{9 T_{c}^{3}} \left(c^{2} + 6 c d + 6 d^{2}
        \right) - \frac{T^{2}}{2 T_{c}^{2}} \left(a d + c^{2} + 3 c d
        + 2 d^{2}\right) + \frac{T}{T_{c}} \left(2 a c + 2 a d + c^{2} 
        + 2 c d + d^{2}\right) + \frac{1}{30} \left(30 a^{2} - 60 a c 
        - 30 a d + 30 b - 10 c^{2} - 15 c d - 6 d^{2}\right) \log{\left 
        (T + \frac{1}{60 a^{2} - 60 a c - 30 a d + 30 b - 10 c^{2} - 15 c d
        - 6 d^{2}} \left(- 30 a^{2} T_{c} + 60 a c T_{c} + 30 a d T_{c} 
        - 30 b T_{c} + 10 c^{2} T_{c} + 15 c d T_{c} + 6 d^{2} T_{c}
        + T_{c} \left(30 a^{2} - 60 a c - 30 a d + 30 b - 10 c^{2} - 15 c d
        - 6 d^{2}\right)\right) \right )}
    
    Strictly speaking, the integral over T has an imaginary component, but
    only the real component is relevant and the complex part discarded.
    
    Examples
    --------
    Hydrogen liquid heat capacity; DIPPR coefficients normally in J/kmol/K:
    
    >>> f = EQ114(33.19, 66.653, 6765.9, -123.63, 478.27)
    >>> f(20)
    19423.948911676463
    
    """
    t = 1.-T/Tc
    return (a**2./t + b - 2.*a*c*t - a*d*t**2. - c**2.*t**3./3. 
            - c*d*t**4./2. - d**2*t**5./5.)

@functor
def DIPPR_EQ114_derivative_by_T(T, Tc, a, b, c, d):
    """T-Derivative of DPPR Equation #114."""
    return (a**2/(Tc*(-T/Tc + 1)**2) + 2*a*c/Tc + 2*a*d*(-T/Tc + 1)/Tc 
            + c**2*(-T/Tc + 1)**2/Tc + 2*c*d*(-T/Tc + 1)**3/Tc 
            + d**2*(-T/Tc + 1)**4/Tc)

@functor
def DIPPR_EQ114_integral_by_T(T, Tc, a, b, c, d):
    """T-Integral of DPPR Equation #114."""
    return (-a**2*Tc*clog(T - Tc).real + d**2*T**6/(30*Tc**5) 
                - T**5*(c*d + 2*d**2)/(10*Tc**4) 
                + T**4*(c**2 + 6*c*d + 6*d**2)/(12*Tc**3) - T**3*(a*d + c**2 
                + 3*c*d + 2*d**2)/(3*Tc**2) + T**2*(2*a*c + 2*a*d + c**2 + 2*c*d 
                + d**2)/(2*Tc) + T*(-2*a*c - a*d + b - c**2/3 - c*d/2 - d**2/5))

@functor
def DIPPR_EQ114_integral_by_T_over_T(T, Tc, a, b, c, d):
    """T-Integral of DPPR Equation #114 divided by T."""
    return (-a**2*clog(T + (-60*a**2*Tc + 60*a*c*Tc + 30*a*d*Tc - 30*b*Tc 
                + 10*c**2*Tc + 15*c*d*Tc + 6*d**2*Tc)/(60*a**2 - 60*a*c 
                - 30*a*d + 30*b - 10*c**2 - 15*c*d - 6*d**2)).real 
                + d**2*T**5/(25*Tc**5) - T**4*(c*d + 2*d**2)/(8*Tc**4) 
                + T**3*(c**2 + 6*c*d + 6*d**2)/(9*Tc**3) - T**2*(a*d + c**2
                + 3*c*d + 2*d**2)/(2*Tc**2) + T*(2*a*c + 2*a*d + c**2 + 2*c*d
                + d**2)/Tc + (30*a**2 - 60*a*c - 30*a*d + 30*b - 10*c**2
                - 15*c*d - 6*d**2)*clog(T + (-30*a**2*Tc + 60*a*c*Tc 
                + 30*a*d*Tc - 30*b*Tc + 10*c**2*Tc + 15*c*d*Tc + 6*d**2*Tc 
                + Tc*(30*a**2 - 60*a*c - 30*a*d + 30*b - 10*c**2 - 15*c*d 
               - 6*d**2))/(60*a**2 - 60*a*c - 30*a*d + 30*b - 10*c**2 
                    - 15*c*d - 6*d**2)).real/30)

@functor
def DIPPR_EQ115(T, a, b, c=0, d=0, e=0):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ115 method, 
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e : float
        Regressed coefficients.
    
    Notes
    -----
    DIPPR Equation #115. No major uses; has been used as an alternate
    liquid viscosity expression, and as a model for vapor pressure.
    Only parameters a and b are required.
    
    .. math::
        Y = \exp\left(a + \frac{b}{T} + c\log T + d T^2 + \frac{e}{T^2}\right)
    
    No coefficients found for this expression.
    This function is not integrable for either dT or Y/T dT.
    
    """
    T2 = T**2
    return exp(a+b/T+c*log(T)+d*T2 + e/T2)

@functor
def DIPPR_EQ116(T, Tc, a, b, c, d, e):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ116 method,
    as described in [1]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    a,b,c,d,e : float
        Regressed coefficients.
    
    
    See also
    --------
    DIPPR_EQ116_derivative_by_T
    DIPPR_EQ116_integral_by_T
    DIPPR_EQ116_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation #116. Used to describe the molar density of water fairly
    precisely; no other uses listed. All 5 parameters are needed, as well as
    the critical temperature.
    
    .. math::
        Y = a + b\tau^{0.35} + c\tau^{2/3} + d\tau + e\tau^{4/3}
        
        \tau = 1 - \frac{T}{T_c}
    
    The derivative with respect to T and integral with respect to T are 
    computed as follows. The integral divided by T with respect to T has an
    extremely complicated (but still elementary) integral which can be read 
    from the source. It was computed with Rubi; the other expressions can 
    readily be obtained with SymPy.
    
    .. math::
        \frac{d Y}{dT} = - \frac{7 b}{20 T_c \left(- \frac{T}{T_c} + 1\right)^{
        \frac{13}{20}}} - \frac{2 c}{3 T_c \sqrt[3]{- \frac{T}{T_c} + 1}} 
        - \frac{d}{T_c} - \frac{4 e}{3 T_c} \sqrt[3]{- \frac{T}{T_c} + 1}
    
    .. math::
        \int Y dT = a T - \frac{20 b}{27} T_c \left(- \frac{T}{T_c} + 1\right)^{
        \frac{27}{20}} - \frac{3 c}{5} T_c \left(- \frac{T}{T_c} + 1\right)^{
        \frac{5}{3}} + d \left(- \frac{T^{2}}{2 T_c} + T\right) - \frac{3 e}{7} 
        T_c \left(- \frac{T}{T_c} + 1\right)^{\frac{7}{3}}
                
    Examples
    --------
    Water liquid molar density; DIPPR coefficients normally in kmol/m^3:
    
    >>> f = EQ116(647.096, 17.863, 58.606, -95.396, 213.89, -141.26)
    >>> f(300.)
    55.17615446406527
    
    """
    tau = 1-T/Tc
    return a + b*tau**0.35 + c*tau**(2/3.) + d*tau + e*tau**(4/3.)

@functor
def DIPPR_EQ116_derivative_by_T(T, Tc, a, b, c, d, e):
    """T-Derivative of DPPR Equation #116."""
    return (-7*b/(20*Tc*(-T/Tc + 1)**(13/20)) 
                - 2*c/(3*Tc*(-T/Tc + 1)**(1/3)) 
                - d/Tc - 4*e*(-T/Tc + 1)**(1/3)/(3*Tc))

@functor
def DIPPR_EQ116_integral_by_T(T, Tc, a, b, c, d, e):
    """T-Integral of DPPR Equation #116."""
    return (a*T - 20*b*Tc*(-T/Tc + 1)**(27/20)/27 
                - 3*c*Tc*(-T/Tc + 1)**(5/3)/5 + d*(-T**2/(2*Tc) + T)
                - 3*e*Tc*(-T/Tc + 1)**(7/3)/7)

@functor
def DIPPR_EQ116_integral_by_T_over_T(T, Tc, a, b, c, d, e):
    """T-Integral of DPPR Equation #116 divided by T."""
    # 3x increase in speed - cse via sympy
    x0 = log(T)
    x1 = 0.5*x0
    x2 = 1/Tc
    x3 = T*x2
    x4 = -x3 + 1
    x5 = 1.5*c
    x6 = x4**0.333333333333333
    x7 = 2*b
    x8 = x4**0.05
    x9 = log(-x6 + 1)
    x10 = sqrt(3)
    x11 = x10*atan(x10*(2*x6 + 1)/3)
    x12 = sqrt(5)
    x13 = 0.5*x12
    x14 = x13 + 0.5
    x15 = b*x14
    x16 = sqrt(x13 + 2.5)
    x17 = 2*x8
    x18 = -x17
    x19 = -x13
    x20 = x19 + 0.5
    x21 = b*x20
    x22 = sqrt(x19 + 2.5)
    x23 = b*x16
    x24 = 0.5*sqrt(0.1*x12 + 0.5)
    x25 = x12 + 1
    x26 = 4*x8
    x27 = -x26
    x28 = sqrt(10)*b/sqrt(x12 + 5)
    x29 = 2*x12
    x30 = sqrt(x29 + 10)
    x31 = 1/x30
    x32 = -x12 + 1
    x33 = 0.5*b*x22
    x34 = -x2*(T - Tc)
    x35 = 2*x34**0.1
    x36 = x35 + 2
    x37 = x34**0.05
    x38 = x30*x37
    x39 = 0.5*b*x16
    x40 = x37*sqrt(-x29 + 10)
    x41 = 0.25*x12
    x42 = b*(-x41 + 0.25)
    x43 = x12*x37
    x44 = x35 + x37 + 2
    x45 = b*(x41 + 0.25)
    x46 = -x43
    x47 = x35 - x37 + 2
    return a*x0 + 2.85714285714286*b*x4**0.35 - c*x1 + c*x11 + d*x0 - d*x3 - e*x1 - e*x11 + 0.75*e*x4**1.33333333333333 + 3*e*x6 + 1.5*e*x9 - x15*atan(x14*(x16 + x17)) + x15*atan(x14*(x16 + x18)) - x21*atan(x20*(x17 + x22)) + x21*atan(x20*(x18 + x22)) + x23*atan(x24*(x25 + x26)) - x23*atan(x24*(x25 + x27)) - x28*atan(x31*(x26 + x32)) + x28*atan(x31*(x27 + x32)) - x33*log(x36 - x38) + x33*log(x36 + x38) + x39*log(x36 - x40) - x39*log(x36 + x40) + x4**0.666666666666667*x5 - x42*log(x43 + x44) + x42*log(x46 + x47) + x45*log(x43 + x47) - x45*log(x44 + x46) + x5*x9 + x7*atan(x8) - x7*atanh(x8)

@functor
def DIPPR_EQ127(T, a, b, c, d, e, f, g):
    r"""
    Create a functor of temperature (T; in K) based on the DIPPR EQ127 method,
    as described in [1]_.
    
    Parameters
    ----------
    a,b,c,d,e,f,g : float
        Regressed coefficients.
    
    See also
    --------
    DIPPR_EQ127_derivative_by_T
    DIPPR_EQ127_integral_by_T
    DIPPR_EQ127_integral_by_T_over_T
    
    Notes
    -----
    DIPPR Equation #127. Rarely used, and then only in calculating
    ideal-gas heat capacity. All 7 parameters are required.
    
    .. math::
        Y = a+b\left[\frac{\left(\frac{c}{T}\right)^2\exp\left(\frac{c}{T}
        \right)}{\left(\exp\frac{c}{T}-1 \right)^2}\right]
        +d\left[\frac{\left(\frac{e}{T}\right)^2\exp\left(\frac{e}{T}\right)}
        {\left(\exp\frac{e}{T}-1 \right)^2}\right]
        +f\left[\frac{\left(\frac{g}{T}\right)^2\exp\left(\frac{g}{T}\right)}
        {\left(\exp\frac{g}{T}-1 \right)^2}\right]
    
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. All expressions can be
    obtained with SymPy readily.
    
    .. math::
        \frac{d Y}{dT} = - \frac{b c^{3} e^{\frac{c}{T}}}{T^{4}
        \left(e^{\frac{c}{T}} - 1\right)^{2}} + \frac{2 b c^{3} 
        e^{\frac{2 c}{T}}}{T^{4} \left(e^{\frac{c}{T}} - 1\right)^{3}} 
        - \frac{2 b c^{2} e^{\frac{c}{T}}}{T^{3} \left(e^{\frac{c}{T}} 
        - 1\right)^{2}} - \frac{d e^{3} e^{\frac{e}{T}}}{T^{4} 
        \left(e^{\frac{e}{T}} - 1\right)^{2}} + \frac{2 d e^{3} 
        e^{\frac{2 e}{T}}}{T^{4} \left(e^{\frac{e}{T}} - 1\right)^{3}}
        - \frac{2 d e^{2} e^{\frac{e}{T}}}{T^{3} \left(e^{\frac{e}{T}} 
        - 1\right)^{2}} - \frac{f g^{3} e^{\frac{g}{T}}}{T^{4}
        \left(e^{\frac{g}{T}} - 1\right)^{2}} + \frac{2 f g^{3}
        e^{\frac{2 g}{T}}}{T^{4} \left(e^{\frac{g}{T}} - 1\right)^{3}}
        - \frac{2 f g^{2} e^{\frac{g}{T}}}{T^{3} \left(e^{\frac{g}{T}} 
        - 1\right)^{2}}
        
    .. math::
        \int Y dT = a T + \frac{b c^{2}}{c e^{\frac{c}{T}} - c} 
        + \frac{d e^{2}}{e e^{\frac{e}{T}} - e} 
        + \frac{f g^{2}}{g e^{\frac{g}{T}} - g}
        
    .. math::
        \int \frac{Y}{T} dT = a \log{\left (T \right )} + b c^{2} \left(
        \frac{1}{c T e^{\frac{c}{T}} - c T} + \frac{1}{c T} - \frac{1}{c^{2}} 
        \log{\left (e^{\frac{c}{T}} - 1 \right )}\right) + d e^{2} \left(
        \frac{1}{e T e^{\frac{e}{T}} - e T} + \frac{1}{e T} - \frac{1}{e^{2}} 
        \log{\left (e^{\frac{e}{T}} - 1 \right )}\right) + f g^{2} \left(
        \frac{1}{g T e^{\frac{g}{T}} - g T} + \frac{1}{g T} - \frac{1}{g^{2}} 
        \log{\left (e^{\frac{g}{T}} - 1 \right )}\right)
            
    Examples
    --------
    Ideal gas heat capacity of methanol; DIPPR coefficients normally in
    J/kmol/K:
    
    >>> f = EQ127(3.3258E4, 3.6199E4, 1.2057E3, 1.5373E7, 3.2122E3, -1.5318E7, 3.2122E3)
    >>> f(20.)
    33258.0
    
    """
    return (a+b*((c/T)**2*exp(c/T)/(exp(c/T) - 1)**2) + 
        d*((e/T)**2*exp(e/T)/(exp(e/T)-1)**2) + 
        f*((g/T)**2*exp(g/T)/(exp(g/T)-1)**2))

@functor
def DIPPR_EQ127_derivative_by_T(T, a, b, c, d, e, f, g):
    """T-Derivative of DPPR Equation #127."""
    return (-b*c**3*exp(c/T)/(T**4*(exp(c/T) - 1)**2) 
                + 2*b*c**3*exp(2*c/T)/(T**4*(exp(c/T) - 1)**3) 
                - 2*b*c**2*exp(c/T)/(T**3*(exp(c/T) - 1)**2) 
                - d*e**3*exp(e/T)/(T**4*(exp(e/T) - 1)**2) 
                + 2*d*e**3*exp(2*e/T)/(T**4*(exp(e/T) - 1)**3) 
                - 2*d*e**2*exp(e/T)/(T**3*(exp(e/T) - 1)**2) 
                - f*g**3*exp(g/T)/(T**4*(exp(g/T) - 1)**2)
                + 2*f*g**3*exp(2*g/T)/(T**4*(exp(g/T) - 1)**3) 
                - 2*f*g**2*exp(g/T)/(T**3*(exp(g/T) - 1)**2))

@functor
def DIPPR_EQ127_integral_by_T(T, a, b, c, d, e, f, g):
    """T-Integral of DPPR Equation #127."""
    return (a*T + b*c**2/(c*exp(c/T) - c) + d*e**2/(e*exp(e/T) - e)
                + f*g**2/(g*exp(g/T) - g))

@functor
def DIPPR_EQ127_integral_by_T_over_T(T, a, b, c, d, e, f, g):
    """T-Integral of DPPR Equation #127 divided by T."""
    return (a*log(T) + b*c**2*(1/(c*T*exp(c/T) - c*T) + 1/(c*T)
                - log(exp(c/T) - 1)/c**2) + d*e**2*(1/(e*T*exp(e/T) - e*T) 
                + 1/(e*T) - log(exp(e/T) - 1)/e**2)
                + f*g**2*(1/(g*T*exp(g/T) - g*T) + 1/(g*T) - log(exp(g/T) 
                - 1)/g**2))
