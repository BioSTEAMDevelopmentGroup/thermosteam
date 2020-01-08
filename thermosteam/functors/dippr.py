# -*- coding: utf-8 -*-

__all__ = ['DIPPR_EQ100', 'DIPPR_EQ101', 'DIPPR_EQ102',
           'DIPPR_EQ104', 'DIPPR_EQ105', 'DIPPR_EQ106',
           'DIPPR_EQ107', 'DIPPR_EQ114', 'DIPPR_EQ115', 
           'DIPPR_EQ116', 'DIPPR_EQ127']

from ..base.functor import functor
from ..base.thermo_model import TDependentModel
from math import log, exp, sinh, cosh, atan, atanh, sqrt, tanh
from cmath import log as clog
from cmath import sqrt as csqrt
from scipy.special import hyp2f1

order_not_found_msg = ('Only the actual property calculation, first temperature '
                       'derivative, first temperature integral, and first '
                       'temperature integral over temperature are supported '
                       'with order=  0, 1, -1, or -1j respectively')

@functor(math=r"Y = A + BT + CT^2 + DT^3 + ET^4 + FT^5 + GT^6")
def DIPPR_EQ100(T, A=0., B=0., C=0., D=0., E=0., F=0., G=0., order=0.):
    r'''Used in calculating the molar heat capacities
    of liquids and solids, liquid thermal conductivity, and solid density.
    All parameters default to zero. As this is a straightforward polynomial,
    no restrictions on parameters apply. Note that high-order polynomials like
    this may need large numbers of decimal places to avoid unnecessary error.

    {Math}

    Parameters
    ----------
    T : float
        Temperature, [K]
    A-G : float
        Parameter for the equation; chemical and property specific [-]
    order : int, optional
        Order of the calculation. 0 for the calculation of the result itself;
        for 1, the first derivative of the property is returned, for
        -1, the indefinite integral of the property with respect to temperature
        is returned; and for -1j, the indefinite integral of the property
        divided by temperature with respect to temperature is returned. No 
        other integrals or derivatives are implemented, and an exception will 
        be raised if any other order is given.

    Returns
    -------
    Y : float
        Property [constant-specific; if order == 1, property/K; if order == -1,
                  property*K; if order == -1j, unchanged from default]

    Notes
    -----
    The derivative with respect to T, integral with respect to T, and integral
    over T with respect to T are computed as follows. All derivatives and 
    integrals are easily computed with SymPy.
    
    .. math::
        \frac{d Y}{dT} = B + 2 C T + 3 D T^{2} + 4 E T^{3} + 5 F T^{4} 
        + 6 G T^{5}
        
    .. math::
        \int Y dT = A T + \frac{B T^{2}}{2} + \frac{C T^{3}}{3} + \frac{D 
        T^{4}}{4} + \frac{E T^{5}}{5} + \frac{F T^{6}}{6} + \frac{G T^{7}}{7}
        
    .. math::
        \int \frac{Y}{T} dT = A \log{\left (T \right )} + B T + \frac{C T^{2}}
        {2} + \frac{D T^{3}}{3} + \frac{E T^{4}}{4} + \frac{F T^{5}}{5} 
        + \frac{G T^{6}}{6}

    Examples
    --------
    Water liquid heat capacity; DIPPR coefficients normally listed in J/kmol/K.

    >>> EQ100(300, 276370., -2090.1, 8.125, -0.014116, 0.0000093701)
    75355.81000000003

    References
    ----------
    .. [1] Design Institute for Physical Properties, 1996. DIPPR Project 801
       DIPPR/AIChE
    '''
    if order == 0:
        return A + T*(B + T*(C + T*(D + T*(E + T*(F + G*T)))))
    elif order == 1:
        return B + T*(2*C + T*(3*D + T*(4*E + T*(5*F + 6*G*T))))
    elif order == -1:
        return T*(A + T*(B/2 + T*(C/3 + T*(D/4 + T*(E/5 + T*(F/6 + G*T/7))))))
    elif order == -1j:
        return A*log(T) + T*(B + T*(C/2 + T*(D/3 + T*(E/4 + T*(F/5 + G*T/6)))))
    else:
        raise Exception(order_not_found_msg)

@functor
def DIPPR_EQ101(T, A, B, C, D, E):
    return exp(A + B/T + C*log(T) + D*T**E)

@functor
def DIPPR_EQ102(T, A, B, C, D):
    return A*T**B/(1. + C/T + D/(T*T))

@functor
def DIPPR_EQ102_derivative_by_T(T, A, B, C, D):
    return (A*B*T**B/(T*(C/T + D/T**2 + 1)) 
            + A*T**B*(C/T**2 + 2*D/T**3)/(C/T + D/T**2 + 1)**2)
    
@functor
def DIPPR_EQ102_integral_by_T(T, A, B, C, D):
    # imaginary part is 0
    return (2*A*T**(3+B)*hyp2f1(1, 3+B, 4+B, -2*T/(C - csqrt(C*C 
            - 4*D)))/((3+B)*(C - csqrt(C*C-4*D))*csqrt(C*C-4*D))
            -2*A*T**(3+B)*hyp2f1(1, 3+B, 4+B, -2*T/(C + csqrt(C*C - 4*D)))/(
                    (3+B)*(C + csqrt(C*C-4*D))*csqrt(C*C-4*D))).real
    
@functor
def DIPPR_EQ102_integral_by_T_over_T(T, A, B, C, D):
        return (2*A*T**(2+B)*hyp2f1(1, 2+B, 3+B, -2*T/(C - csqrt(C*C - 4*D)))/(
                (2+B)*(C - csqrt(C*C-4*D))*csqrt(C*C-4*D)) - 2*A*T**(2+B)*
            hyp2f1(1, 2+B, 3+B, -2*T/(C + csqrt(C*C - 4*D)))/((2+B)*(C + csqrt(
                                C*C-4*D))*csqrt(C*C-4*D))).real
        
@functor
def DIPPR_EQ104(T, A, B, C, D, E, order=0):
    if order == 0:
        T2 = T*T
        return A + (B + (C + (D + E/T)/(T2*T2*T))/T2)/T
    elif order == 1:
        T2 = T*T
        T4 = T2*T2
        return (-B + (-3*C + (-8*D - 9*E/T)/(T4*T))/T2)/T2
    elif order == -1:
        return A*T + B*log(T) - (28*C*T**6 + 8*D*T + 7*E)/(56*T**8)
    elif order == -1j:
        return A*log(T) - (72*B*T**8 + 24*C*T**6 + 9*D*T + 8*E)/(72*T**9)
    else:
        raise Exception(order_not_found_msg)

@functor
def DIPPR_EQ105(T, A, B, C, D, vol=False):
    if vol:
        return B**(1. + (1. - T/C)**D)/A
    else:
        return A/B**(1. + (1. - T/C)**D)

@functor
def DIPPR_EQ106(T, Tc, A, B, C=0, D=0, E=0):
    Tr = T/Tc
    return A*(1. - Tr)**(B + Tr*(C + Tr*(D + E*Tr)))

@functor
def DIPPR_EQ107(T, A=0, B=0, C=0, D=0, E=0, order=0):
    if order == 0:
        return A + B*((C/T)/sinh(C/T))**2 + D*((E/T)/cosh(E/T))**2
    elif order == 1:
        return (2*B*C**3*cosh(C/T)/(T**4*sinh(C/T)**3) 
                - 2*B*C**2/(T**3*sinh(C/T)**2) 
                + 2*D*E**3*sinh(E/T)/(T**4*cosh(E/T)**3)
                - 2*D*E**2/(T**3*cosh(E/T)**2))
    elif order == -1:
        return A*T + B*C/tanh(C/T) - D*E*tanh(E/T)
    elif order == -1j:
        return (A*log(T) + B*C/tanh(C/T)/T - B*log(sinh(C/T)) 
                - D*E*tanh(E/T)/T + D*log(cosh(E/T)))
    else:
        raise Exception(order_not_found_msg)

@functor
def DIPPR_EQ114(T, Tc, A, B, C, D, order=0):
    if order == 0:
        t = 1.-T/Tc
        return (A**2./t + B - 2.*A*C*t - A*D*t**2. - C**2.*t**3./3. 
                - C*D*t**4./2. - D**2*t**5./5.)
    elif order == 1:
        return (A**2/(Tc*(-T/Tc + 1)**2) + 2*A*C/Tc + 2*A*D*(-T/Tc + 1)/Tc 
                + C**2*(-T/Tc + 1)**2/Tc + 2*C*D*(-T/Tc + 1)**3/Tc 
                + D**2*(-T/Tc + 1)**4/Tc)
    elif order == -1:
        return (-A**2*Tc*clog(T - Tc).real + D**2*T**6/(30*Tc**5) 
                - T**5*(C*D + 2*D**2)/(10*Tc**4) 
                + T**4*(C**2 + 6*C*D + 6*D**2)/(12*Tc**3) - T**3*(A*D + C**2 
                + 3*C*D + 2*D**2)/(3*Tc**2) + T**2*(2*A*C + 2*A*D + C**2 + 2*C*D 
                + D**2)/(2*Tc) + T*(-2*A*C - A*D + B - C**2/3 - C*D/2 - D**2/5))
    elif order == -1j:
        return (-A**2*clog(T + (-60*A**2*Tc + 60*A*C*Tc + 30*A*D*Tc - 30*B*Tc 
                + 10*C**2*Tc + 15*C*D*Tc + 6*D**2*Tc)/(60*A**2 - 60*A*C 
                - 30*A*D + 30*B - 10*C**2 - 15*C*D - 6*D**2)).real 
                + D**2*T**5/(25*Tc**5) - T**4*(C*D + 2*D**2)/(8*Tc**4) 
                + T**3*(C**2 + 6*C*D + 6*D**2)/(9*Tc**3) - T**2*(A*D + C**2
                + 3*C*D + 2*D**2)/(2*Tc**2) + T*(2*A*C + 2*A*D + C**2 + 2*C*D
                + D**2)/Tc + (30*A**2 - 60*A*C - 30*A*D + 30*B - 10*C**2
                - 15*C*D - 6*D**2)*clog(T + (-30*A**2*Tc + 60*A*C*Tc 
                + 30*A*D*Tc - 30*B*Tc + 10*C**2*Tc + 15*C*D*Tc + 6*D**2*Tc 
                + Tc*(30*A**2 - 60*A*C - 30*A*D + 30*B - 10*C**2 - 15*C*D 
               - 6*D**2))/(60*A**2 - 60*A*C - 30*A*D + 30*B - 10*C**2 
                    - 15*C*D - 6*D**2)).real/30)
    else:
        raise Exception(order_not_found_msg)

@functor
def DIPPR_EQ115(T, A, B, C=0, D=0, E=0):
    T2 = T**2
    return exp(A+B/T+C*log(T)+D*T2 + E/T2)

@functor
def DIPPR_EQ116(T, Tc, A, B, C, D, E, order=0):
    if order == 0:
        tau = 1-T/Tc
        return A + B*tau**0.35 + C*tau**(2/3.) + D*tau + E*tau**(4/3.)
    elif order == 1:
        return (-7*B/(20*Tc*(-T/Tc + 1)**(13/20)) 
                - 2*C/(3*Tc*(-T/Tc + 1)**(1/3)) 
                - D/Tc - 4*E*(-T/Tc + 1)**(1/3)/(3*Tc))
    elif order == -1:
        return (A*T - 20*B*Tc*(-T/Tc + 1)**(27/20)/27 
                - 3*C*Tc*(-T/Tc + 1)**(5/3)/5 + D*(-T**2/(2*Tc) + T)
                - 3*E*Tc*(-T/Tc + 1)**(7/3)/7)
    elif order == -1j:
        # 3x increase in speed - cse via sympy
        x0 = log(T)
        x1 = 0.5*x0
        x2 = 1/Tc
        x3 = T*x2
        x4 = -x3 + 1
        x5 = 1.5*C
        x6 = x4**0.333333333333333
        x7 = 2*B
        x8 = x4**0.05
        x9 = log(-x6 + 1)
        x10 = sqrt(3)
        x11 = x10*atan(x10*(2*x6 + 1)/3)
        x12 = sqrt(5)
        x13 = 0.5*x12
        x14 = x13 + 0.5
        x15 = B*x14
        x16 = sqrt(x13 + 2.5)
        x17 = 2*x8
        x18 = -x17
        x19 = -x13
        x20 = x19 + 0.5
        x21 = B*x20
        x22 = sqrt(x19 + 2.5)
        x23 = B*x16
        x24 = 0.5*sqrt(0.1*x12 + 0.5)
        x25 = x12 + 1
        x26 = 4*x8
        x27 = -x26
        x28 = sqrt(10)*B/sqrt(x12 + 5)
        x29 = 2*x12
        x30 = sqrt(x29 + 10)
        x31 = 1/x30
        x32 = -x12 + 1
        x33 = 0.5*B*x22
        x34 = -x2*(T - Tc)
        x35 = 2*x34**0.1
        x36 = x35 + 2
        x37 = x34**0.05
        x38 = x30*x37
        x39 = 0.5*B*x16
        x40 = x37*sqrt(-x29 + 10)
        x41 = 0.25*x12
        x42 = B*(-x41 + 0.25)
        x43 = x12*x37
        x44 = x35 + x37 + 2
        x45 = B*(x41 + 0.25)
        x46 = -x43
        x47 = x35 - x37 + 2
        return A*x0 + 2.85714285714286*B*x4**0.35 - C*x1 + C*x11 + D*x0 - D*x3 - E*x1 - E*x11 + 0.75*E*x4**1.33333333333333 + 3*E*x6 + 1.5*E*x9 - x15*atan(x14*(x16 + x17)) + x15*atan(x14*(x16 + x18)) - x21*atan(x20*(x17 + x22)) + x21*atan(x20*(x18 + x22)) + x23*atan(x24*(x25 + x26)) - x23*atan(x24*(x25 + x27)) - x28*atan(x31*(x26 + x32)) + x28*atan(x31*(x27 + x32)) - x33*log(x36 - x38) + x33*log(x36 + x38) + x39*log(x36 - x40) - x39*log(x36 + x40) + x4**0.666666666666667*x5 - x42*log(x43 + x44) + x42*log(x46 + x47) + x45*log(x43 + x47) - x45*log(x44 + x46) + x5*x9 + x7*atan(x8) - x7*atanh(x8)
    else:
        raise Exception(order_not_found_msg)

@functor
def DIPPR_EQ127(T, A, B, C, D, E, F, G, order=0):
    if order == 0:
        return (A+B*((C/T)**2*exp(C/T)/(exp(C/T) - 1)**2) + 
            D*((E/T)**2*exp(E/T)/(exp(E/T)-1)**2) + 
            F*((G/T)**2*exp(G/T)/(exp(G/T)-1)**2))
    elif order == 1:
        return (-B*C**3*exp(C/T)/(T**4*(exp(C/T) - 1)**2) 
                + 2*B*C**3*exp(2*C/T)/(T**4*(exp(C/T) - 1)**3) 
                - 2*B*C**2*exp(C/T)/(T**3*(exp(C/T) - 1)**2) 
                - D*E**3*exp(E/T)/(T**4*(exp(E/T) - 1)**2) 
                + 2*D*E**3*exp(2*E/T)/(T**4*(exp(E/T) - 1)**3) 
                - 2*D*E**2*exp(E/T)/(T**3*(exp(E/T) - 1)**2) 
                - F*G**3*exp(G/T)/(T**4*(exp(G/T) - 1)**2)
                + 2*F*G**3*exp(2*G/T)/(T**4*(exp(G/T) - 1)**3) 
                - 2*F*G**2*exp(G/T)/(T**3*(exp(G/T) - 1)**2))
    elif order == -1:
        return (A*T + B*C**2/(C*exp(C/T) - C) + D*E**2/(E*exp(E/T) - E)
                + F*G**2/(G*exp(G/T) - G))
    elif order == -1j:
        return (A*log(T) + B*C**2*(1/(C*T*exp(C/T) - C*T) + 1/(C*T)
                - log(exp(C/T) - 1)/C**2) + D*E**2*(1/(E*T*exp(E/T) - E*T) 
                + 1/(E*T) - log(exp(E/T) - 1)/E**2)
                + F*G**2*(1/(G*T*exp(G/T) - G*T) + 1/(G*T) - log(exp(G/T) 
                - 1)/G**2))
    else:
        raise Exception(order_not_found_msg)
