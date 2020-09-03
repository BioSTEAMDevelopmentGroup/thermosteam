# -*- coding: utf-8 -*-
"""
All data and methods related to a chemical's surface tension.

References
----------
.. [1] Diky, Vladimir, Robert D. Chirico, Chris D. Muzny, Andrei F.
    Kazakov, Kenneth Kroenlein, Joseph W. Magee, Ilmutdin Abdulagatov, and
    Michael Frenkel. "ThermoData Engine (TDE): Software Implementation of
    the Dynamic Data Evaluation Concept." Journal of Chemical Information
    and Modeling 53, no. 12 (2013): 3418-30. doi:10.1021/ci4005699.
.. [2] Somayajulu, G. R. "A Generalized Equation for Surface Tension from
    the Triple Point to the Critical Point." International Journal of
    Thermophysics 9, no. 4 (July 1988): 559-66. doi:10.1007/BF00503154.
.. [3] Jasper, Joseph J. "The Surface Tension of Pure Liquid Compounds."
    Journal of Physical and Chemical Reference Data 1, no. 4
    (October 1, 1972): 841-1010. doi:10.1063/1.3253106.
.. [4] Speight, James. Lange's Handbook of Chemistry. 16 edition.
    McGraw-Hill Professional, 2005.
.. [5] Brock, James R., and R. Byron Bird. "Surface Tension and the
    Principle of Corresponding States." AIChE Journal 1, no. 2
    (June 1, 1955): 174-77. doi:10.1002/aic.690010208
.. [6] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
   New York: McGraw-Hill Professional, 2000.
.. [7] Curl, R. F., and Kenneth Pitzer. "Volumetric and Thermodynamic
   Properties of Fluids-Enthalpy, Free Energy, and Entropy." Industrial &
   Engineering Chemistry 50, no. 2 (February 1, 1958): 265-74.
   doi:10.1021/ie50578a047
.. [8] Pitzer, K. S.: Thermodynamics, 3d ed., New York, McGraw-Hill,
   1995, p. 521.
.. [9] Sastri, S. R. S., and K. K. Rao. "A Simple Method to Predict
    Surface Tension of Organic Liquids." The Chemical Engineering Journal
    and the Biochemical Engineering Journal 59, no. 2 (October 1995): 181-86.
    doi:10.1016/0923-0467(94)02946-6.
.. [10] Zuo, You-Xiang, and Erling H. Stenby. "Corresponding-States and
    Parachor Models for the Calculation of Interfacial Tensions." The
    Canadian Journal of Chemical Engineering 75, no. 6 (December 1, 1997):
    1130-37. doi:10.1002/cjce.5450750617
.. [11] Hakim, D. I., David Steinberg, and L. I. Stiel. "Generalized
    Relationship for the Surface Tension of Polar Fluids." Industrial &
    Engineering Chemistry Fundamentals 10, no. 1 (February 1, 1971): 174-75.
    doi:10.1021/i160037a032.
.. [12] Miqueu, C, D Broseta, J Satherley, B Mendiboure, J Lachaise, and
    A Graciaa. "An Extended Scaled Equation for the Temperature Dependence
    of the Surface Tension of Pure Compounds Inferred from an Analysis of
    Experimental Data." Fluid Phase Equilibria 172, no. 2 (July 5, 2000):
    169-82. doi:10.1016/S0378-3812(00)00384-8.
.. [13] Aleem, W., N. Mellon, S. Sufian, M. I. A. Mutalib, and D. Subbarao.
    "A Model for the Estimation of Surface Tension of Pure Hydrocarbon 
    Liquids." Petroleum Science and Technology 33, no. 23-24 (December 17, 
    2015): 1908-15. doi:10.1080/10916466.2015.1110593.
.. [14] Mersmann, Alfons, and Matthias Kind. "Prediction of Mechanical and 
    Thermal Properties of Pure Liquids, of Critical Data, and of Vapor 
    Pressure." Industrial & Engineering Chemistry Research, January 31, 
    2017. https://doi.org/10.1021/acs.iecr.6b04323.

"""
from math import log, exp
from ..base import functor, TDependentHandleBuilder, InterpolatedTDependentModel
from .._constants import N_A, k
from chemicals.interface import (
    sigma_data_Mulero_Cachadina,
    sigma_data_Jasper_Lange,
    sigma_data_Somayajulu,
    sigma_data_Somayajulu_2,
    sigma_data_VDI_PPDS_11,
)
from chemicals.miscdata import (
    VDI_saturation_dict,
    lookup_VDI_tabular_data,
)
from .dippr import DIPPR_EQ106

__all__ = ('REFPROP', 
           'Somayajulu', 
           'Jasper', 
           'Brock_Bird', 
           'Pitzer', 
           'Sastri_Rao', 
           'Zuo_Stenby',
           'Hakim_Steinberg_Stiel', 
           'Miqueu',
           'Mersmann_Kind'
)


### Regressed coefficient-based functions

@functor(var='sigma')
def REFPROP(T, Tc, sigma0, n0, sigma1=0, n1=0, sigma2=0, n2=0):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the regression-based REFPROP 
    method, as described in [1]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    sigma0,n0,sigma1,n1,sigma2,n2 : float
        Regressed coefficients.
    
    Notes
    -----
    Relatively recent, and most accurate. Function as implemented in [1]_:
        
    .. math::
        \sigma(T)=\sigma_0\left(1-\frac{T}{T_c}\right)^{n_0}+
        \sigma_1\left(1-\frac{T}{T_c}\right)^{n_1}+
        \sigma_2\left(1-\frac{T}{T_c}\right)^{n_2}
        
    Form of function returns imaginary results when T > Tc, in which case None 
    is returned instead.
    
    Examples
    --------
    Parameters for water at 298.15 K:
    
    >>> f = REFPROP(647.096, -0.1306, 2.471, 0.2151, 1.233)
    >>> f(298.15)
    0.07205503890847453
    
    """
    Tr = T/Tc
    invTr = 1. - Tr
    return sigma0*(invTr)**n0 + sigma1*(invTr)**n1 + sigma2*(invTr)**n2

@functor(var='sigma')
def Somayajulu(T, Tc, a, b, c):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Somayajulu method, as 
    described in [2]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    a,b,c : float
        Regressed coefficients.
    
    Notes
    -----
    The surface tension in [2]_ is given by:
    
    .. math::
        \sigma=aX^{5/4}+bX^{9/4}+cX^{13/4}
        
        X=(T_c-T)/T_c
    
    Internal units are mN/m. Form of function returns imaginary results
    when T > Tc; None is returned if this is the case. Function is claimed
    valid from the triple to the critical point. Results can be evaluated
    beneath the triple point.
    
    Examples
    --------
    Water at 300 K:
    
    >>> f = Somayajulu(647.126, 232.713514, -140.18645, -4.890098)
    >>> f(300)
    0.07166386387996757
    
    """
    X = (Tc-T)/Tc
    return (a*X**1.25 + b*X**2.25 + c*X**3.25)/1000.

@functor(var='sigma')
def Jasper(T, a, b):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Jasper method, as 
    described in [3]_ [4]_.
    
    Parameters
    ----------
    a,b : float
        Regressed coefficients.
    
    Notes
    -----
    Calculates surface tension of a fluid given two parameters, a linear
    fit in Celcius from [1]_ with data reprinted in [2]_.
    
    .. math::
        \sigma = a - bT
    
    Internal units are mN/m, and degrees Celcius.
    This function has been checked against several references.
    
    Examples
    --------
    >>> f = Jasper(24, 0.0773)
    >>> f(298.15)
    0.0220675
    
    """
    return (a - b*(T-273.15))/1000


### CSP methods

@functor(var='sigma')
def Brock_Bird(T, Tb, Tc, Pc):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Brock Bird method, as 
    described in [5]_.
    
    Parameters
    ----------
    Tb : float
        Boiling point temperature [K].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    
    Notes
    -----
    Calculates air-water surface tension  using the [1]_
    emperical method. Old and tested.
    
    .. math::
        \sigma = P_c^{2/3}T_c^{1/3}Q(1-T_r)^{11/9}
        
        Q = 0.1196 \left[ 1 + \frac{T_{br}\ln (P_c/1.01325)}{1-T_{br}}\right]-0.279
    
    Numerous arrangements of this equation are available.
    This is a DIPPR Procedure 7A: Method for the Surface Tension of Pure,
    Nonpolar, Nonhydrocarbon Liquids.
    The exact equation is not in the original paper.
    If the equation yields a negative result, returns None instead.
    
    Examples
    --------
    p-dichloribenzene at 412.15 K, from DIPPR; value differs due to a slight
    difference in method:
    
    >>> f = Brock_Bird(447.3, 685, 3.952E6)
    >>> f(412.15)
    0.02208448325192495
    
    Chlorobenzene from Poling, as compared with a % error value at 293 K:
    
    >>> f = Brock_Bird(404.75, 633.0, 4530000.0)
    >>> f(293.15)
    0.032985686413713036
    
    """
    Tbr = Tb/Tc
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    Q = 0.1196*(1 + Tbr*log(Pc/1.01325)/(1-Tbr))-0.279
    sigma = (Pc)**(2/3.)*Tc**(1/3.)*Q*(1-Tr)**(11/9.)
    return sigma/1000  # convert to N/m

@functor(var='sigma')
def Pitzer(T, Tc, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Pitzer method, as described in [6]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    
    Notes
    -----
    Calculates air-water surface tension using the correlation derived
    by [6]_ from the works of [7]_ and [8]_. Based on critical property CSP
    methods.
    
    .. math::
        \sigma = P_c^{2/3}T_c^{1/3}\frac{1.86 + 1.18\omega}{19.05}
        \left[ \frac{3.75 + 0.91 \omega}{0.291 - 0.08 \omega}\right]^{2/3} (1-T_r)^{11/9}
    
    The source of this equation has not been reviewed.
    Internal units of presure are bar, surface tension of mN/m.
    
    Examples
    --------
    Chlorobenzene from Poling, as compared with a % error value at 293 K:
    
    >>> f = Pitzer(633.0, 4530000.0, 0.249)
    >>> f(293.)
    0.03458453513446387
    
    """
    Tr = T/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = Pc**(2/3.0)*Tc**(1/3.0)*(1.86+1.18*omega)/19.05 * (
        (3.75+0.91*omega)/(0.291-0.08*omega))**(2/3.0)*(1-Tr)**(11/9.0)
    return sigma/1000.  # N/m, please

@functor(var='sigma')
def Sastri_Rao(T, Tb, Tc, Pc, chemicaltype=None):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Sastri Rao method, as
    described in [9]_.
    
    Parameters
    ----------
    Tb : float
        Boiling point temperature [K].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    
    Other Parameters
    ----------------
    chemicaltype : 'alcohol' or 'acid'
    
    Notes
    -----
    Calculates air-water surface tension using the correlation derived by
    [9]_ based on critical property CSP methods and chemical classes.
    
    .. math::
        \sigma = K P_c^xT_b^y T_c^z\left[\frac{1-T_r}{1-T_{br}}\right]^m
    
    The source of this equation has not been reviewed.
    Internal units of presure are bar, surface tension of mN/m.
    
    Examples
    --------
    Chlorobenzene from Poling, as compared with a % error value at 293 K.
    
    >>> f = Sastri_Rao(404.75, 633.0, 4530000.0)
    >>> f(293.15)
    0.03234567739694441
    
    """
    if chemicaltype == 'alcohol':
        k, x, y, z, m = 2.28, 0.25, 0.175, 0, 0.8
    elif chemicaltype == 'acid':
        k, x, y, z, m = 0.125, 0.50, -1.5, 1.85, 11/9.0
    else:
        k, x, y, z, m = 0.158, 0.50, -1.5, 1.85, 11/9.0
    Tr = T/Tc
    Tbr = Tb/Tc
    Pc = Pc/1E5  # Convert to bar
    sigma = k*Pc**x*Tb**y*Tc**z*((1 - Tr)/(1 - Tbr))**m 
    return sigma/1000.  # N/m

def ST_r(ST, Tc, Pc):
    return log(1. + ST/(Tc**(1/3.0)*Pc**(2/3.0)))

@functor(var='sigma')
def Zuo_Stenby(T, Tc, Pc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface
    tension (sigma; in N/m) of a chemical using the Zuo Stenby method, as
    described in [10]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    
    Notes
    -----
    Calculates air-water surface tension using the reference fluids
    methods of [10]_:
    
    .. math::
        \sigma^{(1)} = 40.520(1-T_r)^{1.287}
        
        \sigma^{(2)} = 52.095(1-T_r)^{1.21548}
        
        \sigma_r = \sigma_r^{(1)}+ \frac{\omega - \omega^{(1)}}
        {\omega^{(2)}-\omega^{(1)}} (\sigma_r^{(2)}-\sigma_r^{(1)})
        
        \sigma = T_c^{1/3}P_c^{2/3}[\exp{(\sigma_r)} -1]
    
    Presently untested. Have not personally checked the sources.
    I strongly believe it is broken.
    The reference values for methane and n-octane are from the DIPPR database.
    
    Examples
    --------
    Chlorobenzene:
    
    >>> f = Zuo_Stenby(633.0, 4530000.0, 0.249)
    >>> f(293.)
    0.03345569011871088    
    
    """
    Tc_1, Pc_1, omega_1 = 190.56, 4599000.0/1E5, 0.012
    Tc_2, Pc_2, omega_2 = 568.7, 2490000.0/1E5, 0.4
    Pc = Pc/1E5
    ST_1 = 40.520*(1 - T/Tc)**1.287  # Methane
    ST_2 = 52.095*(1 - T/Tc)**1.21548  # n-octane
    ST_r_1, ST_r_2 = ST_r(ST_1, Tc_1, Pc_1), ST_r(ST_2, Tc_2, Pc_2)
    sigma_r = ST_r_1 + (omega-omega_1)/(omega_2 - omega_1)*(ST_r_2-ST_r_1)
    sigma = Tc**(1/3.0)*Pc**(2/3.0)*(exp(sigma_r)-1)
    return sigma/1000  # N/m, please

@functor(var='sigma')
def Hakim_Steinberg_Stiel(T, Tc, Pc, omega, StielPolar=0):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface
    tension (sigma; in N/m) of a chemical using the Hakim Steinberg Stiel 
    method, as described in [11]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
    omega : float
        Acentric factor [-].
    StielPolar : float
        Regressed coefficient.
    
    Notes
    -----
    Calculates air-water surface tension using the reference fluids methods
    of [11]_:
    
    .. math::
        \sigma = 4.60104\times 10^{-7} P_c^{2/3}T_c^{1/3}Q_p \left(\frac{1-T_r}{0.4}\right)^m
        
        Q_p = 0.1574+0.359\omega-1.769\chi-13.69\chi^2-0.51\omega^2+1.298\omega\chi
        
        m = 1.21+0.5385\omega-14.61\chi-32.07\chi^2-1.65\omega^2+22.03\omega\chi
    
    Original equation for m and Q are used. Internal units are atm and mN/m.
    
    Examples
    --------
    1-butanol, as compared to value in CRC Handbook of 0.02493:
    
    >>> f = Hakim_Steinberg_Stiel(563.0, 4414000.0, 0.59, StielPolar=-0.07872)
    >>> f(298.15)
    0.021907902575190447
    
    """
    Q = (0.1574 + 0.359*omega - 1.769*StielPolar - 13.69*StielPolar**2
        - 0.510*omega**2 + 1.298*StielPolar*omega)
    m = (1.210 + 0.5385*omega - 14.61*StielPolar - 32.07*StielPolar**2
        - 1.656*omega**2 + 22.03*StielPolar*omega)
    Tr = T/Tc
    Pc = Pc/101325.
    sigma = Pc**(2/3.)*Tc**(1/3.)*Q*((1 - Tr)/0.4)**m
    sigma = sigma/1000.  # convert to N/m
    return sigma

@functor(var='sigma')
def Miqueu(T, Tc, Vc, omega):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface 
    tension (sigma; in N/m) of a chemical using the Miqueu method, as 
    described in [12]_.
    
    Parameters
    ----------
    Tc : float
        Critical point temperature [K].
    Vc : float
        Critical point volume [m^3/mol].
    omega : float
        Acentric factor [-].
    
    Notes
    -----
    Calculates air-water surface tension using the methods of [1]_.
    
    .. math::
        \sigma = k T_c \left( \frac{N_a}{V_c}\right)^{2/3}
        (4.35 + 4.14 \omega)t^{1.26}(1+0.19t^{0.5} - 0.487t)
    
    Uses Avogadro's constant and the Boltsman constant.
    Internal units of volume are mL/mol and mN/m. However, either a typo
    is in the article or author's work, or my value of k is off by 10; this is
    corrected nonetheless.
    Created with 31 normal fluids, none polar or hydrogen bonded. Has an
    AARD of 3.5%.
    
    Examples
    --------
    Bromotrifluoromethane, 2.45 mN/m:
    
    >>> f = Miqueu(340.1, 0.000199, 0.1687)
    >>> f(300.)
    0.003474099603581931
    
    """
    Vc = Vc*1E6
    t = 1.-T/Tc
    return k*Tc*(N_A/Vc)**(2/3.)*(4.35 + 4.14*omega)*t**1.26*(1+0.19*t**0.5 - 0.25*t)*10000

@functor(var='sigma')
def Mersmann_Kind(T, Tm, Tb, Tc, Pc, n_associated=1):
    r"""
    Create a functor of temperature (T; in K) that estimates the surface
    tension (sigma; in N/m) of a chemical using the Mersmann Kind surface
    tension method, as described in [14]_.
    
    Parameters
    ----------
    Tm : float
        Melting point temperature [K].
    Tb : float
        Boiling point temperature [K].
    Tc : float
        Critical point temperature [K].
    Pc : float
        Critical point pressure [Pa].
        
    Other Parameters
    ----------------
    n_associated : int
        Number of associated molecules in a cluster (2 for alcohols, 1
        otherwise), [-]
    
    Notes
    -----
    Estimates the surface tension of organic liquid substances
    according to the method of [1]_:
    
    .. math::
        \sigma^* = \frac{\sigma n_{ass}^{1/3}} {(kT_c)^{1/3} T_{rm}P_c^{2/3}}
        
        \sigma^* = \left(\frac{T_b - T_m}{T_m}\right)^{1/3}
        \left[6.25(1-T_r) + 31.3(1-T_r)^{4/3}\right]
    
    In the equation, all quantities must be in SI units. `k` is the boltzman
    constant.
    
    Examples
    --------
    MTBE at STP (the actual value is 0.0181):
        
    >>> f = Mersmann_Kind_surface_tension(164.15, 328.25, 497.1, 3430000.0)
    >>> f(298.15)
    0.016744309508833335
    
    """
    Tr = T/Tc
    sigma_star = ((Tb - Tm)/Tm)**(1/3.)*(6.25*(1. - Tr) + 31.3*(1. - Tr)**(4/3.))
    sigma = sigma_star*(k*Tc)**(1/3.)*(Tm/Tc)*Pc**(2/3.)*n_associated**(-1/3.)
    return sigma

@TDependentHandleBuilder('sigma')
def surface_tension_handle(handle, CAS, MW, Tb, Tc, Pc, Vc, Zc, omega, StielPolar):
    add_model = handle.add_model
    if CAS in sigma_data_Mulero_Cachadina:
        _, sigma0, n0, sigma1, n1, sigma2, n2, Tc, Tmin, Tmax = sigma_data_Mulero_Cachadina[CAS]
        STREFPROP_coeffs = (Tc, sigma0, n0, sigma1, n1, sigma2, n2)
        add_model(REFPROP.from_args(STREFPROP_coeffs), Tmin, Tmax)
    if CAS in sigma_data_Somayajulu_2:
        _, Tt, Tc, a, b, c = sigma_data_Somayajulu_2[CAS]
        SOMAYAJULU2_coeffs = (Tc, a, b, c)
        Tmin = Tt; Tmax = Tc
        add_model(Somayajulu.from_args(SOMAYAJULU2_coeffs), Tmin, Tmax)
    elif CAS in sigma_data_Somayajulu:
        _, Tt, Tc, a, b, c = sigma_data_Somayajulu[CAS]
        SOMAYAJULU_coeffs = (Tc, a, b, c)
        Tmin = Tt; Tmax = Tc
        add_model(Somayajulu.from_args(SOMAYAJULU_coeffs), Tmin, Tmax)
    if CAS in VDI_saturation_dict:
        Ts, Ys = lookup_VDI_tabular_data(CAS, 'sigma')
        Tmin = Ts[0]
        *Ts, Tmax = Ts
        Ys = Ys[:-1]
        add_model(InterpolatedTDependentModel(Ts, Ys), Tmin, Tmax)
    if CAS in sigma_data_Jasper_Lange:
        _, a, b, Tmin, Tmax= sigma_data_Jasper_Lange[CAS]
        JASPER_coeffs = (a, b)
        add_model(Jasper.from_args(JASPER_coeffs))
    data = (Tc, Vc, omega)
    if all(data):
        add_model(Miqueu.from_args(data), 0.0, Tc)
    data = (Tb, Tc, Pc)
    if all(data):
        add_model(Brock_Bird.from_args(data), 0.0, Tc)
        add_model(Sastri_Rao.from_args(data), 0.0, Tc)
    data = (Tc, Pc, omega)
    if all(data):
        add_model(Pitzer.from_args(data), 0.0, Tc)
        add_model(Zuo_Stenby.from_args(data), 0.0, Tc)
    if CAS in sigma_data_VDI_PPDS_11:
        _, Tm, Tc, a, b, c, d, e = sigma_data_VDI_PPDS_11[CAS]
        VDI_PPDS_coeffs = (Tc, a, b, c, d, e)
        add_model(DIPPR_EQ106.from_args(VDI_PPDS_coeffs))
