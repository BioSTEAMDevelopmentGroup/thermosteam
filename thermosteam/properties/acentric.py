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
'''
All data and methods for estimating a chemical's accentric factor.

References
----------
.. [1] Pitzer, K. S., D. Z. Lippmann, R. F. Curl, C. M. Huggins, and
   D. E. Petersen: The Volumetric and Thermodynamic Properties of Fluids.
   II. Compressibility Factor, Vapor Pressure and Entropy of Vaporization.
   J. Am. Chem. Soc., 77: 3433 (1955).
.. [2] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
   Jürgen Gmehling. "PSRK Group Contribution Equation of State:
   Comprehensive Revision and Extension IV, Including Critical Constants
   and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
   227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
.. [3] Passut, Charles A., and Ronald P. Danner. "Acentric Factor. A
   Valuable Correlating Parameter for the Properties of Hydrocarbons."
   Industrial & Engineering Chemistry Process Design and Development 12,
   no. 3 (July 1, 1973): 365-68. doi:10.1021/i260047a026.
.. [4] Yaws, Carl L. Thermophysical Properties of Chemicals and
   Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
   Publishing, 2014.
.. [5] Lee, Byung Ik, and Michael G. Kesler. "A Generalized Thermodynamic
   Correlation Based on Three-Parameter Corresponding States." AIChE Journal
   21, no. 3 (1975): 510-527. doi:10.1002/aic.690210313.

'''

__all__ = ('compute_acentric_factor',
           'estimate_acentric_factor_LK',
           'acentric_factor_PSRK', 'acentric_factor_LK',
           'acentric_factor_PD', 'acentric_factor_Yaws',
           'acentric_factor', 'Stiel_Polar', 'compute_Stiel_Polar'
)

from math import log, log10
from .critical import (critical_point_temperature,
                       critical_point_pressure)
from .data import (critical_data_PSRKR4, 
                   critical_data_PassutDanner, 
                   critical_data_Yaws)
from .phase_change import normal_boiling_point_temperature
from .vapor_pressure import vapor_pressure_handle


# %% Direct computation methods

def compute_acentric_factor(P_at_Tr_seventenths, Pc):
    return -log10(P_at_Tr_seventenths/Pc) - 1.0

def estimate_acentric_factor_LK(Tb, Tc, Pc):
    T_br = Tb/Tc
    omega = (log(101325.0/Pc) - 5.92714 + 6.09648/T_br + 1.28862*log(T_br) -
             0.169347*T_br**6)/(15.2518 - 15.6875/T_br - 13.4721*log(T_br) +
             0.43577*T_br**6)
    return omega

def compute_Stiel_Polar(P_at_Tr_sixtenths, omega, Pc):
    Pr = P_at_Tr_sixtenths/Pc
    return log10(Pr) + 1.70*omega + 1.552

# %% Retrievers

acentric_factor_PSRK = critical_data_PSRKR4.retriever('omega')
acentric_factor_PD = critical_data_PassutDanner.retriever('omega')
acentric_factor_Yaws = critical_data_Yaws.retriever('omega')

def acentric_factor_DEFINITION(CASRN, 
                     Tc=None, Pc=None,
                     Psat=None):
    r"""
    Compute acentric factor by definition. Return None if acentric factor
    cannot be computed.

    Parameters
    ----------
    CASRN : str
        CASRN for retrieving Tb, Tc, and Pc.
    Tc : float, optional
        Critical temperature of the fluid [K]
    Pc : float, optional
        Critical pressure of the fluid [Pa]
    Psat : function(T)
        Model to calculate vapor pressure [Pa].
    
    Notes
    -----
    Calculation is based on the definition of omega as presented in [1]_,
    using vapor pressure data:
    
    .. math::
        \omega \equiv -\log_{10}\left[\lim_{T/T_c=0.7}(P^{sat}/P_c)\right]-1.0

    """
    if not Pc: Pc = critical_point_pressure(CASRN)
    if not Pc: return 
    if not Tc: Tc = critical_point_temperature(CASRN)
    if not Tc: return 
    if not Psat: Psat = vapor_pressure_handle([CASRN, None, Tc, Pc, None])
    P_at_Tr_seventenths = Psat.try_out(0.7 * Tc)
    if not P_at_Tr_seventenths: return
    return compute_acentric_factor(P_at_Tr_seventenths, Pc)

def acentric_factor_LK(CASRN, Tb=None, Tc=None, Pc=None):
    r'''
    Estimate the acentric factor of a fluid using a correlation in [5]_.
    Return None if acentric factor cannot be estimated.

    Parameters
    ----------
    CASRN : str
        CASRN for retrieving Tb, Tc, and Pc.
    Tb : float, optional
        Boiling temperature of the fluid [K]
    Tc : float, optional
        Critical temperature of the fluid [K]
    Pc : float, optional
        Critical pressure of the fluid [Pa]
        
    Notes
    -----
    Internal units are atmosphere and Kelvin. The correlation is as follows:
    
    .. math::
        \omega = \frac{\ln P_{br}^{sat} - 5.92714 + 6.09648/T_{br} + 1.28862
        \ln T_{br} -0.169347T_{br}^6}
        {15.2518 - 15.6875/T_{br} - 13.4721 \ln T_{br} + 0.43577 T_{br}^6}

    Examples
    --------
    Example value for Isopropylbenzene from Reid (1987):

    >>> acentric_factor_LK(425.6, 631.1, 32.1E5)
    0.32544249926397856

    Using ASPEN V8.4, LK method gives 0.325595. 
    
    '''
    if not Tb: Tb = normal_boiling_point_temperature(CASRN),
    if not Tc: Tc = critical_point_temperature(CASRN),
    if not Pc: Pc = critical_point_pressure(CASRN)
    if not (Tb and Tc and Pc): return None
    return estimate_acentric_factor_LK(Tb, Tc, Pc)


# %% Retrieval interface

def acentric_factor(CASRN, 
                    Tb=None, Tc=None, Pc=None, Psat=None,
                    method="Any"):
    r'''
    Retrieve the acentric factor of a chemical, `omega`, or calculate it
    from correlations or directly through the definition of acentric factor,
    if possible. Requires a known boiling point, critical temperature
    and pressure for use of the correlations. Requires accurate vapor
    pressure data for direct calculation.

    Will automatically select a method to use if no method is provided;
    returns None if the data is not available and cannot be calculated.

    Examples
    --------
    >>> acentric_factor(CASRN='64-17-5')
    0.635

    Parameters
    ----------
    CASRN : str

    Returns
    -------
    omega : float or dict(str-float)
        Acentric factor of the fluid [-].

    Other Parameters
    ----------------
    method : string, optional
        The method name to use. Accepted methods are 'PSRK', 'PD', 'Yaws', 
        'LK', and 'DEFINITION'. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.
    Tb : float, optional
        Boiling temperature of the fluid [K]
    Tc : float, optional
        Critical temperature of the fluid [K]
    Pc : float, optional
        Critical pressure of the fluid [Pa]
    Psat : function(T), optional
        Model to calculate vapor pressure [Pa].
    
    Notes
    -----
    A total of five sources are available for this function. They are:

        * 'PSRK', a compillation of experimental and estimated data published 
          in the Appendix of [2]_, the fourth revision of the PSRK model.
        * 'PD', an older compillation of
          data published in (Passut & Danner, 1973) [3]_.
        * 'Yaws', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [4]_.
        * 'LK', a estimation method for hydrocarbons [5]_.
        * 'DEFINITION', based on the definition of omega as
          presented in [1]_, using vapor pressure data.
    
    See also
    --------
    acentric_factor_LK
    acentric_factor_DEFINITION
    
    '''
    if method == 'All':
        omega = {'PSRK': acentric_factor_PSRK(CASRN),
                 'PD': acentric_factor_PD(CASRN),
                 'Yaws': acentric_factor_Yaws(CASRN),
                 'LK': acentric_factor_LK(Tb, Tc, Pc),            
                 'DEFINITION': acentric_factor_DEFINITION(CASRN,
                                                Tc, Pc, Psat)
                 }
    elif method == 'Any':
        omega = (acentric_factor_PSRK(CASRN) 
                 or acentric_factor_PD(CASRN)
                 or acentric_factor_Yaws(CASRN)
                 or acentric_factor_LK(Tb, Tc, Pc)
                 or acentric_factor_DEFINITION(CASRN, Tc, Pc, Psat)
                 )
    elif method == 'PSRK':
        omega = acentric_factor_PSRK(CASRN)
    elif method == 'PD':
        omega = acentric_factor_PD(CASRN)
    elif method == 'Yaws':
        omega = acentric_factor_Yaws(CASRN)
    elif method == 'LK':
        omega = acentric_factor_LK(Tb, Tc, Pc)
    elif method == 'DEFINITION':
        omega = acentric_factor_DEFINITION(CASRN, Tc, Pc, Psat)
    else:
        raise ValueError("invalid method; method must be one of the following: "
                         "'PSRK', 'PD', 'Yaws', 'LK', or 'DEFINITION'.")
    return omega

def Stiel_Polar(CASRN, 
               Tc=None, Pc=None, omega=None, Psat=None):
    r'''
    Return the Stiel Polar factor of a chemical through the definition of 
    Stiel-polar factor if possible.


    Parameters
    ----------
    CASRN : string
    Tc : float, optional
        Critical temperature of fluid [K].
    Pc : float, optional
        Critical pressure of fluid [Pa].
    omega : float, optional
        Acentric factor of the fluid [-]
    Psat : function(T)
        Vapor pressure model [Pa].

    Returns
    -------
    factor : float
        Stiel polar factor of compound

    Notes
    -----
    The Stiel Polar factor as presented in [1]_ is given by:
        
    .. math::
        x = \log P_r|_{T_r=0.6} + 1.70 \omega + 1.552

    A few points have also been published in [2]_, which may be used for
    comparison. Currently this is only used for a surface tension correlation.

    Examples
    --------
    >>> Stiel_Polar(647.3, 22048321.0, 0.344, CASRN='7732-18-5')
    0.024581140348734376

    References
    ----------
    .. [1] Halm, Roland L., and Leonard I. Stiel. "A Fourth Parameter for the
       Vapor Pressure and Entropy of Vaporization of Polar Fluids." AIChE
       Journal 13, no. 2 (1967): 351-355. doi:10.1002/aic.690130228.
    .. [2] D, Kukoljac Miloš, and Grozdanić Dušan K. "New Values of the
       Polarity Factor." Journal of the Serbian Chemical Society 65, no. 12
       (January 1, 2000). http://www.shd.org.rs/JSCS/Vol65/No12-Pdf/JSCS12-07.pdf
    
    '''
    if not Tc: Tc = critical_point_temperature(CASRN)
    if not Tc: return
    if not Pc: Pc = critical_point_pressure(CASRN)
    if not Pc: return
    if not Psat: Psat = vapor_pressure_handle(CASRN, Tc=Tc, Pc=Pc)
    if not Psat: return
    if not omega: omega = acentric_factor(CASRN,
                                          Pc=Pc, Psat=Psat,
                                          Tc=Tc)
    if not omega: return
    P_at_Tr_sixtenths = Psat(0.6 * Tc)
    return compute_Stiel_Polar(P_at_Tr_sixtenths, omega, Pc)

