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
All data and methods related to the estimation of a chemical's critical properties.

References
----------
.. [1] Ambrose, Douglas, and Colin L. Young. "Vapor-Liquid Critical
   Properties of Elements and Compounds. 1. An Introductory Survey."
   Journal of Chemical & Engineering Data 41, no. 1 (January 1, 1996):
   154-154. doi:10.1021/je950378q.
.. [2] Ambrose, Douglas, and Constantine Tsonopoulos. "Vapor-Liquid
   Critical Properties of Elements and Compounds. 2. Normal Alkanes."
   Journal of Chemical & Engineering Data 40, no. 3 (May 1, 1995): 531-46.
   doi:10.1021/je00019a001.
.. [3] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
   Critical Properties of Elements and Compounds. 3. Aromatic
   Hydrocarbons." Journal of Chemical & Engineering Data 40, no. 3
   (May 1, 1995): 547-58. doi:10.1021/je00019a002.
.. [4] Gude, Michael, and Amyn S. Teja. "Vapor-Liquid Critical Properties
   of Elements and Compounds. 4. Aliphatic Alkanols." Journal of Chemical
   & Engineering Data 40, no. 5 (September 1, 1995): 1025-36.
   doi:10.1021/je00021a001.
.. [5] Daubert, Thomas E. "Vapor-Liquid Critical Properties of Elements
   and Compounds. 5. Branched Alkanes and Cycloalkanes." Journal of
   Chemical & Engineering Data 41, no. 3 (January 1, 1996): 365-72.
   doi:10.1021/je9501548.
.. [6] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
   Critical Properties of Elements and Compounds. 6. Unsaturated Aliphatic
   Hydrocarbons." Journal of Chemical & Engineering Data 41, no. 4
   (January 1, 1996): 645-56. doi:10.1021/je9501999.
.. [7] Kudchadker, Arvind P., Douglas Ambrose, and Constantine Tsonopoulos.
   "Vapor-Liquid Critical Properties of Elements and Compounds. 7. Oxygen
   Compounds Other Than Alkanols and Cycloalkanols." Journal of Chemical &
   Engineering Data 46, no. 3 (May 1, 2001): 457-79. doi:10.1021/je0001680.
.. [8] Tsonopoulos, Constantine, and Douglas Ambrose. "Vapor-Liquid
   Critical Properties of Elements and Compounds. 8. Organic Sulfur,
   Silicon, and Tin Compounds (C + H + S, Si, and Sn)." Journal of Chemical
   & Engineering Data 46, no. 3 (May 1, 2001): 480-85.
   doi:10.1021/je000210r.
.. [9] Marsh, Kenneth N., Colin L. Young, David W. Morton, Douglas Ambrose,
   and Constantine Tsonopoulos. "Vapor-Liquid Critical Properties of
   Elements and Compounds. 9. Organic Compounds Containing Nitrogen."
   Journal of Chemical & Engineering Data 51, no. 2 (March 1, 2006):
   305-14. doi:10.1021/je050221q.
.. [10] Marsh, Kenneth N., Alan Abramson, Douglas Ambrose, David W. Morton,
   Eugene Nikitin, Constantine Tsonopoulos, and Colin L. Young.
   "Vapor-Liquid Critical Properties of Elements and Compounds. 10. Organic
   Compounds Containing Halogens." Journal of Chemical & Engineering Data
   52, no. 5 (September 1, 2007): 1509-38. doi:10.1021/je700336g.
.. [11] Ambrose, Douglas, Constantine Tsonopoulos, and Eugene D. Nikitin.
   "Vapor-Liquid Critical Properties of Elements and Compounds. 11. Organic
   Compounds Containing B + O; Halogens + N, + O, + O + S, + S, + Si;
   N + O; and O + S, + Si." Journal of Chemical & Engineering Data 54,
   no. 3 (March 12, 2009): 669-89. doi:10.1021/je800580z.
.. [12] Ambrose, Douglas, Constantine Tsonopoulos, Eugene D. Nikitin, David
   W. Morton, and Kenneth N. Marsh. "Vapor-Liquid Critical Properties of
   Elements and Compounds. 12. Review of Recent Data for Hydrocarbons and
   Non-Hydrocarbons." Journal of Chemical & Engineering Data, October 5,
   2015, 151005081500002. doi:10.1021/acs.jced.5b00571.
.. [13] Mathews, Joseph F. "Critical Constants of Inorganic Substances."
   Chemical Reviews 72, no. 1 (February 1, 1972): 71-100.
   doi:10.1021/cr60275a004.
.. [14] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
   Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
.. [15] Horstmann, Sven, Anna Jabłoniec, Jörg Krafczyk, Kai Fischer, and
   Jürgen Gmehling. "PSRK Group Contribution Equation of State:
   Comprehensive Revision and Extension IV, Including Critical Constants
   and Α-Function Parameters for 1000 Components." Fluid Phase Equilibria
   227, no. 2 (January 25, 2005): 157-64. doi:10.1016/j.fluid.2004.11.002.
.. [16] Passut, Charles A., and Ronald P. Danner. "Acentric Factor. A
   Valuable Correlating Parameter for the Properties of Hydrocarbons."
   Industrial & Engineering Chemistry Process Design and Development 12,
   no. 3 (July 1, 1973): 365–68. doi:10.1021/i260047a026.
.. [17] Yaws, Carl L. Thermophysical Properties of Chemicals and
   Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
   Publishing, 2014.
.. [18] Mersmann, Alfons, and Matthias Kind. "Correlation for the Prediction
    of Critical Molar Volume." Industrial & Engineering Chemistry Research,
    October 16, 2017. https://doi.org/10.1021/acs.iecr.7b03171.
.. [19] Mersmann, Alfons, and Matthias Kind. "Prediction of Mechanical and 
   Thermal Properties of Pure Liquids, of Critical Data, and of Vapor 
   Pressure." Industrial & Engineering Chemistry Research, January 31, 
   2017. https://doi.org/10.1021/acs.iecr.6b04323.
.. [20] Ihmels, E. Christian. "The Critical Surface." Journal of Chemical
    & Engineering Data 55, no. 9 (September 9, 2010): 3474-80.
    doi:10.1021/je100167w.
.. [21] Meissner, H. P., and E. M. Redding. "Prediction of Critical
    Constants." Industrial & Engineering Chemistry 34, no. 5
    (May 1, 1942): 521-26. doi:10.1021/ie50389a003.
.. [22] Grigoras, Stelian. "A Structural Approach to Calculate Physical
    Properties of Pure Organic Substances: The Critical Temperature,
    Critical Volume and Related Properties." Journal of Computational
    Chemistry 11, no. 4 (May 1, 1990): 493-510.
    doi:10.1002/jcc.540110408

"""

from collections import namedtuple
from .data import critical_data_sources, get_from_data_sources
from .._constants import N_A

__all__ = ('critical_point_temperature',
           'critical_point_pressure',
           'critical_point_volume',
           'Mersmann_Kind_predictor',
           'Ihmels', 'Meissner', 'Grigoras',)
 

# %%

CriticalPoint = namedtuple('CriticalPoint', ('Tc', 'Pc', 'Vc'), module=__file__)
del namedtuple

def critical_point_temperature(CASRN, method='Any'):
    r'''
    This function handles the retrieval of a chemical's critical
    temperature. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'Matthews' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tc : float
        Critical temperature, [K]

    Other Parameters
    ----------------
    method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'Matthews', 
        'CRC', 'PSRK', 'Passut Danner', and 'Yaws'. If method is "Any",
        the first available value from these methods will returned.
        If method is "All", a dictionary of method results will be returned.

    Notes
    -----
    A total of six sources are available for this function. They are:

        * 'IUPAC Organic Critical Properties', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'Matthews Inorganic Critical Properties', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC Organic Critical Properties', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK Revision 4 Appendix', a compillation of experimental and
          estimated data published in [15]_.
        * 'Passut Danner 1973 Critical Properties', an older compillation of
          data published in [16]_
        * 'Yaws Critical Properties', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.

    Examples
    --------
    >>> critical_point_temperature(CASRN='64-17-5')
    514.0

    '''
    return get_from_data_sources(critical_data_sources, CASRN, 'Tc', method)


def critical_point_pressure(CASRN, method='Any'):
    r'''
    Retrieve the critical pressure of a chemical. Lookup is based on CASRNs.
    Automatically select a data source to use if no Method is provided.
    Return None if the data is not available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'Matthews' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> critical_point_pressure(CASRN='64-17-5')
    6137000.0

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Pc : float
        Critical pressure, [Pa]

    Other Parameters
    ---------------
    method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'Matthews', 
        'CRC', 'PSRK', 'Passut Danner', and 'Yaws'. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.

    Notes
    -----
    A total of six sources are available for this function. They are:

        * 'IUPAC Organic Critical Properties', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'Matthews Inorganic Critical Properties', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC Organic Critical Properties', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK Revision 4 Appendix', a compillation of experimental and
          estimated data published in [15]_.
        * 'Passut Danner 1973 Critical Properties', an older compillation of
          data published in [16]_
        * 'Yaws Critical Properties', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
          
    '''
    return get_from_data_sources(critical_data_sources, CASRN, 'Pc', method)


def critical_point_volume(CASRN, method='Any'):
    r'''
    Retrieve the critical volume of a chemical. Lookup is based on CASRNs.
    Automatically select a data source to use if no Method is provided;
    return None if the data is not available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'Matthews' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> critical_point_volume(CASRN='64-17-5')
    0.000168

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Vc : float
        Critical volume, [m^3/mol]

    Other Parameters
    ---------------
    method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'Matthews', 
        'CRC', 'PSRK', 'Passut Danner', and 'Yaws'. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.

    Notes
    -----
    A total of six sources are available for this function. They are:

        * 'IUPAC Organic Critical Properties', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'Matthews Inorganic Critical Properties', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC Organic Critical Properties', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK Revision 4 Appendix', a compillation of experimental and
          estimated data published in [15]_.
        * 'Passut Danner 1973 Critical Properties', an older compillation of
          data published in [16]_
        * 'Yaws Critical Properties', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
          
    '''
    return get_from_data_sources(critical_data_sources, CASRN, 'Vc', method)

rcovs_Mersmann_Kind = {'C': 0.77, 'Cl': 0.99, 'I': 1.33, 'H': 0.37, 'F': 0.71, 
                       'S': 1.04, 'O': 0.6, 'N': 0.71, 'Si': 1.17, 'Br': 1.14}

rcovs_regressed =  {
    u'Nb': 0.5139380605234125,
    u'Ne': 0.7708216694154189,
    u'Al': 1.004994775098707,
    u'Re': 1.1164444694484814,
    u'Rb': 2.9910506044828837,
    u'Rn': 1.9283158156480653,
    u'Xe': 1.694221043013319,
    u'Ta': 1.1185133195453156,
    u'Bi': 1.8436438207262267,
    u'Br': 1.3081458724155532,
    u'Hf': 0.8829545460486594,
    u'Mo': 0.740396259301556,
    u'He': 0.9808144122544257,
    u'C': 0.6068586007600608,
    u'B': 0.7039677272439753,
    u'F': 0.5409105884533288,
    u'I': 1.7262432419406561,
    u'H': 0.33296601702348533,
    u'K': 0.7384112258842432,
    u'O': 0.5883254088243008,
    u'N': 0.5467979701131293,
    u'P': 1.0444655158949694,
    u'Si': 1.4181434041348049,
    u'U': 1.5530287578073485,
    u'Sn': 1.3339487990207999,
    u'W': 0.8355335838735266,
    u'V': 0.6714619384794069,
    u'Sb': 0.8840680681215854,
    u'Se': 1.5747549515496795,
    u'Ge': 1.0730584829731715,
    u'Kr': 1.393999829252709,
    u'Cl': 1.0957835025011224,
    u'S': 1.0364452121761167,
    u'Hg': 0.6750818243474633,
    u'As': 0.6750687692915264,
    u'Ar': 1.2008872952022298,
    u'Cs': 3.433699060142929,
    u'Zr': 0.9346554283483623}


def Mersmann_Kind_predictor(atoms, coeff=3.645, power=0.5, 
                            covalent_radii=rcovs_Mersmann_Kind):
    r'''Predicts the critical molar volume of a chemical based only on its
    atomic composition according to [18]_ and [19]_. This is a crude approach,
    but provides very reasonable
    estimates in practice. Optionally, the `coeff` used and the `power` in the
    fraction as well as the atomic contributions can be adjusted; this method
    is general and atomic contributions can be regressed to predict other
    properties with this routine.
    
    .. math::
        \frac{\left(\frac{V_c}{n_a N_A}\right)^{1/3}}{d_a}
        = \frac{3.645}{\left(\frac{r_a}{r_H}\right)^{1/2}}

        r_a = d_a/2
        
        d_a = 2 \frac{\sum_i (n_i r_i)}{n_a}
        
    In the above equations, :math:`n_i` is the number of atoms of species i in
    the molecule, :math:`r_i` is the covalent atomic radius of the atom, and 
    :math:`n_a` is the total number of atoms in the molecule.
    
    Parameters
    ----------
    atoms : dict
        Dictionary of atoms and their counts, [-]
    coeff : float, optional
        Coefficient used in the relationship, [m^2]
    power : float, optional
        Power applied to the relative atomic radius, [-]
    covalent_radii : dict or indexable, optional
        Object which can be indexed to atomic contrinbutions (by symbol), [-]

    Returns
    -------
    Vc : float
        Predicted critical volume of the chemical, [m^3/mol]
    
    Notes
    -----    
    Using the :obj:`thermo.elements.periodic_table` covalent radii (from RDKit), 
    the coefficient and power should be 4.261206523632586 and 0.5597281770786228
    respectively for best results.
    
    Examples
    --------
    Prediction of critical volume of decane:
        
    >>> Mersmann_Kind_predictor({'C': 10, 'H': 22})
    0.0005851859052024729
    
    This is compared against the experimental value, 0.000624 (a 6.2% relative
    error)
    
    Using custom fitted coefficients we can do a bit better:
        
    >>> from thermo.critical import rcovs_regressed
    >>> Mersmann_Kind_predictor({'C': 10, 'H': 22}, coeff=4.261206523632586, 
    ... power=0.5597281770786228, covalent_radii=rcovs_regressed)
    0.0005956871011923075
    
    The relative error is only 4.5% now. This is compared to an experimental 
    uncertainty of 5.6%.
    
    Evaluating 1321 critical volumes in the database, the average relative
    error is 5.0%; standard deviation 6.8%; and worst value of 79% relative
    error for phosphorus.
    
    '''
    H_RADIUS_COV = covalent_radii['H']
    tot = 0
    atom_count = 0
    for atom, count in atoms.items():
        if atom not in covalent_radii:
            raise Exception('Atom %s is not supported by the supplied dictionary' %atom)
        tot += count*covalent_radii[atom]
        atom_count += count
    da = 2.*tot/atom_count
    ra = da/2.
    da_SI = da*1e-10 # Convert from angstrom to m
    return ((coeff/(ra/H_RADIUS_COV)**power)*da_SI)**3*N_A*atom_count


### Critical Property Relationships


def Ihmels(Tc=None, Pc=None, Vc=None):
    r'''
    Most recent, and most recommended method of estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 421 organic compounds to derive equation.
    The general equation is in [20]_:

    .. math::
        P_c = -0.025 + 2.215 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float, optional
        Critical temperature of fluid [K]
    Pc : float, optional
        Critical pressure of fluid [Pa]
    Vc : float, optional
        Critical volume of fluid [m^3/mol]

    Returns
    -------
    critical_point : CriticalPoint
        Critical point of the fluid [K], [Pa], and [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are MPa, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    Their equation was also compared with 56 inorganic and elements.
    Devations of 20% for <200K or >1000K points.

    Examples
    --------a
    Succinic acid [110-15-6]

    >>> Ihmels(Tc=851.0, Vc=0.000308)
    6095016.233766234

    '''
    critical_point = CriticalPoint(Tc, Pc, Vc)
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = -0.025+2.215*Tc/Vc
        critical_point.Pc = Pc*1E6  # MPa to Pa
    elif Tc and Pc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = 443*Tc/(200*Pc+5)
        critical_point.Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        critical_point.Tc = 5.0/443*(40*Pc*Vc + Vc)
    else:
        raise ValueError('Two of Tc, Pc, and Vc must be provided')
    return critical_point


def Meissner(Tc=None, Pc=None, Vc=None):
    r'''Old (1942) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 42 organic and inorganic compounds to derive the equation.
    The general equation is in [21]_:

    .. math::
        P_c = \frac{2.08 T_c}{V_c-8}

    Parameters
    ----------
    Tc : float, optional
        Critical temperature of fluid [K]
    Pc : float, optional
        Critical pressure of fluid [Pa]
    Vc : float, optional
        Critical volume of fluid [m^3/mol]

    Returns
    -------
    critical_point : CriticalPoint
        Critical point of the fluid [K], [Pa], and [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are atm, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    This equation is less accurate than that of Ihmels, but surprisingly close.
    The author also proposed means of estimated properties independently.

    Examples
    --------
    Succinic acid [110-15-6]

    >>> Meissner(Tc=851.0, Vc=0.000308)
    5978445.199999999

    '''
    critical_point = CriticalPoint(Tc, Pc, Vc)
    if Tc and Vc:
        Vc = Vc*1E6
        Pc = 20.8*Tc/(Vc-8)
        critical_point.Pc = 101325*Pc  # atm to Pa
    elif Tc and Pc:
        Pc = Pc/101325.  # Pa to atm
        Vc = 104/5.0*Tc/Pc+8
        critical_point.Vc = Vc/1E6  # cm^3/mol to m^3/mol
    elif Pc and Vc:
        Pc = Pc/101325.  # Pa to atm
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        critical_point.Tc = 5./104.0*Pc*(Vc-8)
    else:
        raise ValueError('Two of Tc, Pc, and Vc must be provided')
    return critical_point


def Grigoras(Tc=None, Pc=None, Vc=None):
    r'''Relatively recent (1990) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 137 organic and inorganic compounds to derive the equation.
    The general equation is in [22]_:

    .. math::
        P_c = 2.9 + 20.2 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float, optional
        Critical temperature of fluid [K]
    Pc : float, optional
        Critical pressure of fluid [Pa]
    Vc : float, optional
        Critical volume of fluid [m^3/mol]

    Returns
    -------
    critical_point : CriticalPoint
        Critical point of the fluid [K], [Pa], and [m^3/mol]

    Notes
    -----
    The prediction of Tc from Pc and Vc is not tested, as this is not necessary
    anywhere, but it is implemented.
    Internal units are bar, cm^3/mol, and K. A slight error occurs when
    Pa, cm^3/mol and K are used instead, on the order of <0.2%.
    This equation is less accurate than that of Ihmels, but surprisingly close.
    The author also investigated an early QSPR model.

    Examples
    --------
    Succinic acid [110-15-6]

    >>> Grigoras(Tc=851.0, Vc=0.000308)
    5871233.766233766

    '''
    critical_point = CriticalPoint(Tc, Pc, Vc)
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = 2.9 + 20.2*Tc/Vc
        critical_point.Pc = Pc*1E5  # bar to Pa
    elif Tc and Pc:
        Pc = Pc/1E5  # Pa to bar
        Vc = 202.0*Tc/(10*Pc-29.0)
        critical_point.Vc = Vc/1E6  # cm^3/mol to m^3/mol
    elif Pc and Vc:
        Pc = Pc/1E5  # Pa to bar
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        critical_point.Tc = 1.0/202*(10*Pc-29.0)*Vc
    else:
        raise ValueError('Two of Tc, Pc, and Vc must be provided')
    return critical_point

