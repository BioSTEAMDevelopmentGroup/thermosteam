# -*- coding: utf-8 -*-
'''Chemical Engineering Design Library (ChEDL). Utilities for process modeling.
Copyright (C) 2016, Caleb Bell <Caleb.Andrew.Bell@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.'''

from __future__ import division

__all__ = ['Tc', 'Pc', 'Vc', 'Zc', 'Mersmann_Kind_predictor', 'third_property', 'critical_surface', 
           'Ihmels', 'Meissner', 'Grigoras', 'Li', 
           'Chueh_Prausnitz_Tc', 'Grieves_Thodos', 'modified_Wilson_Tc', 
           'Tc_mixture', 'Pc_mixture', 'Chueh_Prausnitz_Vc', 
           'modified_Wilson_Vc', 'Vc_mixture']
__all__.extend(['Tc_methods', 'Pc_methods', 'Vc_methods', 'Zc_methods', 
                'critical_surface_methods', '_crit_IUPAC', '_crit_Matthews', 
                '_crit_CRC', '_crit_PSRKR4', '_crit_PassutDanner', '_crit_Yaws'])

import os
import numpy as np
import pandas as pd
from math import log
from .utils import CASDataReader
from ..constants import R, N_A

### Read the various data files
# TODO: check out 12E of this data http://pubsdc3.acs.org/doi/10.1021/acs.jced.5b00571

read = CASDataReader(__file__, 'Critical Properties')
_crit_IUPAC = read('IUPACOrganicCriticalProps.tsv')  # IUPAC Organic data series
_crit_Matthews = read('Mathews1972InorganicCriticalProps.tsv')
_crit_CRC = read('CRCCriticalOrganics.tsv') # CRC Handbook from TRC Organic data section (only in 2015)
_crit_PSRKR4 = read('Appendix to PSRK Revision 4.tsv')
_crit_PassutDanner = read('PassutDanner1973.tsv')
_crit_Yaws = read('Yaws Collection.tsv')
 
_crit_CRC.df['Zc'] = pd.Series(_crit_CRC.df['Pc']*_crit_CRC.df['Vc']/_crit_CRC.df['Tc']/R,
                            index=_crit_CRC.index)
_crit_PSRKR4.df['Zc'] = pd.Series(_crit_PSRKR4.df['Pc']*_crit_PSRKR4.df['Vc']/_crit_PSRKR4.df['Tc']/R,
                               index=_crit_PSRKR4.index)
_crit_Yaws.df['Zc'] = pd.Series(_crit_Yaws.df['Pc']*_crit_Yaws.df['Vc']/_crit_Yaws.df['Tc']/R,
                                index=_crit_Yaws.index)

### Strings defining each method

IUPAC = 'IUPAC'
MATTHEWS = 'MATTHEWS'
CRC = 'CRC'
PSRK = 'PSRK'
PD = 'PD'
YAWS = 'YAWS'
SURF = 'SURF'
NONE = 'NONE'
Tc_methods = [IUPAC, MATTHEWS, CRC, PSRK, PD, YAWS, SURF]


def Tc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    temperature. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Tc : float
        Critical temperature, [K]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Tc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'PD', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Tc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Tc for the desired chemical, and will return methods instead of Tc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of seven sources are available for this function. They are:

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
        * Critical Surface', an estimation method using a
          simple quadratic method for estimating Tc from Pc and Vc. This is
          ignored and not returned as a method by default, as no compounds
          have values of Pc and Vc but not Tc currently.

    Examples
    --------
    >>> Tc(CASRN='64-17-5')
    514.0

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
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Tc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Tc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Tc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Tc']):
            methods.append(PSRK)
        if CASRN in _crit_PassutDanner.index and not np.isnan(_crit_PassutDanner.at[CASRN, 'Tc']):
            methods.append(PD)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Tc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Tc = float(_crit_IUPAC.at[CASRN, 'Tc'])
    elif Method == MATTHEWS:
        _Tc = float(_crit_Matthews.at[CASRN, 'Tc'])
    elif Method == PSRK:
        _Tc = float(_crit_PSRKR4.at[CASRN, 'Tc'])
    elif Method == PD:
        _Tc = float(_crit_PassutDanner.at[CASRN, 'Tc'])
    elif Method == CRC:
        _Tc = float(_crit_CRC.at[CASRN, 'Tc'])
    elif Method == YAWS:
        _Tc = float(_crit_Yaws.at[CASRN, 'Tc'])
    elif Method == SURF:
        _Tc = third_property(CASRN=CASRN, T=True)
    elif Method == NONE:
        _Tc = None
    else:
        raise Exception('Failure in in function')
    return _Tc


Pc_methods = [IUPAC, MATTHEWS, CRC, PSRK, PD, YAWS, SURF]


def Pc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    pressure. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Pc(CASRN='64-17-5')
    6137000.0

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Pc : float
        Critical pressure, [Pa]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Pc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'PD', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Pc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Pc for the desired chemical, and will return methods instead of Pc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of seven sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'PD', an older compillation of
          data published in [16]_
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [17]_.
        * SURF', an estimation method using a
          simple quadratic method for estimating Pc from Tc and Vc. This is
          ignored and not returned as a method by default.

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
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Pc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Pc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Pc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Pc']):
            methods.append(PSRK)
        if CASRN in _crit_PassutDanner.index and not np.isnan(_crit_PassutDanner.at[CASRN, 'Pc']):
            methods.append(PD)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Pc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Pc = float(_crit_IUPAC.at[CASRN, 'Pc'])
    elif Method == MATTHEWS:
        _Pc = float(_crit_Matthews.at[CASRN, 'Pc'])
    elif Method == CRC:
        _Pc = float(_crit_CRC.at[CASRN, 'Pc'])
    elif Method == PSRK:
        _Pc = float(_crit_PSRKR4.at[CASRN, 'Pc'])
    elif Method == PD:
        _Pc = float(_crit_PassutDanner.at[CASRN, 'Pc'])
    elif Method == YAWS:
        _Pc = float(_crit_Yaws.at[CASRN, 'Pc'])
    elif Method == SURF:
        _Pc = third_property(CASRN=CASRN, P=True)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Pc


Vc_methods = [IUPAC, MATTHEWS, CRC, PSRK, YAWS, SURF]


def Vc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[SURF]):
    r'''This function handles the retrieval of a chemical's critical
    volume. Lookup is based on CASRNs. Will automatically select a data
    source to use if no Method is provided; returns None if the data is not
    available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Vc(CASRN='64-17-5')
    0.000168

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Vc : float
        Critical volume, [m^3/mol]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Vc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'YAWS', and 'SURF'. All valid values are also held  
        in the list `Vc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Vc for the desired chemical, and will return methods instead of Vc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of six sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [16]_.
        * 'SURF', an estimation method using a
          simple quadratic method for estimating Pc from Tc and Vc. This is
          ignored and not returned as a method by default

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
    .. [16] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Vc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Vc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Vc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Vc']):
            methods.append(PSRK)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Vc']):
            methods.append(YAWS)
        if CASRN:
            methods.append(SURF)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == IUPAC:
        _Vc = float(_crit_IUPAC.at[CASRN, 'Vc'])
    elif Method == PSRK:
        _Vc = float(_crit_PSRKR4.at[CASRN, 'Vc'])
    elif Method == MATTHEWS:
        _Vc = float(_crit_Matthews.at[CASRN, 'Vc'])
    elif Method == CRC:
        _Vc = float(_crit_CRC.at[CASRN, 'Vc'])
    elif Method == YAWS:
        _Vc = float(_crit_Yaws.at[CASRN, 'Vc'])
    elif Method == SURF:
        _Vc = third_property(CASRN=CASRN, V=True)
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Vc


COMBINED = 'COMBINED'
Zc_methods = [IUPAC, MATTHEWS, CRC, PSRK, YAWS, COMBINED]


def Zc(CASRN, AvailableMethods=False, Method=None, IgnoreMethods=[COMBINED]):
    r'''This function handles the retrieval of a chemical's critical
    compressibility. Lookup is based on CASRNs. Will automatically select a
    data source to use if no Method is provided; returns None if the data is
    not available.

    Prefered sources are 'IUPAC' for organic chemicals, and 'MATTHEWS' for 
    inorganic chemicals. Function has data for approximately 1000 chemicals.

    Examples
    --------
    >>> Zc(CASRN='64-17-5')
    0.24100000000000002

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Zc : float
        Critical compressibility, [-]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain Vc with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'IUPAC', 'MATTHEWS', 
        'CRC', 'PSRK', 'YAWS', and 'COMBINED'. All valid values are also held  
        in `Zc_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        Zc for the desired chemical, and will return methods instead of Zc
    IgnoreMethods : list, optional
        A list of methods to ignore in obtaining the full list of methods,
        useful for for performance reasons and ignoring inaccurate methods

    Notes
    -----
    A total of five sources are available for this function. They are:

        * 'IUPAC', a series of critically evaluated
          experimental datum for organic compounds in [1]_, [2]_, [3]_, [4]_,
          [5]_, [6]_, [7]_, [8]_, [9]_, [10]_, [11]_, and [12]_.
        * 'MATTHEWS', a series of critically
          evaluated data for inorganic compounds in [13]_.
        * 'CRC', a compillation of critically
          evaluated data by the TRC as published in [14]_.
        * 'PSRK', a compillation of experimental and
          estimated data published in [15]_.
        * 'YAWS', a large compillation of data from a
          variety of sources; no data points are sourced in the work of [16]_.

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
    .. [16] Yaws, Carl L. Thermophysical Properties of Chemicals and
       Hydrocarbons, Second Edition. Amsterdam Boston: Gulf Professional
       Publishing, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in _crit_IUPAC.index and not np.isnan(_crit_IUPAC.at[CASRN, 'Zc']):
            methods.append(IUPAC)
        if CASRN in _crit_Matthews.index and not np.isnan(_crit_Matthews.at[CASRN, 'Zc']):
            methods.append(MATTHEWS)
        if CASRN in _crit_CRC.index and not np.isnan(_crit_CRC.at[CASRN, 'Zc']):
            methods.append(CRC)
        if CASRN in _crit_PSRKR4.index and not np.isnan(_crit_PSRKR4.at[CASRN, 'Zc']):
            methods.append(PSRK)
        if CASRN in _crit_Yaws.index and not np.isnan(_crit_Yaws.at[CASRN, 'Zc']):
            methods.append(YAWS)
        if Tc(CASRN) and Vc(CASRN) and Pc(CASRN):
            methods.append(COMBINED)
        if IgnoreMethods:
            for Method in IgnoreMethods:
                if Method in methods:
                    methods.remove(Method)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IUPAC:
        _Zc = float(_crit_IUPAC.at[CASRN, 'Zc'])
    elif Method == PSRK:
        _Zc = float(_crit_PSRKR4.at[CASRN, 'Zc'])
    elif Method == MATTHEWS:
        _Zc = float(_crit_Matthews.at[CASRN, 'Zc'])
    elif Method == CRC:
        _Zc = float(_crit_CRC.at[CASRN, 'Zc'])
    elif Method == YAWS:
        _Zc = float(_crit_Yaws.at[CASRN, 'Zc'])
    elif Method == COMBINED:
        _Zc = Vc(CASRN)*Pc(CASRN)/Tc(CASRN)/R
    elif Method == NONE:
        return None
    else:
        raise Exception('Failure in in function')
    return _Zc


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
    atomic composition according to [1]_ and [2]_. This is a crude approach,
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
    
    References
    ----------
    .. [1] Mersmann, Alfons, and Matthias Kind. "Correlation for the Prediction
       of Critical Molar Volume." Industrial & Engineering Chemistry Research,
       October 16, 2017. https://doi.org/10.1021/acs.iecr.7b03171.
    .. [2] Mersmann, Alfons, and Matthias Kind. "Prediction of Mechanical and 
       Thermal Properties of Pure Liquids, of Critical Data, and of Vapor 
       Pressure." Industrial & Engineering Chemistry Research, January 31, 
       2017. https://doi.org/10.1021/acs.iecr.6b04323.
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
    r'''Most recent, and most recommended method of estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 421 organic compounds to derive equation.
    The general equation is in [1]_:

    .. math::
        P_c = -0.025 + 2.215 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

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

    References
    ----------
    .. [1] Ihmels, E. Christian. "The Critical Surface." Journal of Chemical
           & Engineering Data 55, no. 9 (September 9, 2010): 3474-80.
           doi:10.1021/je100167w.
    '''
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = -0.025+2.215*Tc/Vc
        Pc = Pc*1E6  # MPa to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = 443*Tc/(200*Pc+5)
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/1E6  # Pa to MPa
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 5.0/443*(40*Pc*Vc + Vc)
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


def Meissner(Tc=None, Pc=None, Vc=None):
    r'''Old (1942) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 42 organic and inorganic compounds to derive the equation.
    The general equation is in [1]_:

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
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

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

    References
    ----------
    .. [1] Meissner, H. P., and E. M. Redding. "Prediction of Critical
           Constants." Industrial & Engineering Chemistry 34, no. 5
           (May 1, 1942): 521-26. doi:10.1021/ie50389a003.
    '''
    if Tc and Vc:
        Vc = Vc*1E6
        Pc = 20.8*Tc/(Vc-8)
        Pc = 101325*Pc  # atm to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/101325.  # Pa to atm
        Vc = 104/5.0*Tc/Pc+8
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/101325.  # Pa to atm
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 5./104.0*Pc*(Vc-8)
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


def Grigoras(Tc=None, Pc=None, Vc=None):
    r'''Relatively recent (1990) relationship for estimating critical
    properties from each other. Two of the three properties are required.
    This model uses the "critical surface", a general plot of Tc vs Pc vs Vc.
    The model used 137 organic and inorganic compounds to derive the equation.
    The general equation is in [1]_:

    .. math::
        P_c = 2.9 + 20.2 \frac{T_c}{V_c}

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

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

    References
    ----------
    .. [1] Grigoras, Stelian. "A Structural Approach to Calculate Physical
           Properties of Pure Organic Substances: The Critical Temperature,
           Critical Volume and Related Properties." Journal of Computational
           Chemistry 11, no. 4 (May 1, 1990): 493-510.
           doi:10.1002/jcc.540110408
    '''
    if Tc and Vc:
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Pc = 2.9 + 20.2*Tc/Vc
        Pc = Pc*1E5  # bar to Pa
        return Pc
    elif Tc and Pc:
        Pc = Pc/1E5  # Pa to bar
        Vc = 202.0*Tc/(10*Pc-29.0)
        Vc = Vc/1E6  # cm^3/mol to m^3/mol
        return Vc
    elif Pc and Vc:
        Pc = Pc/1E5  # Pa to bar
        Vc = Vc*1E6  # m^3/mol to cm^3/mol
        Tc = 1.0/202*(10*Pc-29.0)*Vc
        return Tc
    else:
        raise Exception('Two of Tc, Pc, and Vc must be provided')


IHMELS = 'IHMELS'
MEISSNER = 'MEISSNER'
GRIGORAS = 'GRIGORAS'
critical_surface_methods = [IHMELS, MEISSNER, GRIGORAS]


def critical_surface(Tc=None, Pc=None, Vc=None, AvailableMethods=False,
                     Method=None):
    r'''Function for calculating a critical property of a substance from its
    other two critical properties. Calls functions Ihmels, Meissner, and
    Grigoras, each of which use a general 'Critical surface' type of equation.
    Limited accuracy is expected due to very limited theoretical backing.

    Parameters
    ----------
    Tc : float
        Critical temperature of fluid (optional) [K]
    Pc : float
        Critical pressure of fluid (optional) [Pa]
    Vc : float
        Critical volume of fluid (optional) [m^3/mol]
    AvailableMethods : bool
        Request available methods for given parameters
    Method : string
        Request calculation uses the requested method

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----

    Examples
    --------
    Decamethyltetrasiloxane [141-62-8]

    >>> critical_surface(Tc=599.4, Pc=1.19E6, Method='IHMELS')
    0.0010927333333333334
    '''
    def list_methods():
        methods = []
        if (Tc and Pc) or (Tc and Vc) or (Pc and Vc):
            methods.append(IHMELS)
            methods.append(MEISSNER)
            methods.append(GRIGORAS)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]
    # This is the calculate, given the method section
    if Method == IHMELS:
        Third = Ihmels(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == MEISSNER:
        Third = Meissner(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == GRIGORAS:
        Third = Grigoras(Tc=Tc, Pc=Pc, Vc=Vc)
    elif Method == NONE:
        Third = None
    else:
        raise Exception('Failure in in function')
    return Third


def third_property(CASRN=None, T=False, P=False, V=False):
    r'''Function for calculating a critical property of a substance from its
    other two critical properties, but retrieving the actual other critical
    values for convenient calculation.
    Calls functions Ihmels, Meissner, and
    Grigoras, each of which use a general 'Critical surface' type of equation.
    Limited accuracy is expected due to very limited theoretical backing.

    Parameters
    ----------
    CASRN : string
        The CAS number of the desired chemical
    T : bool
        Estimate critical temperature
    P : bool
        Estimate critical pressure
    V : bool
        Estimate critical volume

    Returns
    -------
    Tc, Pc or Vc : float
        Critical property of fluid [K], [Pa], or [m^3/mol]

    Notes
    -----
    Avoids recursion only by eliminating the None and critical surface options
    for calculating each critical property. So long as it never calls itself.
    Note that when used by Tc, Pc or Vc, this function results in said function
    calling the other functions (to determine methods) and (with method specified)

    Examples
    --------
    >>> # Decamethyltetrasiloxane [141-62-8]
    >>> third_property('141-62-8', V=True)
    0.0010920041152263375

    >>> # Succinic acid 110-15-6
    >>> third_property('110-15-6', P=True)
    6095016.233766234
    '''
    Third = None
    if V:
        Tc_methods = Tc(CASRN, AvailableMethods=True)[0:-2]
        Pc_methods = Pc(CASRN, AvailableMethods=True)[0:-2]
        if Tc_methods and Pc_methods:
            _Tc = Tc(CASRN=CASRN, Method=Tc_methods[0])
            _Pc = Pc(CASRN=CASRN, Method=Pc_methods[0])
            Third = critical_surface(Tc=_Tc, Pc=_Pc, Vc=None)
    elif P:
        Tc_methods = Tc(CASRN, AvailableMethods=True)[0:-2]
        Vc_methods = Vc(CASRN, AvailableMethods=True)[0:-2]
        if Tc_methods and Vc_methods:
            _Tc = Tc(CASRN=CASRN, Method=Tc_methods[0])
            _Vc = Vc(CASRN=CASRN, Method=Vc_methods[0])
            Third = critical_surface(Tc=_Tc, Vc=_Vc, Pc=None)
    elif T:
        Pc_methods = Pc(CASRN, AvailableMethods=True)[0:-2]
        Vc_methods = Vc(CASRN, AvailableMethods=True)[0:-2]
        if Pc_methods and Vc_methods:
            _Pc = Pc(CASRN=CASRN, Method=Pc_methods[0])
            _Vc = Vc(CASRN=CASRN, Method=Vc_methods[0])
            Third = critical_surface(Pc=_Pc, Vc=_Vc, Tc=None)
    else:
        raise Exception('Error in function')
    if not Third:
        return None
    return Third

