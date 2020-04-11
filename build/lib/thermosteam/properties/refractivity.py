# -*- coding: utf-8 -*-
"""
All data and methods for estimating a chemical's refractivity.
"""
import os
import pandas as pd
from math import pi
from .._constants import N_A

__all__ = ['CRC_RI_organic', 'RI_methods', 'refractive_index', 
           'polarizability_from_RI', 'molar_refractivity_from_RI', 
           'RI_from_molar_refractivity']

folder = os.path.join(os.path.dirname(__file__), 'Data/Misc')

CRC_RI_organic = pd.read_csv(os.path.join(folder, 'CRC Handbook Organic RI.csv'),
                             sep='\t', index_col=0)

CRC = 'CRC'
NONE = 'NONE'
RI_methods = [CRC]


def refractive_index(CASRN, T=None, AvailableMethods=False, Method=None,
                     full_info=True):
    r'''This function handles the retrieval of a chemical's refractive
    index. Lookup is based on CASRNs. Will automatically select a data source
    to use if no Method is provided; returns None if the data is not available.

    Function has data for approximately 4500 chemicals.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    RI : float
        Refractive Index on the Na D line, [-]
    T : float, only returned if full_info == True
        Temperature at which refractive index reading was made
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain RI with the given inputs

    Other Parameters
    ----------------
    Method : string, optional
        A string for the method name to use, as defined by constants in
        RI_methods
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        RI for the desired chemical, and will return methods instead of RI
    full_info : bool, optional
        If True, function will return the temperature at which the refractive
        index reading was made

    Notes
    -----
    Only one source is available in this function. It is:

        * 'CRC', a compillation of Organic RI data in [1]_.

    Examples
    --------
    >>> refractive_index(CASRN='64-17-5')
    (1.3611, 293.15)

    References
    ----------
    .. [1] Haynes, W.M., Thomas J. Bruno, and David R. Lide. CRC Handbook of
       Chemistry and Physics, 95E. Boca Raton, FL: CRC press, 2014.
    '''
    def list_methods():
        methods = []
        if CASRN in CRC_RI_organic.index:
            methods.append(CRC)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == CRC:
        _RI = float(CRC_RI_organic.at[CASRN, 'RI'])
        if full_info:
            _T = float(CRC_RI_organic.at[CASRN, 'RIT'])
    elif Method == NONE:
        _RI, _T = None, None
    else:
        raise Exception('Failure in in function')
    if full_info:
        return _RI, _T
    else:
        return _RI


def polarizability_from_RI(RI, Vm):
    r'''Returns the polarizability of a fluid given its molar volume and
    refractive index.

    .. math::
        \alpha = \left(\frac{3}{4\pi N_A}\right)
        \left(\frac{n^2-1}{n^2+2}\right)V_m

    Parameters
    ----------
    RI : float
        Refractive Index on Na D line, [-]
    Vm : float
        Molar volume of fluid, [m^3/mol]

    Returns
    -------
    alpha : float
        Polarizability [m^3]

    Notes
    -----
    This Lorentz-Lorentz-expression is most correct when van der Waals
    interactions dominate. Alternate conversions have been suggested.
    This is often expressed in units of cm^3 or Angstrom^3. To convert to these
    units, multiply by 1E9 or 1E30 respectively.

    Examples
    --------
    >>> polarizability_from_RI(1.3611, 5.8676E-5)
    5.147658123614415e-30

    References
    ----------
    .. [1] Panuganti, Sai R., Fei Wang, Walter G. Chapman, and Francisco M.
       Vargas. "A Simple Method for Estimation of Dielectric Constants and
       Polarizabilities of Nonpolar and Slightly Polar Hydrocarbons."
       International Journal of Thermophysics 37, no. 7 (June 6, 2016): 1-24.
       doi:10.1007/s10765-016-2075-8.
    '''
    return 3/(4*pi*N_A)*(RI**2-1)/(RI**2+2)*Vm


def molar_refractivity_from_RI(RI, Vm):
    r'''Returns the molar refractivity of a fluid given its molar volume and
    refractive index.

    .. math::
        R_m = \left(\frac{n^2-1}{n^2+2}\right)V_m

    Parameters
    ----------
    RI : float
        Refractive Index on Na D line, [-]
    Vm : float
        Molar volume of fluid, [m^3/mol]

    Returns
    -------
    Rm : float
        Molar refractivity [m^3/mol]

    Notes
    -----

    Examples
    --------
    >>> molar_refractivity_from_RI(1.3611, 5.8676E-5)
    1.2985217089649597e-05

    References
    ----------
    .. [1] Panuganti, Sai R., Fei Wang, Walter G. Chapman, and Francisco M.
       Vargas. "A Simple Method for Estimation of Dielectric Constants and
       Polarizabilities of Nonpolar and Slightly Polar Hydrocarbons."
       International Journal of Thermophysics 37, no. 7 (June 6, 2016): 1-24.
       doi:10.1007/s10765-016-2075-8.
    '''
    return (RI**2 - 1.)/(RI**2 + 2.)*Vm


def RI_from_molar_refractivity(Rm, Vm):
    r'''Returns the refractive index of a fluid given its molar volume and
    molar refractivity.

    .. math::
        RI = \sqrt{\frac{-2R_m - V_m}{R_m-V_m}}

    Parameters
    ----------
    Rm : float
        Molar refractivity [m^3/mol]
    Vm : float
        Molar volume of fluid, [m^3/mol]

    Returns
    -------
    RI : float
        Refractive Index on Na D line, [-]

    Notes
    -----

    Examples
    --------
    >>> RI_from_molar_refractivity(1.2985e-5, 5.8676E-5)
    1.3610932757685672

    References
    ----------
    .. [1] Panuganti, Sai R., Fei Wang, Walter G. Chapman, and Francisco M.
       Vargas. "A Simple Method for Estimation of Dielectric Constants and
       Polarizabilities of Nonpolar and Slightly Polar Hydrocarbons."
       International Journal of Thermophysics 37, no. 7 (June 6, 2016): 1-24.
       doi:10.1007/s10765-016-2075-8.
    '''
    Rm = ((-2*Rm - Vm)/(Rm-Vm))**0.5
    return Rm


### Mixtures

#def Lorentz_Lorenz(ws=None, RIs=None, rhos=None, rho=None):
#    if not length_check([ws, RIs, rhos]):
#        raise Exception('Input dimentions are inconsistent')
