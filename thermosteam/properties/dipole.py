# -*- coding: utf-8 -*-
'''
All methods and data associated to the dipole moment of a chemical.
'''


__all__ = ['dipole_moment', '_dipole_Poling', '_dipole_CCDB', '_dipole_Muller', 'dipole_methods']
import numpy as np
from .utils import CASDataReader

read = CASDataReader(__file__, 'Misc')
_dipole_Poling = read('Poling Dipole.csv')
_dipole_CCDB = read('cccbdb.nist.gov Dipoles.csv')
_dipole_Muller = read('Muller Supporting Info Dipoles.csv')

CCCBDB = 'CCCBDB'
MULLER = 'MULLER'
POLING = 'POLING'
NONE = 'NONE'

dipole_methods = [CCCBDB, MULLER, POLING]


def dipole_moment(CASRN, AvailableMethods=False, Method=None):
    r'''
    Retrieve a chemical's dipole moment. Lookup is based on CASRNs.
    Automatically select a data source to use if no Method is provided.
    Returns None if the data is not available.

    Prefered source is 'CCCBDB'. Considerable variation in reported data has
    found.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    dipole : float
        Dipole moment, [debye]
    methods : list, only returned if AvailableMethods == True
        List of methods which can be used to obtain dipole moment with the
        given inputs

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. Accepted methods are 'CCCBDB', 'MULLER', or
        'POLING'. All valid values are also held in the list `dipole_methods`.
    AvailableMethods : bool, optional
        If True, function will determine which methods can be used to obtain
        the dipole moment for the desired chemical, and will return methods
        instead of the dipole moment

    Notes
    -----
    A total of three sources are available for this function. They are:

        * 'CCCBDB', a series of critically evaluated data for compounds in
          [1]_, intended for use in predictive modeling.
        * 'MULLER', a collection of data in a
          group-contribution scheme in [2]_.
        * 'POLING', in the appendix in [3].
        
    This function returns dipole moment in units of Debye. This is actually
    a non-SI unit; to convert to SI, multiply by 3.33564095198e-30 and its
    units will be in ampere*second^2 or equivalently and more commonly given,
    coulomb*second. The constant is the result of 1E-21/c, where c is the
    speed of light.
        
    Examples
    --------
    >>> dipole_moment(CASRN='64-17-5')
    1.44

    References
    ----------
    .. [1] NIST Computational Chemistry Comparison and Benchmark Database
       NIST Standard Reference Database Number 101 Release 17b, September 2015,
       Editor: Russell D. Johnson III http://cccbdb.nist.gov/
    .. [2] Muller, Karsten, Liudmila Mokrushina, and Wolfgang Arlt. "Second-
       Order Group Contribution Method for the Determination of the Dipole
       Moment." Journal of Chemical & Engineering Data 57, no. 4 (April 12,
       2012): 1231-36. doi:10.1021/je2013395.
    .. [3] Poling, Bruce E. The Properties of Gases and Liquids. 5th edition.
       New York: McGraw-Hill Professional, 2000.
    '''
    def list_methods():
        methods = []
        if CASRN in _dipole_CCDB.index and not np.isnan(_dipole_CCDB.at[CASRN, 'Dipole']):
            methods.append(CCCBDB)
        if CASRN in _dipole_Muller.index and not np.isnan(_dipole_Muller.at[CASRN, 'Dipole']):
            methods.append(MULLER)
        if CASRN in _dipole_Poling.index and not np.isnan(_dipole_Poling.at[CASRN, 'Dipole']):
            methods.append(POLING)
        methods.append(NONE)
        return methods
    if AvailableMethods:
        return list_methods()
    if not Method:
        Method = list_methods()[0]

    if Method == CCCBDB:
        dipole = float(_dipole_CCDB.at[CASRN, 'Dipole'])
    elif Method == MULLER:
        dipole = float(_dipole_Muller.at[CASRN, 'Dipole'])
    elif Method == POLING:
        dipole = float(_dipole_Poling.at[CASRN, 'Dipole'])
    elif Method == NONE:
        dipole = None
    else:
        raise Exception('Failure in in function')
    return dipole
