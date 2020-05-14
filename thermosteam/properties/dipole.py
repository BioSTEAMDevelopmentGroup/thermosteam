# -*- coding: utf-8 -*-
'''
All methods and data associated to the dipole moment of a chemical.
'''


__all__ = ('dipole_moment',)

from .data import dipole_data_sources, get_from_data_sources

def dipole_moment(CASRN, method='Any'):
    r'''
    Retrieve a chemical's dipole moment. Lookup is based on CASRNs.
    Automatically select a data source to use if no method is provided.
    Return None if the data is not available.

    Prefered source is 'CCCBDB'. Considerable variation in reported data has
    found.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    dipole : float
        Dipole moment [debye].

    Other Parameters
    ----------------
    method : string, optional
        The method name to use. Accepted methods are 'CCCBDB', 'Muller', or
        'Poling'. If method is "Any", the first available
        value from these methods will returned. If method is "All",
        a dictionary of method results will be returned.

    Notes
    -----
    A total of three sources are available for this function. They are:

        * 'CCCBDB', a series of critically evaluated data for compounds in
          [1]_, intended for use in predictive modeling.
        * 'Muller', a collection of data in a
          group-contribution scheme in [2]_.
        * 'Poling', in the appendix in [3].
        
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
    return get_from_data_sources(dipole_data_sources, CASRN, 'Dipole', method)
