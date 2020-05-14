# -*- coding: utf-8 -*-
"""
All data and methods for estimating a chemical's triple point.

References
----------
.. [1] Staveley, L. A. K., L. Q. Lobo, and J. C. G. Calado. "Triple-Points
       of Low Melting Substances and Their Use in Cryogenic Work." Cryogenics
       21, no. 3 (March 1981): 131-144. doi:10.1016/0011-2275(81)90264-2.

"""

from .data import triple_point_sources, get_from_data_sources

__all__ = ('triple_point_temperature', 
           'triple_point_pressure',
)


def triple_point_temperature(CASRN, method='Any'):
    r'''
    Return a chemical's triple point temperature. Lookup is based on CASRNs.
    Automatically select a data source to use if no Method is provided.
    Return None if the data is not available.

    Parameters
    ----------
    CASRN : string
        CASRN [-].

    Returns
    -------
    Tt : float
        Triple point temperature [K].

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. The only accepted method for now is 'Staveley'.
        If method is "Any", the first available value from these methods
        will returned. If method is "All", a dictionary of method results will
        be returned.

    Notes
    -----
    Returns data from [1]_.
    Median difference between melting points and triple points is 0.02 K.
    Accordingly, this should be more than good enough for engineering
    applications.

    Temperatures are on the ITS-68 scale.

    Examples
    --------
    Ammonia

    >>> triple_point_temperature('7664-41-7')
    195.47999999999999
    
    '''
    return get_from_data_sources(triple_point_sources, CASRN, 'Tt68', method)

def triple_point_pressure(CASRN, method='Any'):
    r'''
    Return a chemical's triple point pressure. Lookup is based on CASRNs.
    Automatically select a data source to use if no Method is provided.
    Return None if the data is not available.

    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    Pt : float
        Triple point pressure [Pa].

    Other Parameters
    ----------------
    Method : string, optional
        The method name to use. The only accepted method for now is 'Staveley'.
        If method is "Any", the first available value from these methods
        will returned. If method is "All", a dictionary of method results will
        be returned.

    Notes
    -----
    Returns data from [1]_.

    Examples
    --------
    Ammonia

    >>> triple_point_pressure('7664-41-7')
    6079.5
    
    '''
    return get_from_data_sources(triple_point_sources, CASRN, 'Pt', method)
