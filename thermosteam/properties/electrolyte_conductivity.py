# -*- coding: utf-8 -*-
"""
Data for the conductivity of electrolytes as presented in [1]_.

References
----------
.. [1] Speight, James. Lange's Handbook of Chemistry. 16 edition.
   McGraw-Hill Professional, 2005.

"""
from .data import Lange_cond_pure

### Electrical Conductivity

def conductivity(CASRN):
    r'''
    Lookup a chemical's conductivity by CASRN.
    
    Parameters
    ----------
    CASRN : string
        CASRN [-]

    Returns
    -------
    kappa : float
        Electrical conductivity of the fluid, [S/m]
    T : float
        Temperature at which conductivity measurement was made
        
    Notes
    -----
    Only one source is available in this function: Lange's Handbook, Table 8.34 Electrical Conductivity of Various Pure Liquids', a compillation of data in [1]_. This function has data for approximately 100 chemicals.

    Examples
    --------
    >>> conductivity('7732-18-5')
    (4e-06, 291.15)

    '''
    kappa = float(Lange_cond_pure.at[CASRN, 'Conductivity'])
    T = float(Lange_cond_pure.at[CASRN, 'T'])
    return kappa, T
    