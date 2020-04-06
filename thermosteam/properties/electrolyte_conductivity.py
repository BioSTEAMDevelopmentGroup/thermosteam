# -*- coding: utf-8 -*-
"""
Data for the conductivity of electrolytes as presented in [1]_.

References
----------
.. [1] Speight, James. Lange's Handbook of Chemistry. 16 edition.
   McGraw-Hill Professional, 2005.

"""
import os
from collections import namedtuple
from math import e
from .._constants import N_A
from .utils import to_nums, CASDataReader
import pandas as pd

F = e*N_A

### Electrical Conductivity

read = CASDataReader(__file__, 'Electrolytes')
Lange_cond_pure = read('Lange Pure Species Conductivity.tsv')
Marcus_ion_conductivities = read('Marcus Ion Conductivities.tsv')
CRC_ion_conductivities = read('CRC conductivity infinite dilution.tsv')
Magomedovk_thermal_cond = read('Magomedov Thermal Conductivity.tsv')
CRC_aqueous_thermodynamics = read('CRC Thermodynamic Properties of Aqueous Ions.csv')
electrolyte_dissociation_reactions = read('Electrolyte dissociations.csv', sep=None)

McCleskey_parameters = namedtuple("McCleskey_parameters",
                                  ["Formula", 'lambda_coeffs', 'A_coeffs', 'B', 'multiplier'])

McCleskey_conductivities = {}
with open(os.path.join(read.folder, 'McCleskey Electrical Conductivity.csv')) as f:
    next(f)
    for line in f:
        values = line.strip().split('\t')
        formula, CASRN, lbt2, lbt, lbc, At2, At, Ac, B, multiplier = to_nums(values)
        McCleskey_conductivities[CASRN] = McCleskey_parameters(formula, 
            [lbt2, lbt, lbc], [At2, At, Ac], B, multiplier)

Lange_cond_pure = pd.read_csv(os.path.join(read.folder, 'Lange Pure Species Conductivity.tsv'),
                          sep='\t', index_col=0)

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
    
del read