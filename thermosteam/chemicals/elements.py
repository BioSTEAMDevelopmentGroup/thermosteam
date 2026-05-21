# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the elements module from the chemicals library:
# https://github.com/CalebBell/chemicals
# Copyright (C) 2020 Caleb Bell <Caleb.Andrew.Bell@gmail.com>
#
# This module is under a dual license:
# 1. The UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
# 
# 2. The MIT open-source license. See
# https://github.com/CalebBell/chemicals/blob/master/LICENSE.txt for details.
from chemicals import elements, periodic_table
import numpy as np

elements.__all__.extend([
    'atoms_to_array', 'array_to_atoms',
])

#: Dict[str, dict[str, int]] Cache of atomic counts.
formula_to_atoms = {}

#: Dict[str, int] Symbol - index pairs for atomic arrays.
symbol_to_index = {e.symbol: e.number - 1 for e in periodic_table}

#: tuple[str] Symbols for atomic arrays.
symbols = tuple(symbol_to_index)

def atoms_to_array(atoms: dict) -> np.ndarray:
    index = symbol_to_index
    array = np.zeros(118)
    if atoms is None: return array
    for symbol, value in atoms.items():
        array[index[symbol]] = value
    return array

def array_to_atoms(array: np.ndarray) -> dict:
    index, = np.where(array != 0.)
    return dict(zip([symbols[i] for i in index], array[index]))

def get_atoms(formula):
    if formula in formula_to_atoms:
        return formula_to_atoms[formula]
    else:
        formula_to_atoms[formula] = atoms = elements.simple_formula_parser(formula)
        if len(formula_to_atoms) > 50: del formula_to_atoms[next(iter(formula_to_atoms))]
    return atoms.copy() # Prevent cached atoms from being altered

def mass_fractions(atoms: dict[str, int], MW: float | None=None) -> dict[str, float]:
    r"""Calculates the mass fractions of each element in a compound,
    given a dictionary of its atoms and their counts, in the format
    {symbol: count}.

    .. math::
        w_i =  \frac{n_i MW_i}{\sum_i n_i MW_i}

    Parameters
    ----------
    atoms : dict
        Dictionary of counts of individual atoms, indexed by symbol with
        proper capitalization, [-]
    MW : float, optional
        Molecular weight, [g/mol]

    Returns
    -------
    mfracs : dict
        Dictionary of mass fractions of individual atoms, indexed by symbol
        with proper capitalization, [-]

    Notes
    -----
    Molecular weight is optional, but speeds up the calculation slightly. It
    is calculated using the function `molecular_weight` if not specified.

    Elemental data is from rdkit, with CAS numbers added. An exception is
    raised if an incorrect element symbol is given. Elements up to 118 are
    supported.

    Examples
    --------
    >>> mass_fractions({'H': 12, 'C': 20, 'O': 5})
    {'H': 0.03639798802478244, 'C': 0.7228692758981262, 'O': 0.24073273607709128}

    References
    ----------
    .. [1] RDKit: Open-source cheminformatics; http://www.rdkit.org
    """
    if not MW: MW = elements.molecular_weight(atoms)
    mfracs = {}
    for i, count in atoms.items():
        if i in periodic_table:
            mfracs[i] = periodic_table[i].MW*count/MW
        elif i == "D":
            mfracs[i] = 2.014102*count / MW
        elif i == "T":
            mfracs[i] = 3.0160492*count / MW
        elif i == 'Ash':
            mfracs[i] = count / MW
        else:
            raise ValueError("Molecule includes unknown atoms")
    return mfracs

elements.mass_fractions = mass_fractions
elements.get_atoms = get_atoms 
elements.atoms_to_array = atoms_to_array
elements.array_to_atoms = array_to_atoms
elements.symbol_to_index = symbol_to_index
elements.symbols = symbols