# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module extends the dipole module from the chemicals's library:
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
    'atoms_to_array', 'array_to_atoms', 'atomic_index',
])

#: Dict[str, int] Symbol - index pairs for atomic arrays.
symbol_to_index = {e.symbol: e.number - 1 for e in periodic_table}

#: tuple[str] Symbols for atomic arrays.
symbols = tuple(symbol_to_index)

def atoms_to_array(atoms: dict) -> np.ndarray:
    index = symbol_to_index
    array = np.zeros(118)
    for symbol, value in atoms.items():
        array[index[symbol]] = value
    return array

def array_to_atoms(array: np.ndarray) -> dict:
    index, = np.where(array != 0.)
    return dict(zip([symbols[i] for i in index], array[index]))

elements.atoms_to_array = atoms_to_array
elements.array_to_atoms = array_to_atoms
elements.symbol_to_index = symbol_to_index
elements.symbols = symbols