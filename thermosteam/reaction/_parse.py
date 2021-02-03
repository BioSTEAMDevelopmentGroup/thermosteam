# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.

import numpy as np

__all__ = ('get_stoichiometric_array',
           'get_stoichiometric_string',
           'str2dct', 'dct2str', 'arr2dct')

def get_stoichiometric_array(reaction, chemicals):
    """Return stoichiometric array given a string defining the reaction and chemicals."""
    isa = isinstance
    if isa(reaction, dict):
        stoichiometry_dict = reaction
    elif isa(reaction, str):
        stoichiometry_dict = str2dct(reaction)
    elif isa(reaction, np.ndarray):
        return reaction
    else:
        raise ValueError(f"reaction must be either a str, dict, or array; not a '{type(reaction).__name__}' object")
    stoichiometric_array = dct2arr(stoichiometry_dict, chemicals)
    return stoichiometric_array

def get_stoichiometric_string(reaction, chemicals):
    """Return a string defining the reaction given the stoichiometric array and chemicals."""
    if isinstance(reaction, dict):
        stoichiometric_dict = reaction
    elif isinstance(reaction, np.ndarray):
        stoichiometric_dict = arr2dct(reaction, chemicals)
    else:
        raise ValueError(f"reaction must be either a str or an array; not a '{type(reaction).__name__}' object")
    return dct2str(stoichiometric_dict)

def dct2arr(dct, chemicals):
    arr = np.zeros(chemicals.size)
    chemical_index = chemicals._index
    for ID, coefficient in dct.items():
        arr[chemical_index[ID]] = coefficient
    return arr 

def split_coefficient(nID, sign):
    for i, letter in enumerate(nID):
        if letter != 'e' and (letter.isalpha() or letter in '()[]{}'): break
    if i: 
        ID = nID[i:]
        n = sign * float(nID[:i])
    else: 
        ID = nID
        n = sign
    return n, ID

def extract_coefficients(nIDs, dct, sign):
    for nID in nIDs:
        n, ID = split_coefficient(nID, sign)
        if ID in dct:
            raise ValueError('chemicals can only appear once in a reaction; '
                            f'multiple instances of {repr(ID)} found')
        dct[ID] = n

def str2dct(reaction) -> dict:
    reaction = reaction.replace(' ', '')
    left, right = reaction.split('->')
    reactants = left.split('+')
    products = right.split('+')
    dct = {}
    extract_coefficients(reactants, dct, -1)
    extract_coefficients(products, dct, 1)
    return dct

def dct2str(dct):
    if not dct: return "no reaction"
    left = []
    right = []
    for ID, n in dct.items():
        nf = format(n, '.3g')
        if nf == '-1': left.append(ID)
        elif nf == '1': right.append(ID)
        elif n < 0: left.append(f"{-n:.3g} {ID}")
        else: right.append(f"{nf} {ID}")
    left = ' + '.join(left)
    right = ' + '.join(right)
    reaction = left + ' -> ' + right
    return reaction

def arr2dct(arr, chemicals):
    return {ID: n for n, ID in zip(arr, chemicals.IDs) if n}

