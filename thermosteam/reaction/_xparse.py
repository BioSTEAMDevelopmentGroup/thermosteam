# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
import numpy as np
from .._phase import valid_phases

__all__ = ('get_phases',
           'get_stoichiometric_array',
           'get_stoichiometric_string',
           'str2dct', 'dct2str', 'arr2dct')

def get_phases(reaction):
    """Return all available phases in a reaction string."""
    isa = isinstance
    if isa(reaction, dict):
        phases = []
        for i, j in reaction.items():
            if isa(j, str): return None
            phase = j[0]
            if phase not in valid_phases:
                raise ValueError(f'invalid phase {repr(phase)} encountered while parsing reaction')
            phases.append(phase)
    elif isa(reaction, str):
        phases = []
        for i, x in enumerate(reaction):
            try: phase = reaction[i+1]
            except: break
            if x == ',':
                if phase not in valid_phases:
                    raise ValueError(f'invalid phase {repr(phase)} encountered while parsing reaction')
                phases.append(phase)
        return tuple(phases)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")

def get_stoichiometric_array(reaction, phases, chemicals):
    """Return stoichiometric array given a string defining the reaction and chemicals."""
    if isinstance(reaction, dict):
        stoichiometry_dict = reaction
    elif isinstance(reaction, str):
        stoichiometry_dict = str2dct(reaction)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")
    stoichiometric_array = dct2arr(stoichiometry_dict, phases, chemicals)
    return stoichiometric_array

def get_stoichiometric_string(reaction, phases, chemicals):
    """Return a string defining the reaction given the stoichiometric array and chemicals."""
    if isinstance(reaction, dict):
        stoichiometric_dict = reaction
    elif isinstance(reaction, np.ndarray):
        stoichiometric_dict = arr2dct(reaction, phases, chemicals)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")
    return dct2str(stoichiometric_dict)

def dct2arr(dct, phases, chemicals):
    phase_index = {j:i for i,j in enumerate(phases)}
    arr = np.zeros([len(phases), chemicals.size])
    chemical_index = chemicals._index
    for ID, (phase, coefficient) in dct.items():
        arr[phase_index[phase], chemical_index[ID]] = coefficient
    return arr 

def split_coefficient(nID, sign):
    for i, letter in enumerate(nID):
        if letter != 'e' and letter.isalpha(): break
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
        if ID[-2] == ',': 
            phase = ID[-1]
            if phase not in valid_phases:
                raise 
            ID = ID[:-2]
        else:
            raise ValueError('phase must be specified for each chemical; '
                            f'no phase given for {repr(ID)}')
        if ID in dct:
            raise ValueError('chemicals can only appear once in a reaction; '
                            f'multiple instances of {repr(ID)} found')
        dct[ID] = (phase, n)
        
def str2dct(reaction):
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
    for ID, (phase, n) in dct.items():
        ID += ',' + phase
        n_int = int(n)
        if n_int == n: n = n_int
        if n == -1: left.append(ID)
        elif n == 1: right.append(ID)
        elif n < 0: left.append(f"{-n:.3g} {ID}")
        else: right.append(f"{n:.3g} {ID}")
    left = ' + '.join(left)
    right = ' + '.join(right)
    reaction = left + ' -> ' + right
    return reaction

def arr2dct(arr, phases, chemicals):
    dct = {}
    phase_index = {j:i for i,j in enumerate(phases)}
    for phase, index in phase_index.items():
        dct.update({ID: (phase, n) for n, ID in zip(arr[index], chemicals.IDs) if n})
    return dct

