# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
from collections.abc import Sized
from .._phase import PhaseIndexer, valid_phases, phase_tuple
from ..base import sparse_array, SparseArray, SparseVector

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
            if isa(j, Sized): phase = j[0]
            else: continue
            phases.append(phase)
    elif isa(reaction, str):
        phases = []
        for i, x in enumerate(reaction):
            if x == ',':
                try: phase = reaction[i+1]
                except: break
                phases.append(phase)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")
    return phase_tuple(phases)

def get_stoichiometric_array(reaction, phases, chemicals):
    """Return stoichiometric array given a string defining the reaction and chemicals."""
    isa = isinstance
    if isa(reaction, dict):
        stoichiometry_dict = reaction
    elif isa(reaction, str):
        stoichiometry_dict = str2dct(reaction)
    elif hasattr(reaction, '__iter__'):
        return sparse_array(reaction, chemicals.size)
    else:
        raise ValueError(f"reaction must be either a str or a dict; not a '{type(reaction).__name__}' object")
    stoichiometric_array = dct2arr(stoichiometry_dict, phases, chemicals)
    return stoichiometric_array

def get_stoichiometric_string(reaction, phases, chemicals):
    """Return a string defining the reaction given the stoichiometric array and chemicals."""
    if isinstance(reaction, dict):
        stoichiometric_dict = reaction
    elif hasattr(reaction, '__iter__'):
        stoichiometric_dict = arr2dct(reaction, phases, chemicals)
    else:
        raise ValueError(f"reaction must be either a str or an array; not a '{type(reaction).__name__}' object")
    return dct2str(stoichiometric_dict)

def dct2arr(dct, phases, chemicals):
    phase_index = PhaseIndexer(phases)
    rows = [{} for i in range(len(phases))]
    chemical_index = chemicals.index
    chemical_groups = chemicals.chemical_groups
    for ID, (phase, coefficient) in dct.items():
        if ID in chemical_groups: 
            raise ValueError(
                f"'{ID}' is a chemical group; chemical groups cannot be used "
                 "in reaction definition"
            )
        rows[phase_index(phase)][chemical_index(ID)] = coefficient
    return SparseArray.from_rows(
        [SparseVector.from_dict(i, chemicals.size) for i in rows]
    ) 

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
                raise ValueError(f'invalid phase {repr(phase)} encountered while parsing reaction')
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
    extract_coefficients(reactants, dct, -1.)
    extract_coefficients(products, dct, 1.)
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
    phase_index = PhaseIndexer(phases)
    IDs = chemicals.IDs
    for phase in phases:
        index = phase_index(phase)
        sv = arr[index]
        svdct = sv.dct
        index = sorted(sv.nonzero_keys())
        for i in index: dct[IDs[i]] = (phase, svdct[i])
    return dct

