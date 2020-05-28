# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Created on Sat Jun 29 20:47:24 2019

@author: yoelr
"""
from numpy import ndarray

__all__ = ('get_stoichiometric_array',
           'get_stoichiometric_string',
           'str2dct', 'dct2str', 'arr2dct')

def get_stoichiometric_array(reaction, chemicals):
    """Return stoichiometric array given a string defining the reaction and chemicals."""
    if isinstance(reaction, dict):
        stoichiometry_dict = reaction
    elif isinstance(reaction, str):
        stoichiometry_dict = str2dct(reaction)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")
    stoichiometric_array = chemicals.kwarray(stoichiometry_dict)
    return stoichiometric_array

def get_stoichiometric_string(reaction, chemicals):
    """Return a string defining the reaction given the stoichiometric array and chemicals."""
    if isinstance(reaction, dict):
        stoichiometric_dict = reaction
    elif isinstance(reaction, ndarray):
        stoichiometric_dict = arr2dct(reaction, chemicals)
    else:
        raise ValueError(f"reaction must be either a str or a dict, not a '{type(reaction).__name__}' object")
    return dct2str(stoichiometric_dict)

def str2dct(reaction) -> dict:
    reaction = reaction.replace(' ', '')
    left, right = reaction.split('->')
    reactants = left.split('+')
    products = right.split('+')
    dct = {}
    for nID in reactants:
        for i, letter in enumerate(nID):
            if letter == 'e': continue
            if letter.isalpha(): break
        if i: dct[nID[i:]] = -float(nID[:i])
        else: dct[nID] = -1
    for nID in products:
        for i, letter in enumerate(nID):
            if letter == 'e': continue
            if letter.isalpha(): break
        if i: dct[nID[i:]] = float(nID[:i])
        else: dct[nID] = 1
    return dct

def dct2str(dct):
    if not dct: return "no reaction"
    left = []
    right = []
    for ID, N in dct.items():
        N_int = int(N)
        if N_int == N: N = N_int
        if N == -1: left.append(ID)
        elif N == 1: right.append(ID)
        elif N < 0: left.append(f"{-N:.3g} {ID}")
        else: right.append(f"{N:.3g} {ID}")
    left = ' + '.join(left)
    right = ' + '.join(right)
    reaction = left + ' -> ' + right
    return reaction

def arr2dct(arr, chemicals):
    return {ID: N for N, ID in zip(arr, chemicals.IDs) if N}

