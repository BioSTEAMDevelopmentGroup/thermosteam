# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
This module contains all doctests for thermosteam.

"""
import thermosteam as tmo
from doctest import testmod

__all__ = ('test_chemical',
           'test_chemicals',
           'test_thermo_data',
           'test_stream',
           'test_multi_stream',
           'test_reaction',
           'test_equilibrium',
           'test_thermosteam',
)

def test_chemical():
    from thermosteam import _chemical
    testmod(_chemical)

def test_chemicals(): 
    from thermosteam import _chemicals
    testmod(_chemicals)

def test_thermo_data(): 
    from thermosteam import _thermo_data
    testmod(_thermo_data)

def test_stream(): 
    from thermosteam import _thermo_data
    testmod(_thermo_data)

def test_multi_stream():
    from thermosteam import _multi_stream
    testmod(_multi_stream)

def test_reaction(): 
    from thermosteam.reaction import _reaction
    testmod(_reaction)

def test_equilibrium(): 
    testmod(tmo.equilibrium.bubble_point)
    testmod(tmo.equilibrium.dew_point)
    testmod(tmo.equilibrium.vle)
    testmod(tmo.equilibrium.lle)
    
def test_thermosteam():
    test_chemical()
    test_chemicals()
    test_thermo_data()
    test_thermo_data()
    test_stream()
    test_multi_stream()
    test_reaction()
    test_equilibrium()