# -*- coding: utf-8 -*-
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# Part of the BioSTEAM project. Under the UIUC open-source license.
# See github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
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

def test_chemical(): testmod(tmo._chemical)

def test_chemicals(): testmod(tmo._chemicals)

def test_thermo_data(): testmod(tmo._thermo_data)

def test_stream(): testmod(tmo._stream)

def test_multi_stream(): testmod(tmo._multi_stream)

def test_reaction(): testmod(tmo.reaction._reaction)

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