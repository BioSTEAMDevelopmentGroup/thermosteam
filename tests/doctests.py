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
import doctest
from doctest import testmod

__all__ = ('test_chemical',
           'test_chemicals',
           'test_thermo_data',
           'test_stream',
           'test_multi_stream',
           'test_reaction',
           'test_equilibrium',
)

kwargs = dict(optionflags=doctest.ELLIPSIS ^ doctest.NORMALIZE_WHITESPACE)

def test_chemical():
    from thermosteam import _chemical
    testmod(_chemical, **kwargs)

def test_chemicals(): 
    from thermosteam import _chemicals
    testmod(_chemicals, **kwargs)

def test_thermo_data(): 
    from thermosteam import _thermo_data
    testmod(_thermo_data, **kwargs)

def test_stream(): 
    from thermosteam import _thermo_data
    testmod(_thermo_data, **kwargs)

def test_multi_stream():
    from thermosteam import _multi_stream
    testmod(_multi_stream, **kwargs)

def test_reaction(): 
    from thermosteam.reaction import _reaction
    testmod(_reaction, **kwargs)

def test_equilibrium(): 
    testmod(tmo.equilibrium.bubble_point, **kwargs)
    testmod(tmo.equilibrium.dew_point, **kwargs)
    testmod(tmo.equilibrium.vle, **kwargs)
    testmod(tmo.equilibrium.lle, **kwargs)
    
def test_separations():
    testmod(tmo.separations, **kwargs)
    
if __name__ == '__main__':
    test_chemical()
    test_chemicals()
    test_thermo_data()
    test_stream()
    test_multi_stream()
    test_reaction()
    test_equilibrium()
    test_separations()