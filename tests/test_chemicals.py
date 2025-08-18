# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import thermosteam as tmo
from numpy.testing import assert_allclose

def test_aliases():
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol'], cache=True)
    chemicals.compile()
    chemicals.define_group('alcohols', ['Water', 'Ethanol'])
    assert set(chemicals.get_aliases('H2O')) == {'7732-18-5', 'H2O', 'WATER', 'Water', 'oxidane', 'water'}
    
def test_chemical_cache():
    
    @tmo.utils.chemical_cache
    def create_ethanol_chemicals(other_chemicals):
        chemicals = tmo.Chemicals(['Water', 'Ethanol', *other_chemicals])
        return chemicals
    
    chemicals_a = create_ethanol_chemicals(('CO2', 'O2'))
    chemicals_b = create_ethanol_chemicals(('CO2', 'O2'))
    assert chemicals_a is chemicals_b
    
    chemicals_c = create_ethanol_chemicals(('O2',))
    assert chemicals_a is not chemicals_c 
    assert chemicals_a.Water is chemicals_c.Water
    assert chemicals_b.O2 is chemicals_c.O2
    
    
if __name__ == '__main__':
    test_aliases()
    test_chemical_cache()