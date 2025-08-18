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

def test_chemical_ID_parsing():
    H2O = tmo.Chemical('H2O,g')
    assert H2O.ID == 'H2O'
    assert H2O.locked_state == 'g'
    Steam = tmo.Chemical('Steam,g', search_ID='H2O')
    assert Steam.ID == 'Steam'
    assert Steam.locked_state == 'g'
    chemicals = tmo.Chemicals(['H2O', 'Ethanol', 'O2,g', 'CO2,g'])
    assert [i.ID for i in chemicals] == ['H2O', 'Ethanol', 'O2', 'CO2']
    assert [i.locked_state for i in chemicals] == [None, None, 'g', 'g']
    with pytest.raises(ValueError):
        tmo.Chemical('CO2,v')
    