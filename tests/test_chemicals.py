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
    assert chemicals.get_aliases('H2O') == ['7732-18-5', 'Water', 'water', 'oxidane', 'H2O']
    
    