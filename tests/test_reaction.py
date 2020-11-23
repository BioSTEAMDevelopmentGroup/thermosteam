# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import thermosteam as tmo
from numpy.testing import assert_allclose

def test_reaction():
    # Test corners in code
    tmo.settings.set_thermo(['H2O', 'H2', 'O2'], cache=True)
    reaction = tmo.Reaction('', reactant='H2O', X=1.,
                            correct_atomic_balance=True)
    assert not reaction.stoichiometry.any()
    
    # Test math cycles, making sure they balance out
    reaction = tmo.Reaction('2H2O -> 2H2 + O2', reactant='H2O',
                            correct_atomic_balance=True, X=0.5)
    same_reaction = reaction.copy()
    reaction += same_reaction
    reaction -= same_reaction
    assert_allclose(reaction.X, same_reaction.X)
    reaction *= 2.
    reaction /= 2.
    assert_allclose(reaction.X, same_reaction.X)
    
    # Test negative math
    negative_reaction = 2 * -reaction
    assert_allclose(negative_reaction.X, -1.)
    
    # Test errors with incompatible phases
    reaction = tmo.Reaction('H2O,l -> H2,g + O2,g', reactant='H2O',
                            correct_atomic_balance=True, X=0.7)
    stream = tmo.MultiStream(None, l=[('H2O', 10)], phases='lL')
    with pytest.raises(ValueError): reaction(stream)
    
    # Test errors with incompatible chemicals
    stream = tmo.MultiStream(None, l=[('Water', 10)], 
                             thermo=tmo.Thermo(['Water', 'Ethanol']),
                             phases='gl')
    with pytest.raises(ValueError): reaction(stream)
    
if __name__ == '__main__':
    test_reaction()