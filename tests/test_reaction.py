# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
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
    
def test_reaction_enthalpy_balance():
    # Combustion; ensure heat of gas phase reaction without sensible heats is 
    # the lower heating value
    chemicals = H2O, Methane, CO2, O2, H2 = tmo.Chemicals(['H2O', 'Methane', 'CO2', 'O2', 'H2'])
    H2O.H.g.Hvap_Tb = 44011.496 # Depending on the model, this value may be different.
    tmo.settings.set_thermo(chemicals)
    combustion = tmo.Reaction('Methane + O2 -> H2O + CO2',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    Tref = 298.15
    Tb = H2O.Tb
    feed = tmo.Stream(Methane=1, O2=2, T=Tb, phase='g')
    H0 = feed.Hnet - Methane.Cn.g.integrate_by_T(Tref, Tb) - 2 * O2.Cn.g.integrate_by_T(Tref, Tb) 
    combustion(feed)
    Hf = feed.Hnet - 2 * H2O.Cn.l.integrate_by_T(Tref, Tb) - CO2.Cn.g.integrate_by_T(Tref, Tb)
    assert_allclose(Hf - H0, Methane.LHV)
    
    # Electrolysis of water; ensure heat of reaction without sensible
    # heats is the higher heating value of hydrogen (with opposite sign)
    tmo.settings.set_thermo(chemicals)
    reaction = tmo.Reaction('2H2O,l -> 2H2,g + O2,g', reactant='H2O', X=1)
    feed = tmo.Stream('feed', H2O=1)
    H0 = feed.Hnet
    feed.phases = ('g', 'l') # Gas and liquid phases must be available
    reaction(feed) # Call to run reaction on molar flow
    Hf = feed.Hnet
    assert_allclose(Hf - H0, -H2.HHV)
    
    # Electrolysis of water; ensure gas phase heat of reaction without sensible
    # heats is the lower heating value of hydrogen (with opposite sign)
    reaction = tmo.Reaction('2H2O -> 2H2 + O2', reactant='H2O', X=1)
    feed = tmo.Stream('feed', H2O=1, T=Tref, phase='g')
    H0 = feed.Hnet - H2O.Cn.l.integrate_by_T(Tref, H2O.Tb) - H2O.Cn.g.integrate_by_T(H2O.Tb, Tref)
    reaction(feed) # Call to run reaction on molar flow
    Hf = feed.Hnet
    assert_allclose(Hf - H0, -H2.LHV)
    
if __name__ == '__main__':
    test_reaction()
    test_reaction_enthalpy_balance()