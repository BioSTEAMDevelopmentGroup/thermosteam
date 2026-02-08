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
import numpy as np
from thermosteam.reaction import (
    Reaction, ParallelReaction, SeriesReaction, ReactionSystem, ReactionItem
)
from numpy.testing import assert_allclose

def test_reaction():
    # Test corners in code
    tmo.settings.set_thermo(['H2O', 'H2', 'O2'], cache=True)
    reaction = tmo.Reaction('', reactant='H2O', X=1.,
                            correct_atomic_balance=True)
    assert not reaction.stoichiometry.any()
    
    # Test math cycles, making sure they balance out
    single_phase_reaction = reaction = tmo.Reaction(
        '2H2O -> 2H2 + O2', reactant='H2O',
        correct_atomic_balance=True, X=0.5
    )
    same_reaction = reaction.copy()
    reaction += same_reaction
    reaction -= same_reaction
    reaction += 0 # Working with 0 allows for pythonic sum method
    reaction -= 0
    reaction += None # While this might not make much pythonic sense, it was voted by biosteam users for convinience
    reaction -= None
    reaction = reaction + 0 
    reaction = reaction - 0
    reaction = reaction + None 
    reaction = reaction - None
    assert_allclose(reaction.X, same_reaction.X)
    reaction *= 2.
    reaction /= 2.
    assert_allclose(reaction.X, same_reaction.X)
    
    # Test negative math
    negative_reaction = 2 * -reaction
    assert_allclose(negative_reaction.X, -1.)
    
    # Test errors with incompatible phases
    multi_phase_reaction = reaction = tmo.Reaction(
        'H2O,l -> H2,g + O2,g', reactant='H2O',
        correct_atomic_balance=True, X=0.7
    )
    stream = tmo.MultiStream(None, l=[('H2O', 10)], phases='lL')
    with pytest.raises(ValueError): reaction(stream)
    
    # Test errors with incompatible chemicals
    stream = tmo.MultiStream(None, l=[('Water', 10)], 
                             thermo=tmo.Thermo(['Water', 'Ethanol']),
                             phases='gl')
    with pytest.raises(tmo.exceptions.UndefinedChemical): reaction(stream)
    
    # Test errors with chemical groups
    tmo.settings.chemicals.define_group('CriticalGases', ['H2', 'O2'])
    with pytest.raises(ValueError): 
        reaction = tmo.Reaction('H2O -> CriticalGases', reactant='H2O',
                                correct_atomic_balance=True, X=0.7)
        
    with pytest.raises(tmo.exceptions.UndefinedChemical): 
        reaction = tmo.Reaction('H2O -> UnknownChemical', reactant='H2O',
                                correct_atomic_balance=True, X=0.7)
        
    # Test special methods
    
    # Test product_yield with and without multiple phases
    assert_allclose(single_phase_reaction.product_yield('O2'), 0.25)
    assert_allclose(multi_phase_reaction.product_yield('O2'), 0.35)
    assert_allclose(single_phase_reaction.product_yield('H2'), 0.5)
    assert_allclose(multi_phase_reaction.product_yield('H2'), 0.7)
    assert_allclose(single_phase_reaction.product_yield('O2', basis='wt'), 0.44405082796381734)
    assert_allclose(multi_phase_reaction.product_yield('O2', basis='wt'), 0.6216711591493442)
    
    # Test product_yield setter with and without multiple phases
    single_phase_reaction.product_yield('O2', product_yield=0.1)
    multi_phase_reaction.product_yield('O2', product_yield=0.1)
    assert_allclose(single_phase_reaction.product_yield('O2'), 0.1)
    assert_allclose(multi_phase_reaction.product_yield('O2'), 0.1)
    
    single_phase_reaction.product_yield('H2', product_yield=0.1)
    multi_phase_reaction.product_yield('H2', product_yield=0.1)
    assert_allclose(single_phase_reaction.product_yield('H2'), 0.1)
    assert_allclose(multi_phase_reaction.product_yield('H2'), 0.1)
    
    single_phase_reaction.product_yield('O2', basis='wt', product_yield=0.1)
    multi_phase_reaction.product_yield('O2', basis='wt', product_yield=0.1)
    assert_allclose(single_phase_reaction.product_yield('O2', basis='wt'), 0.1)
    assert_allclose(multi_phase_reaction.product_yield('O2', basis='wt'), 0.1)
    
    # Test reactant_demand with and without multiple phases
    single_phase_reaction = reaction = tmo.Reaction(
        'H2 + O2 -> H2O', reactant='H2',
        correct_atomic_balance=True, X=0.5
    )
    multi_phase_reaction = reaction = tmo.Reaction(
        'H2,g + O2,g -> H2O,l', reactant='H2',
        correct_atomic_balance=True, X=0.7
    )
    assert_allclose(single_phase_reaction.reactant_demand('O2'), 0.25)
    assert_allclose(multi_phase_reaction.reactant_demand('O2'), 0.35)
    assert_allclose(single_phase_reaction.reactant_demand('H2'), 0.5)
    assert_allclose(multi_phase_reaction.reactant_demand('H2'), 0.7)
    assert_allclose(single_phase_reaction.reactant_demand('O2', basis='wt'), 3.9683413695259637)
    
    # Test reactant_demand setter with and without multiple phases
    single_phase_reaction.reactant_demand('O2', reactant_demand=0.1)
    multi_phase_reaction.reactant_demand('O2', reactant_demand=0.1)
    assert_allclose(single_phase_reaction.reactant_demand('O2'), 0.1)
    assert_allclose(multi_phase_reaction.reactant_demand('O2'), 0.1)
    
    single_phase_reaction.reactant_demand('H2', reactant_demand=0.1)
    multi_phase_reaction.reactant_demand('H2', reactant_demand=0.1)
    assert_allclose(single_phase_reaction.reactant_demand('H2'), 0.1)
    assert_allclose(multi_phase_reaction.reactant_demand('H2'), 0.1)
    
    single_phase_reaction.reactant_demand('O2', basis='wt', reactant_demand=0.1)
    multi_phase_reaction.reactant_demand('O2', basis='wt', reactant_demand=0.1)
    assert_allclose(single_phase_reaction.reactant_demand('O2', basis='wt'), 0.1)
    assert_allclose(multi_phase_reaction.reactant_demand('O2', basis='wt'), 0.1)
    
    
    
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
    H0 = feed.Hnet - Methane.Cn.g.T_dependent_property_integral(Tref, Tb) - 2 * O2.Cn.g.T_dependent_property_integral(Tref, Tb) 
    combustion(feed)
    Hf = feed.Hnet - 2 * H2O.Cn.l.T_dependent_property_integral(Tref, Tb) - CO2.Cn.g.T_dependent_property_integral(Tref, Tb)
    assert_allclose(Hf - H0, -Methane.LHV)
    
    # Electrolysis of water; ensure heat of reaction without sensible
    # heats is the higher heating value of hydrogen (with opposite sign)
    tmo.settings.set_thermo(chemicals)
    reaction = tmo.Reaction('2H2O,l -> 2H2,g + O2,g', reactant='H2O', X=1)
    feed = tmo.Stream(None, H2O=1)
    H0 = feed.Hnet
    feed.phases = ('g', 'l') # Gas and liquid phases must be available
    reaction(feed) # Call to run reaction on molar flow
    Hf = feed.Hnet
    assert_allclose(Hf - H0, H2.HHV)
    
    # Electrolysis of water; ensure gas phase heat of reaction without sensible
    # heats is the lower heating value of hydrogen (with opposite sign)
    reaction = tmo.Reaction('2H2O -> 2H2 + O2', reactant='H2O', X=1)
    feed = tmo.Stream(None, H2O=1, T=Tref, phase='g')
    H0 = feed.Hnet - H2O.Cn.l.T_dependent_property_integral(Tref, H2O.Tb) - H2O.Cn.g.T_dependent_property_integral(H2O.Tb, Tref)
    reaction(feed) # Call to run reaction on molar flow
    Hf = feed.Hnet
    assert_allclose(Hf - H0, H2.LHV)
    
def test_reaction_enthalpy_with_phases():
    # Ensure liquid reference phase is accounted for
    tmo.settings.set_thermo(['H2O', 'Methane', 'CO2', 'O2', 'H2'], cache=True)
    combustion = tmo.Reaction('Methane,g + O2,g -> H2O,l + CO2,g',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -890590.0)
    
    combustion = tmo.Reaction('Methane,g + O2,g -> H2O,s + CO2,g',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -902610.0)
    
    tmo.settings.set_thermo(['H2O', 'Methane', 'CO2', 'O2', 'H2'])
    combustion = tmo.Reaction('Methane,g + O2,g -> H2O,g + CO2,g',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -802615.10788, rtol=1e-3)
    
    # Ensure gas reference phase is accounted for
    combustion = tmo.Reaction('Methane,g + O2,g -> H2O,l + CO2,l',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -895855.5436243636, rtol=1e-3)
    
    combustion = tmo.Reaction('Methane,g + O2,g -> H2O,s + CO2,s',
                              reactant='Methane', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -916890.4976790915, rtol=1e-3)
    
    # Ensure solid reference phase is accounted for
    tmo.settings.set_thermo(['H2O', 'Glucose', 'CO2', 'O2', 'H2'], cache=True)
    combustion = tmo.Reaction('Glucose,s + O2,g -> H2O,g + CO2,g',
                              reactant='Glucose', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -2541480.7756171282, rtol=1e-3)
    
    tmo.settings.set_thermo(['H2O', 'Glucose', 'CO2', 'O2', 'H2'], cache=True)
    combustion = tmo.Reaction('Glucose,l + O2,g -> H2O,g + CO2,g',
                              reactant='Glucose', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -2561413.7756171282, rtol=1e-3)
    
    combustion = tmo.Reaction('Glucose,g + O2,g -> H2O,g + CO2,g',
                              reactant='Glucose', X=1,
                              correct_atomic_balance=True)
    assert_allclose(combustion.dH, -2787650.3239119546, rtol=1e-3)

def test_reactive_phase_equilibrium_no_kinetics():
    import thermosteam as tmo
    from numpy.testing import assert_allclose
    tmo.settings.set_thermo(['EthylLactate', 'LacticAcid', 'H2O', 'Ethanol'], cache=True)
    rxn = tmo.Reaction('LacticAcid + Ethanol -> H2O + EthylLactate', reactant='LacticAcid', X=0.2)
    stream = tmo.Stream(
        H2O=1, Ethanol=5, LacticAcid=1
    )
    mol_original = stream.mol.copy()
    F_mass = stream.F_mass
    stream.vle(T=360, P=101325, liquid_conversion=rxn)
    liq_new = stream.imol['l'].copy()
    dliq = rxn.conversion(liq_new)
    assert_allclose(stream.F_mass, F_mass)
    assert_allclose(stream.mol, mol_original + dliq, rtol=1e-6)
    assert_allclose(
        stream.imol['g'], 
        [0.0074480755817397755,
         0.0013627607389679424,
         0.6659671383174538,
         2.8437100641121864],
        atol=1e-6,
        rtol=1e-6,
    )
    assert_allclose(
        stream.imol['l'], 
        [0.15899149544493343, 
         0.8321976682343589, 
         0.5004724327092195, 
         1.98985036486114],
        atol=1e-6,
        rtol=1e-6,
    )
    
    stream.vle(T=340, P=101325, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['l'], 
        stream.mol
    )
    assert_allclose(
        stream.imol['g'], 
        0
    )
    stream.vle(T=450, P=101325, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['g'], 
        stream.mol
    )
    assert_allclose(
        stream.imol['l'], 
        0
    )
    
# TODO: Fix kinetics!
def test_reactive_phase_equilibrium_with_kinetics():
    import thermosteam as tmo
    from math import exp
    from numpy.testing import assert_allclose
    tmo.settings.set_thermo(['EthylLactate', 'LacticAcid', 'H2O', 'Ethanol'], cache=True)
    
    def rate(stream):
        T = stream.T
        if T > 370: return 0 # Prevents multiple steady states.
        R = tmo.constants.R
        kf = 6.52e3 * exp(-4.8e4 / (R * T))
        kr = 2.72e3 * exp(-4.8e4 / (R * T))
        LaEt, La, H2O, EtOH = stream.mol / stream.F_mol
        catalyst = 0.001 # Kg of catalyst
        return catalyst * 3600 * (kf * La * EtOH - kr * LaEt * H2O) # kmol / kg-catalyst / hr
    
    rxn = tmo.Reaction('LacticAcid + Ethanol -> H2O + EthylLactate', reactant='LacticAcid', rate=rate)
    stream = tmo.Stream(
        H2O=2, Ethanol=5, LacticAcid=1, T=355,
    )
    mol_original = stream.mol.copy()
    F_mass = stream.F_mass
    stream.vle(T=360, P=101325, liquid_conversion=rxn)
    liq_new = tmo.Stream(flow=stream.imol['l'].copy(), T=360, P=101325)
    dliq = rxn.conversion(liq_new)
    assert_allclose(stream.F_mass, F_mass)
    assert_allclose(stream.mol, mol_original + dliq, atol=1e-2, rtol=1e-6)
    rxn(stream)
    assert_allclose(
        stream.mol,
        [0.0027372232703874026,
         0.9972627767296126,
         2.0027372232703873,
         4.997262776729612],
        atol=1e-3,
        rtol=1e-3,
    )
    stream = tmo.Stream(
        H2O=2, Ethanol=5, LacticAcid=1, T=355,
    )
    T = 360
    P = 101325
    stream.vle(T=T, P=P, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['l'],
        [0.0025656299861595427,
         0.9953743460341485,
         0.8534956958780036,
         1.814963260010252],
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        stream.imol['g'],
        [0.0001715932842278598,
         0.0018884306954640294,
         1.1492415273923837,
         3.18229951671936],
        rtol=1e-3,
        atol=1e-3,
    )
    V = stream.vapor_fraction
    H = stream.H + stream.Hf
    stream = tmo.Stream(
        H2O=2, Ethanol=5, LacticAcid=1, T=T,
    )
    stream.vle(V=V, P=P, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['l'],
        [0.002565629805882328,
         0.9953743462470112,
         0.853495772294828,
         1.8149631824466637],
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        stream.imol['g'],
        [0.0001715933162454465,
         0.0018884306308610292,
         1.1492414508272997,
         3.1822995944312087],
        rtol=1e-3,
        atol=1e-3,
    )
    stream = tmo.Stream(
        H2O=2, Ethanol=5, LacticAcid=1, T=T,
    )
    stream.vle(V=V, T=T, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['l'],
        [0.0025656303993929954,
         0.9953743433287133,
         0.853495226089156,
         1.814961876362787],
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        stream.imol['g'],
        [0.0001715935470026392,
         0.0018884327248910547,
         1.1492419978572395,
         3.1823008996908175],
        rtol=1e-3,
        atol=1e-3,
    )
    stream = tmo.Stream(
        H2O=2, Ethanol=5, LacticAcid=1, T=T,
    )
    stream.vle(H=H, P=P, liquid_conversion=rxn)
    assert_allclose(
        stream.imol['l'],
        [0.0025656298107399112,
         0.995374346155454,
         0.853495769770851,
         1.814963176478684],
        rtol=1e-3,
        atol=1e-3,
    )
    assert_allclose(
        stream.imol['g'],
        [0.00017159331767585686,
         0.0018884307161302612,
         1.149241453357565,
         3.1822996003929],
        rtol=1e-3,
        atol=1e-3,
    )
    
    
def test_repr():
    cal2joule = 4.184
    Glucan = tmo.Chemical('Glucan', search_db=False, formula='C6H10O5', Hf=-233200*cal2joule, phase='s', default=True)
    Glucose = tmo.Chemical('Glucose', phase='s')
    CO2 = tmo.Chemical('CO2', phase='g')
    HMF = tmo.Chemical('HMF', search_ID='Hydroxymethylfurfural', phase='l', default=True)
    Biomass = Glucose.copy(ID='Biomass')
    tmo.settings.set_thermo(['Water', 'Ethanol', 'LacticAcid', HMF, Glucose, Glucan, CO2, Biomass])
    saccharification = tmo.PRxn([
        tmo.Rxn('Glucan + H2O -> Glucose', reactant='Glucan', X=0.9),
        tmo.Rxn('Glucan -> HMF + 2H2O', reactant='Glucan', X=0.025)
    ])
    fermentation = tmo.SRxn([
        tmo.Rxn('Glucose -> 2LacticAcid', reactant='Glucose', X=0.03),
        tmo.Rxn('Glucose -> 2Ethanol + 2CO2', reactant='Glucose', X=0.95),
    ])
    cell_growth = tmo.Rxn('Glucose -> Biomass', reactant='Glucose', X=1.0)
    cellulosic_rxnsys = tmo.RxnSys(saccharification, fermentation, cell_growth)
    saccharification = eval(repr(saccharification))
    fermentation = eval(repr(fermentation))
    cell_growth = eval(repr(cell_growth))
    cellulosic_rxnsys = eval(repr(cellulosic_rxnsys))
    
    
if __name__ == '__main__':
    test_reaction()
    test_reaction_enthalpy_balance()
    test_reaction_enthalpy_with_phases()
    test_reactive_phase_equilibrium_no_kinetics()
    test_reactive_phase_equilibrium_with_kinetics()
    test_repr()