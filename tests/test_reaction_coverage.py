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
from numpy.testing import assert_allclose
from thermosteam.base import SparseVector, SparseArray
from thermosteam.reaction import (
    Reaction, ParallelReaction, SeriesReaction, ReactionSystem, ReactionItem,
    ReactionSet,
)
from thermosteam.reaction import _parse as prs, _xparse as xprs
from thermosteam.chemicals import reaction as chem_rxn

def set_thermo():
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Glucose', 'O2', 'CO2', 'H2'], cache=True)
    tmo.settings.set_thermo(chemicals)
    return chemicals

def fermentation(X=0.5, **kwargs):
    return Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', X, **kwargs)

def test_reaction_basics():
    chemicals = set_thermo()
    rxn = fermentation()
    assert rxn.basis == 'mol'
    assert rxn.reactant == 'Glucose'
    assert rxn.all_reactants == ['Glucose']
    assert rxn.reaction_indices() == {1, 2, 4} or len(rxn.reaction_indices()) == 3
    assert rxn.has_reaction()
    assert abs(rxn.mass_balance_error()) < 1e-2
    assert all(abs(i) < 1e-6 for i in rxn.atomic_balance_error().values())
    rxn.check_mass_balance()
    rxn.check_atomic_balance()
    assert rxn.X_net() == {'Glucose': 0.5}
    ci = rxn.X_net(indexer=True)
    assert_allclose(ci['Glucose'], 0.5)
    assert rxn.istoichiometry['Ethanol'] == 2
    assert np.isfinite(rxn.dH) and rxn.dH < 0
    assert 'Glucose' in repr(rxn)
    rxn.show()
    df = rxn.to_df()
    assert df.shape[0] == 1
    df = rxn.to_df(index='R1')
    assert list(df.index) == ['R1']
    # Default reactant inferred from single reactant
    rxn2 = Reaction('Glucose -> 2Ethanol + 2CO2')
    assert rxn2.reactant == 'Glucose' and rxn2.X == 1.
    with pytest.raises(ValueError):
        Reaction('Glucose + O2 -> CO2 + Water')
    with pytest.raises(ValueError):
        Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5, basis='kg')
    with pytest.raises(RuntimeError):
        Reaction('Glucose -> 2Ethanol + 2CO2', 'Water', 0.5)
    # Balance checks at construction
    Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5,
             check_mass_balance=True, check_atomic_balance=True)
    rxn3 = Reaction('Glucose -> 2Ethanol + CO2', 'Glucose', 0.5,
                    correct_mass_balance=True)
    assert abs(rxn3.mass_balance_error()) < 1e-2
    with pytest.raises(RuntimeError):
        rxn3 = Reaction('Glucose -> Ethanol', 'Glucose', 0.5)
        rxn3.check_mass_balance()
    with pytest.raises(RuntimeError):
        rxn3.check_atomic_balance()
    # Reaction from dict and from array
    rxn4 = Reaction({'Glucose': -1, 'Ethanol': 2, 'CO2': 2}, 'Glucose', 0.5)
    assert_allclose(rxn4.stoichiometry.to_array(), rxn.stoichiometry.to_array())
    rxn5 = Reaction(rxn.stoichiometry.to_array(), 'Glucose', 0.5)
    assert_allclose(rxn5.stoichiometry.to_array(), rxn.stoichiometry.to_array())

def test_reaction_on_streams_and_arrays():
    chemicals = set_thermo()
    rxn = fermentation()
    s = tmo.Stream(None, Glucose=10, Water=100)
    mass = s.F_mass
    rxn(s)
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    assert_allclose(s.imol['Ethanol'], 10, rtol=1e-2)
    assert_allclose(s.F_mass, mass, rtol=1e-3)
    # By weight
    rxn_wt = rxn.copy(basis='wt')
    assert rxn_wt.basis == 'wt'
    assert abs(rxn_wt.mass_balance_error()) < 1e-2
    assert all(abs(i) < 1e-6 for i in rxn_wt.atomic_balance_error().values())
    assert np.isfinite(rxn_wt.dH)
    s = tmo.Stream(None, Glucose=10, Water=100)
    rxn_wt(s)
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    assert_allclose(s.F_mass, mass, rtol=1e-3)
    rxn_wt.basis = 'mol'
    assert rxn_wt.basis == 'mol'
    with pytest.raises(ValueError):
        rxn_wt.basis = 'kg'
    # Arrays
    arr = np.zeros(chemicals.size); arr[chemicals.index('Glucose')] = 10
    rxn(arr)
    assert_allclose(arr.sum(), 5 + 10 + 10, rtol=1e-2)
    sv = SparseVector.from_size(chemicals.size); sv[chemicals.index('Glucose')] = 10
    rxn(sv)
    assert_allclose(sv[chemicals.index('Glucose')], 5, rtol=1e-2)
    # Stream with different chemicals
    other = tmo.Stream(None, Glucose=10, thermo=tmo.Thermo(['Glucose', 'Ethanol', 'CO2', 'Water'], cache=True))
    rxn(other)
    assert_allclose(other.imol['Glucose'], 5, rtol=1e-2)
    assert other.chemicals.IDs == ('Glucose', 'Ethanol', 'CO2', 'Water')
    # Conversion
    s = tmo.Stream(None, Glucose=10)
    c = rxn.conversion(s)
    assert_allclose(c[chemicals.index('Ethanol')], 10, rtol=1e-2)
    other = tmo.Stream(None, Glucose=10, thermo=tmo.Thermo(['Glucose', 'Ethanol', 'CO2'], cache=True))
    c = rxn.conversion(other)
    assert_allclose(c[chemicals.index('Ethanol')], 10, rtol=1e-2)
    # Indexer input
    s = tmo.Stream(None, Glucose=10)
    rxn(s.imol)
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    # Sparse array inputs
    s = tmo.Stream(None, Glucose=10)
    rxn(s.imass.data) # Mass flows are reacted as an array
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    # Force reaction (no feasibility check)
    s = tmo.Stream(None, Glucose=10)
    rxn.force_reaction(s)
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    s = tmo.Stream(None, Glucose=10)
    rxn_wt = rxn.copy(basis='wt')
    rxn_wt.force_reaction(s)
    assert_allclose(s.imol['Glucose'], 5, rtol=1e-2)
    # Adiabatic reaction
    s = tmo.Stream(None, Glucose=1, Water=100, T=300)
    rxn.adiabatic_reaction(s)
    assert np.isfinite(s.T) and s.T > 250
    with pytest.raises(ValueError):
        rxn.adiabatic_reaction(arr)

def test_reaction_feasibility():
    chemicals = set_thermo()
    rxn = Reaction('Glucose + 6O2 -> 6CO2 + 6Water', 'Glucose', 1.)
    s = tmo.Stream(None, Glucose=10, O2=1)
    with pytest.raises(tmo.exceptions.InfeasibleRegion):
        rxn(s)
    # Very small negative values are set to zero
    s = tmo.Stream(None, Glucose=1e-15, O2=0)
    rxn(s)
    assert (s.mol >= 0).all()
    tmo.reaction.CHECK_FEASIBILITY = False
    try:
        s = tmo.Stream(None, Glucose=10, O2=60)
        rxn(s)
        assert_allclose(s.imol['CO2'], 60, rtol=1e-2)
    finally:
        tmo.reaction.CHECK_FEASIBILITY = True

def test_reaction_math():
    chemicals = set_thermo()
    a = fermentation(0.3)
    b = fermentation(0.2)
    c = a + b
    assert_allclose(c.X, 0.5, rtol=1e-2)
    c = a - b
    assert_allclose(c.X, 0.1, rtol=1e-2)
    assert (a - 0) is a
    d = a.copy()
    d -= b
    assert np.isfinite(d.X)
    assert sum([a, b]).X > 0
    d = a / 2
    assert_allclose(d.X, 0.15, rtol=1e-2)
    d /= 3
    d *= 3
    assert_allclose(d.X, 0.15, rtol=1e-2)
    assert_allclose((-a).X, -0.3)
    assert_allclose((2 * a).X, 0.6)
    # Incompatible math
    other = Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5,
                     chemicals=tmo.Chemicals(['Glucose', 'Ethanol', 'CO2'], cache=True))
    with pytest.raises(ValueError): a + other
    phased = Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5)
    with pytest.raises(ValueError): a + phased
    h2 = Reaction('2Water -> 2H2 + O2', 'Water', 0.5)
    with pytest.raises(ValueError): h2 + Reaction('2H2 + O2 -> 2Water', 'H2', 0.5) # Different reactants

def test_reaction_properties_and_setters():
    chemicals = set_thermo()
    rxn = fermentation()
    # Product yield and reactant demand
    assert_allclose(rxn.product_yield('Ethanol'), 1., rtol=1e-2)
    y_wt = rxn.product_yield('Ethanol', basis='wt')
    assert 0 < y_wt < 1
    rxn.product_yield('Ethanol', basis='wt', product_yield=y_wt)
    assert_allclose(rxn.X, 0.5, rtol=1e-2)
    rxn.product_yield('Ethanol', product_yield=1.6)
    assert_allclose(rxn.X, 0.8, rtol=1e-2)
    assert_allclose(rxn.reactant_demand('Glucose'), 0.8, rtol=1e-2)
    rxn.reactant_demand('Glucose', reactant_demand=0.4)
    assert_allclose(rxn.X, 0.4, rtol=1e-2)
    with pytest.raises(ValueError):
        rxn.product_yield('Ethanol', product_yield=3)
    with pytest.raises(ValueError):
        rxn.product_yield('Ethanol', basis='kg')
    with pytest.raises(ValueError):
        rxn.product_yield('Ethanol', basis='kg', product_yield=0.1)
    rxn_wt = rxn.copy(basis='wt')
    y_mol = rxn_wt.product_yield('Ethanol', basis='mol')
    assert_allclose(y_mol, 0.8, rtol=1e-2)
    rxn_wt.product_yield('Ethanol', basis='mol', product_yield=1.)
    assert_allclose(rxn_wt.X, 0.5, rtol=1e-2)
    # Rate setter
    with pytest.raises(ValueError):
        rxn.rate = 1
    # Reactant setter
    comb = Reaction('Glucose + 6O2 -> 6CO2 + 6Water', 'Glucose', 0.5)
    comb.reactant = 'O2'
    assert comb.reactant == 'O2'
    assert_allclose(comb.stoichiometry[chemicals.index('Glucose')], -1/6, rtol=1e-2)
    # Backwards
    back = fermentation().backwards('Ethanol')
    assert back.reactant == 'Ethanol' and back.X == 1
    h2 = Reaction('2Water -> 2H2 + O2', 'Water', 0.5)
    with pytest.raises(ValueError): h2.backwards()
    rxn = Reaction('Glucose -> 2Ethanol', 'Glucose', 0.5)
    back = rxn.backwards()
    assert np.isfinite(back.X)
    # Reset chemicals
    rxn = fermentation()
    new_chemicals = tmo.CompiledChemicals(['CO2', 'Ethanol', 'Glucose'], cache=True)
    rxn.reset_chemicals(new_chemicals)
    rxn.reset_chemicals(new_chemicals)
    assert rxn.chemicals is new_chemicals
    assert_allclose(rxn.stoichiometry[0], 2)
    # No reaction
    empty = Reaction(None, 'Glucose', 0.5)
    assert not empty.has_reaction()
    # Atomic balance with constants and underspecified reactions
    rxn = Reaction('Glucose + O2 -> CO2 + Water', 'Glucose', 0.5)
    rxn.correct_atomic_balance('Glucose')
    assert_allclose(rxn.stoichiometry[chemicals.index('O2')], -6, rtol=1e-2)
    rxn = Reaction('Glucose + O2 -> CO2 + Water + Ethanol', 'Glucose', 0.5)
    with pytest.raises(RuntimeError): rxn.correct_atomic_balance()
    # Correct mass balance with variable
    rxn = Reaction('Glucose -> 2Ethanol + CO2', 'Glucose', 0.5)
    rxn.correct_mass_balance('CO2')
    assert abs(rxn.mass_balance_error()) < 0.1 * 180

def test_phased_reaction():
    chemicals = set_thermo()
    rxn = Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5)
    assert rxn.reactant == ('l', 'Glucose')
    rxn2 = Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g')
    assert rxn2.reactant == ('l', 'Glucose')
    with pytest.raises(ValueError):
        Reaction('Glucose,l + O2,g -> 2Ethanol,l + 2CO2,g')
    assert rxn.istoichiometry['g', 'CO2'] == 2
    assert np.isfinite(rxn.dH)
    rxn.show()
    assert 'Glucose,l' in repr(rxn)
    assert rxn.to_df().shape[0] == 1
    assert_allclose(rxn.product_yield('Ethanol'), 1., rtol=1e-2)
    s = tmo.Stream(None, Glucose=10, Water=10)
    s.phases = ('g', 'l')
    rxn(s)
    assert_allclose(s.imol['g', 'CO2'], 10, rtol=1e-2)
    s = tmo.Stream(None, Glucose=10)
    with pytest.raises(ValueError): rxn(s) # phases do not match
    ms = tmo.Stream(None, Glucose=10)
    ms.phases = ('s', 'l')
    with pytest.raises(ValueError): rxn(ms.imol) # phases do not match
    arr = np.zeros([2, chemicals.size]); arr[1, chemicals.index('Glucose')] = 10
    rxn(arr)
    assert_allclose(arr[0, chemicals.index('CO2')], 10, rtol=1e-2)
    arr = SparseArray.from_shape([2, chemicals.size])
    arr[1, chemicals.index('Glucose')] = 10
    rxn(arr)
    ms = tmo.Stream(None, Glucose=10, Water=10)
    ms.phases = ('g', 'l')
    rxn(ms.imass.data)
    assert_allclose(ms.imol['l', 'Glucose'], 5, rtol=1e-2)
    # Reactant setter / reset chemicals / correct balance
    rxn.reactant = 'Ethanol'
    assert rxn.reactant == ('l', 'Ethanol')
    new_chemicals = tmo.CompiledChemicals(['CO2', 'Ethanol', 'Glucose'], cache=True)
    rxn.reset_chemicals(new_chemicals)
    assert rxn.chemicals is new_chemicals
    set_thermo()
    rxn = Reaction('Glucose,l -> 2Ethanol,l + CO2,g', 'Glucose', 0.5,
                   correct_atomic_balance=True)
    assert_allclose(rxn.stoichiometry[0, chemicals.index('CO2')], 2, rtol=1e-2)
    # Backwards
    with pytest.raises(ValueError):
        Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5).backwards()
    # Latent heats in all phase combinations
    rxn = Reaction({'Water': ('s', -1), 'O2': ('l', -1), 'Ethanol': ('g', 1), 'H2': ('s', 1)},
                   'Water', 0.5, phases='gls')
    assert np.isfinite(rxn.dH)
    rxn = Reaction({'O2': ('s', -1), 'Water': ('g', 1)}, 'O2', 0.5, phases='gls')
    assert np.isfinite(rxn.dH)
    rxn = Reaction({'Glucose': ('l', -1), 'Ethanol': ('s', 1)}, 'Glucose', 0.5, phases='gls')
    assert np.isfinite(rxn.dH)
    rxn = Reaction({'Glucose': ('g', -1), 'Ethanol': ('l', 1)}, 'Glucose', 0.5, phases='gls')
    dH = rxn.dH
    for stoichiometry in ({'Water': ('L', -1)}, {'O2': ('L', -1)}, {'Glucose': ('L', -1)}):
        rxn = Reaction(stoichiometry, None, 0.5, phases='lL')
        with pytest.raises(RuntimeError): rxn.dH

def test_rate_and_equilibrium():
    chemicals = set_thermo()
    s = tmo.Stream(None, Glucose=10)
    rate = lambda stream: 0.1 * stream.imol['Glucose']
    rxn = Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', rate=rate)
    assert rxn.rate is rate
    assert np.isnan(rxn.X)
    rxn.show()
    assert 'X=' not in repr(rxn)
    assert rxn.to_df().shape[1] == 2
    rxn(s)
    assert_allclose(s.imol['Glucose'], 9, rtol=1e-2)
    handle = rxn.conversion_handle(s)
    s = tmo.Stream(None, Glucose=10)
    c = handle(s.mol)
    assert np.isfinite(c.to_array()).all()
    handle.set_chemicals([chemicals['Glucose'], chemicals['Ethanol']])
    c = handle(np.array([10., 0.]))
    assert np.isfinite(c).all()
    rxn_X = fermentation()
    handle = rxn_X.conversion_handle(s)
    c = handle(s.mol)
    assert_allclose(c[chemicals.index('Ethanol')], 10, rtol=1e-2)
    # Equilibrium: Glucose = Ethanol at equilibrium
    eq_rate = lambda stream: stream.imol['Glucose'] - stream.imol['Ethanol']
    rxn = Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', rate=eq_rate)
    s = tmo.Stream(None, Glucose=10)
    c = rxn.equilibrium_conversion(s)
    assert_allclose(s.imol['Glucose'], 10)
    rxn.equilibrium(s)
    assert_allclose(s.imol['Glucose'], s.imol['Ethanol'], rtol=1e-2, atol=1e-6)
    assert_allclose(s.imol['Glucose'], 20/3, rtol=2e-2)

def test_reaction_sets():
    chemicals = set_thermo()
    kwargs = dict(correct_atomic_balance=True)
    prxn = ParallelReaction([
        Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5),
        Reaction('Glucose + O2 -> CO2 + Water', 'Glucose', 0.2, **kwargs),
    ])
    srxn = SeriesReaction([
        Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5),
        Reaction('Ethanol + O2 -> CO2 + Water', 'Ethanol', 0.2, **kwargs),
    ])
    for rxn in (prxn, srxn):
        assert rxn.basis == 'mol'
        with pytest.raises(TypeError): rxn.basis = 'wt'
        assert len(rxn.X) == 2
        rxn.X = rxn.X
        rxn.X = [0.5, 0.2]
        assert rxn.rate == [None, None]
        assert len(rxn.stoichiometry) == 2
        assert rxn.MWs.shape == (1, chemicals.size)
        assert 'Glucose' in rxn.all_reactants
        assert len(rxn.reaction_indices()) >= 4
        rxn.show()
        assert 'Reaction' in repr(rxn)
        assert rxn.to_df().shape[0] == 2
        assert rxn.to_df(index=['a', 'b']).shape[0] == 2
        s = tmo.Stream(None, Glucose=10, O2=100)
        mass = s.F_mass
        rxn(s)
        assert_allclose(s.F_mass, mass, rtol=1e-3)
        assert (s.mol >= 0).all()
        s = tmo.Stream(None, Glucose=10, O2=100)
        c = rxn.conversion(s)
        assert np.isfinite(c.to_array()).all()
        s = tmo.Stream(None, Glucose=10, O2=100)
        rxn.force_reaction(s)
        s = tmo.Stream(None, Glucose=1, O2=10, Water=100)
        rxn.adiabatic_reaction(s)
        assert np.isfinite(s.T)
        X_net = rxn.X_net()
        assert all(0 <= i <= 1 for i in X_net.values())
        assert rxn.X_net(indexer=True)['Glucose'] > 0
        item = rxn[0]
        assert isinstance(item, ReactionItem)
        assert item.basis == 'mol'
        with pytest.raises(TypeError): item.basis = 'wt'
        item.X = 0.4
        assert_allclose(rxn.X[0], 0.4)
        assert item.rate is None
        item.rate = None
        copy = item.copy()
        assert type(copy) is Reaction
        copy = item.copy(basis='wt')
        assert copy.basis == 'wt'
        sub = rxn[:1]
        assert len(sub.X) == 1
        assert len(list(rxn)) == 2
        with pytest.raises(NotImplementedError): rxn.equilibrium(s, 300, 101325, 'l')
        rxn._rescale()
    assert len(prxn.reduce().X) == 1
    with pytest.raises(TypeError): srxn.reduce()
    # Repeated reactants in X_net
    srxn2 = SeriesReaction([fermentation(0.5), fermentation(0.5)])
    assert_allclose(srxn2.X_net()['Glucose'], 0.75, rtol=1e-2)
    prxn2 = prxn + prxn
    assert_allclose(prxn2.X, 2 * prxn.X, rtol=1e-2)
    assert_allclose(prxn.X_net()['Glucose'], prxn.X.sum(), rtol=1e-2)
    # Item reactant setter
    item = srxn[1]
    item.reactant = 'O2'
    assert srxn.reactants[1] == 'O2'
    # Basis argument
    wt = ParallelReaction([fermentation(0.5), fermentation(0.2)], basis='wt')
    assert wt.basis == 'wt'
    with pytest.raises(ValueError): ParallelReaction([fermentation(), fermentation()], basis='kg')
    with pytest.raises(ValueError): ParallelReaction([fermentation(), fermentation(basis='wt')])
    with pytest.raises(ValueError): ParallelReaction([])
    with pytest.raises(ValueError):
        ParallelReaction([fermentation(), Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5)])
    other = Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5,
                     chemicals=tmo.Chemicals(['Glucose', 'Ethanol', 'CO2'], cache=True))
    with pytest.raises(ValueError): ParallelReaction([fermentation(), other])
    # Backwards of a set with one reactant
    back = ParallelReaction([fermentation(0.5), fermentation(0.2)]).backwards('Ethanol')
    assert back.reactant == 'Ethanol'
    with pytest.raises(RuntimeError): srxn.backwards('CO2') # Multiple reactants

def test_reaction_sets_with_rates_and_phases():
    chemicals = set_thermo()
    rate = lambda stream: 1.
    for cls in (ParallelReaction, SeriesReaction):
        rxn = cls([
            Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', rate=rate),
            fermentation(0.1),
        ])
        s = tmo.Stream(None, Glucose=10)
        rxn(s)
        assert 7.9 < s.imol['Glucose'] < 8.2
        s = tmo.Stream(None, Glucose=10)
        c = rxn.conversion(s)
        assert -2.1 < c[chemicals.index('Glucose')] < -1.8
    kwargs = dict(phases='lg')
    prxn = ParallelReaction([
        Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5, **kwargs),
        Reaction({'Glucose': ('l', -1), 'H2': ('g', 6), 'CO2': ('g', 6), 'Water': ('l', -6)}, 'Glucose', 0.1, **kwargs),
    ])
    assert prxn.reactants == (('l', 'Glucose'), ('l', 'Glucose'))
    prxn.show()
    assert prxn.to_df().shape[0] == 2
    mi = prxn.X_net(indexer=True)
    assert_allclose(mi['l', 'Glucose'], 0.6, rtol=1e-2)
    assert prxn[0].reactant == ('l', 'Glucose')

def test_reset_chemicals_of_sets():
    chemicals = set_thermo()
    new_chemicals = tmo.CompiledChemicals(['CO2', 'Ethanol', 'Glucose', 'O2', 'Water'], cache=True)
    prxn = ParallelReaction([
        Reaction('Glucose,l -> 2Ethanol,l + 2CO2,g', 'Glucose', 0.5),
        Reaction('Glucose,l + 6O2,g -> 6CO2,g + 6Water,l', 'Glucose', 0.2),
    ])
    item = prxn[0]
    item.reset_chemicals(new_chemicals)
    assert prxn.chemicals is new_chemicals
    item.reset_chemicals(new_chemicals)
    sub = prxn[:1]
    assert sub.chemicals is new_chemicals
    sub.reset_chemicals(new_chemicals)

def test_reaction_system():
    chemicals = set_thermo()
    kwargs = dict(correct_atomic_balance=True)
    prxn = ParallelReaction([
        fermentation(0.5),
        Reaction('Glucose + O2 -> CO2 + Water', 'Glucose', 0.2, **kwargs),
    ])
    srxn = SeriesReaction([
        Reaction('Ethanol + O2 -> CO2 + Water', 'Ethanol', 0.2, **kwargs),
        Reaction('2H2 + O2 -> 2Water', 'H2', 0.5),
    ])
    rxn = Reaction('2Water -> 2H2 + O2', 'Water', 0.1)
    rxnsys = ReactionSystem(prxn, srxn, rxn)
    assert rxnsys.reactions == (prxn, srxn, rxn)
    assert rxnsys[2] is rxn
    assert list(rxnsys) == [prxn, srxn, rxn]
    assert len(rxnsys.X) == 3
    rxnsys.X = rxnsys.X
    rxnsys.show()
    assert 'ReactionSystem' in repr(rxnsys)
    assert_allclose(ReactionSystem(fermentation(), fermentation()).X_net()['Glucose'], 0.75)
    X_net = rxnsys.X_net()
    assert all(np.isfinite(i) for i in X_net.values())
    assert rxnsys.X_net(indexer=True)['Glucose'] > 0
    s = tmo.Stream(None, Glucose=10, O2=100, Water=100, H2=1)
    assert 0 < rxnsys.reactant_flux(s, 0) <= 10
    assert 0 < rxnsys.reactant_flux(s, 0, 1) <= 10
    assert 0 <= rxnsys.reactant_flux(s, 1, 1) <= 10
    assert 0 < rxnsys.reactant_flux(s, 2) <= 100
    with pytest.raises(ValueError): rxnsys.reactant_flux(s, 1)
    mass = s.F_mass
    c = rxnsys._conversion(s.mol.copy())
    assert np.isfinite(c.to_array()).all()
    rxnsys(s)
    assert_allclose(s.F_mass, mass, rtol=1e-3)
    assert (s.mol >= 0).all()
    s = tmo.Stream(None, Glucose=1, O2=10, Water=100)
    rxnsys.adiabatic_reaction(s)
    assert np.isfinite(s.T)
    with pytest.raises(NotImplementedError): rxnsys.equilibrium(s, 300, 101325, 'l')
    simple = ReactionSystem(fermentation(), Reaction('2Water -> 2H2 + O2', 'Water', 0.1))
    assert simple.reactants == ['Glucose', 'Water']
    simple.reactants = ['Glucose', 'Water']
    assert simple.all_reactants == ['Water', 'Glucose']
    wt = ReactionSystem(fermentation(), fermentation(), basis='wt')
    assert wt._basis == 'wt'
    s = tmo.Stream(None, Glucose=10)
    wt(s)
    assert_allclose(s.imol['Glucose'], 2.5, rtol=1e-2)
    with pytest.raises(ValueError): ReactionSystem()
    with pytest.raises(ValueError): ReactionSystem(fermentation(), basis='kg')
    with pytest.raises(ValueError): ReactionSystem(fermentation(), fermentation(basis='wt'))
    other = Reaction('Glucose -> 2Ethanol + 2CO2', 'Glucose', 0.5,
                     chemicals=tmo.Chemicals(['Glucose', 'Ethanol', 'CO2'], cache=True))
    with pytest.raises(ValueError): ReactionSystem(fermentation(), other)
    rxnsys = ReactionSystem(fermentation(), fermentation())
    rxnsys._reactions[1].basis = 'wt'
    s = tmo.Stream(None, Glucose=10)
    with pytest.raises(RuntimeError): rxnsys(s)
    with pytest.raises(RuntimeError): rxnsys._conversion(s.mol)

def test_parse():
    chemicals = set_thermo()
    with pytest.raises(ValueError): prs.get_stoichiometric_array(1, chemicals)
    with pytest.raises(ValueError): prs.get_stoichiometric_string(1, chemicals)
    assert prs.get_stoichiometric_string({'Glucose': -1, 'Ethanol': 2}, chemicals) == 'Glucose -> 2 Ethanol'
    assert prs.get_stoichiometric_string(np.array([0, 1., -1, 0, 0, 0]), chemicals) == 'Glucose -> Ethanol'
    assert prs.dct2str({}) == 'no reaction'
    assert prs.str2dct('2.5e-1Glucose -> Ethanol') == {'Glucose': -0.25, 'Ethanol': 1.}
    with pytest.raises(ValueError): prs.str2dct('Glucose + Glucose -> Ethanol')
    with pytest.raises(tmo.exceptions.UndefinedChemical): prs.get_stoichiometric_array('Glucose -> Methanol', chemicals)
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Glucose'], cache=True)
    chemicals.compile()
    chemicals.define_group('Sugars', ['Glucose'])
    with pytest.raises(ValueError): prs.get_stoichiometric_array('Sugars -> Ethanol', chemicals)
    with pytest.raises(ValueError): xprs.get_stoichiometric_array('Sugars,l -> Ethanol,l', 'l', chemicals)

def test_xparse():
    chemicals = set_thermo()
    assert xprs.get_phases({'Glucose': ('l', -1), 'Ethanol': ('g', 1), 'H2': 0}) == ('g', 'l')
    assert xprs.get_phases('Glucose,') == ()
    assert xprs.get_phases(np.zeros(3)) == ()
    with pytest.raises(ValueError): xprs.get_phases(np.zeros([2, 3]))
    with pytest.raises(ValueError): xprs.get_phases(1)
    arr = xprs.get_stoichiometric_array(np.zeros([2, chemicals.size]), 'gl', chemicals)
    assert arr.shape == (2, chemicals.size)
    with pytest.raises(ValueError): xprs.get_stoichiometric_array(1, 'gl', chemicals)
    assert xprs.get_stoichiometric_string({'Glucose': ('l', -1)}, 'l', chemicals) == 'Glucose,l -> '
    with pytest.raises(ValueError): xprs.get_stoichiometric_string(1, 'l', chemicals)
    assert xprs.dct2str({}) == 'no reaction'
    assert xprs.dct2str({'Glucose': ('l', -0.5), 'Ethanol': ('l', 2.5)}) == '0.5 Glucose,l -> 2.5 Ethanol,l'
    with pytest.raises(ValueError): xprs.str2dct('Glucose,x -> Ethanol,l')
    with pytest.raises(ValueError): xprs.str2dct('Glucose -> Ethanol,l')
    with pytest.raises(ValueError): xprs.str2dct('Glucose,l + Glucose,g -> Ethanol,l')

def test_chemicals_reaction_data():
    water = '7732-18-5'
    glucose = '50-99-7'
    assert np.isfinite(chem_rxn.Hf(water))
    for phase in ('s', 'l', 'g'):
        Hf = chem_rxn.Hf(water, phase, Hvap=44e3, Hfus=6e3)
        assert Hf is None or np.isfinite(Hf)
        S0 = chem_rxn.S0(water, phase, Svap=100, Sfus=20)
        assert S0 is None or np.isfinite(S0)
    assert np.isfinite(chem_rxn.Hf(glucose, 'l', Hfus=30e3))
    with pytest.raises(ValueError): chem_rxn.Hf(water, 'x')
    with pytest.raises(ValueError): chem_rxn.S0(water, 'x')
    assert chem_rxn.Hf('0-00-0', 'g') is None
    assert chem_rxn.S0('0-00-0', 'g') is None
    assert chem_rxn.S0('0-00-0') is None
    fe = chem_rxn.free_energy_at_phase
    assert fe(None, 'g', 'l', 1, 1) is None
    assert fe(-10, 'g', 'g', 1, 1) == -10
    assert fe(-10, 'g', 'l', 1, 1) == -11
    assert fe(-10, 'g', 's', 1, 1) == -12
    assert fe(-10, 'l', 'g', 1, 1) == -9
    assert fe(-10, 'l', 's', 1, 1) == -11
    assert fe(-10, 's', 'l', 1, 1) == -9
    assert fe(-10, 's', 'g', 1, 1) == -8
    with pytest.raises(ValueError): fe(-10, 'g', 'x', 1, 1)

if __name__ == '__main__':
    test_reaction_basics()
    test_reaction_on_streams_and_arrays()
    test_reaction_feasibility()
    test_reaction_math()
    test_reaction_properties_and_setters()
    test_phased_reaction()
    test_rate_and_equilibrium()
    test_reaction_sets()
    test_reaction_sets_with_rates_and_phases()
    test_reset_chemicals_of_sets()
    test_reaction_system()
    test_parse()
    test_xparse()
    test_chemicals_reaction_data()
