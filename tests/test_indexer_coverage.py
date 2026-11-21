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
import pickle
import numpy as np
import thermosteam as tmo
from thermosteam import indexer as idx
from thermosteam.base import SparseVector, SparseArray
from numpy.testing import assert_allclose

def get_chemicals():
    chemicals = tmo.CompiledChemicals(['Water', 'Methanol', 'Ethanol'], cache=True)
    chemicals.define_group('Alcohol', ['Methanol', 'Ethanol'], composition=[0.5, 0.5])
    return chemicals

def get_other_chemicals():
    return tmo.CompiledChemicals(['Ethanol', 'Water', 'Glycerol'], cache=True)

def test_split_indexer():
    chemicals = get_chemicals()
    split = idx.SplitIndexer(chemicals, Water=0.5, Ethanol=1.)
    assert split['Water'] == 0.5
    assert split['Alcohol'].tolist() == [0., 1.]
    assert split['Water', 'Ethanol'].tolist() == [0.5, 1.]
    nested = split['Water', 'Alcohol']
    assert nested[0] == 0.5 and nested[1].tolist() == [0., 1.]
    assert split[...] is split.data
    split['Alcohol'] = 0.2
    assert split['Alcohol'].tolist() == [0.2, 0.2]
    split['Alcohol'] = 0.
    split['Alcohol'] = [0.3, 0.]
    assert split['Alcohol'].tolist() == [0.3, 0.]
    split['Water', 'Alcohol'] = 0.1
    assert split.data.to_array().tolist() == [0.1, 0.1, 0.1]
    split['Water', 'Alcohol'] = 0.
    assert not split.data.any()
    split['Water', 'Alcohol'] = [0.4, [0.5, 0.]]
    assert split.data.to_array().tolist() == [0.4, 0.5, 0.]
    split['Water', 'Alcohol'] = [0., 0.6]
    assert split.data.to_array().tolist() == [0., 0.6, 0.6]
    split['Water', 'Alcohol'] = [0.7, 0.]
    assert split.data.to_array().tolist() == [0.7, 0., 0.]
    split['Water', 'Ethanol'] = 0.9
    split['Water', 'Ethanol'] = 0.
    split['Water', 'Ethanol'] = [0.1, 0.]
    split['Water'] = 0.
    assert not split.data.any()
    split[...] = 0.5
    assert split.data.to_array().tolist() == [0.5, 0.5, 0.5]
    split[...] = [1., 0., 0.]
    split[...] = SparseVector([0., 1., 0.])
    split[...] = split.data
    assert split.data.to_array().tolist() == [0., 1., 0.]
    with pytest.raises(IndexError): split['Water'] = [1., 2.]
    with pytest.raises(IndexError): split['Alcohol'] = [[1.], [2.]]
    with pytest.raises(IndexError): split['Water', 'Ethanol'] = [[1.], [2.]]
    with pytest.raises(IndexError): split[...] = [[1.], [2.]]
    # Representation
    split['Methanol'] = 1.
    assert 'Methanol' in split._info(None)
    assert 'Methanol' in split._info(1)
    split.show()
    assert 'all zeros' in idx.SplitIndexer(chemicals)._info(None)
    # Pickle
    new = pickle.loads(pickle.dumps(split))
    assert new.data.sparse_equal(split.data)
    new = idx.SplitIndexer.from_data(np.array([0., 0.2, 1.]), chemicals)
    assert new['Ethanol'] == 1.
    with pytest.raises(TypeError): new.get_conversion_factor('kg/hr')
    new.empty()
    assert new.isempty()

def test_chemical_indexer():
    chemicals = get_chemicals()
    imol = idx.ChemicalMolarFlowIndexer('l', chemicals=chemicals, Water=2., Ethanol=1.)
    assert imol['Water'] == 2.
    assert imol['Alcohol'] == 1.
    assert imol['Water', 'Alcohol'].tolist() == [2., 1.]
    assert imol['Water', 'Ethanol'].tolist() == [2., 1.]
    assert imol[...] is imol.data
    imol['Alcohol'] = 2.
    assert_allclose(imol['Methanol', 'Ethanol'], [1., 1.])
    imol['Alcohol'] = [0., 3.]
    imol['Water', 'Alcohol'] = 4.
    assert_allclose(imol.data.to_array(), [4., 2., 2.])
    imol['Water', 'Alcohol'] = 0.
    assert imol.isempty()
    imol['Water', 'Alcohol'] = [1., 2.]
    assert_allclose(imol.data.to_array(), [1., 1., 1.])
    imol['Water', 'Alcohol'] = [0., 0.]
    imol['Water', 'Ethanol'] = 1.
    imol['Water', 'Ethanol'] = 0.
    imol['Water', 'Ethanol'] = [3., 0.]
    imol['Water'] = 0.
    imol['Water'] = 1.
    with pytest.raises(IndexError): imol['Water'] = [1., 2.]
    with pytest.raises(IndexError): imol['Alcohol'] = [[1.], [2.]]
    with pytest.raises(IndexError): imol['Water', 'Alcohol'] = [[[1.]], [[2.]]]
    with pytest.raises(IndexError): imol['Water', 'Ethanol'] = [[1.], [2.]]
    imol[...] = [1., 0., 2.]
    assert imol.get_index('Ethanol') == 2
    assert imol.sum_across_phases() is imol.data
    # Units
    assert_allclose(imol.get_data('kmol/s'), imol.data.to_array() / 3600)
    assert_allclose(imol.get_data('kmol/s', 'Water'), 1. / 3600)
    assert_allclose(imol.get_data('kmol/s', 'Water', 'Ethanol'), np.array([1., 2.]) / 3600)
    imol.set_data(np.array([1., 0., 1.]), 'kmol/s')
    assert_allclose(imol.data.to_array(), [3600., 0., 3600.])
    imol.set_data(1., 'kmol/s', 'Water')
    imol.set_data(np.array([1., 1.]), 'kmol/s', 'Water', 'Ethanol')
    assert_allclose(imol.data.to_array(), [3600., 0., 3600.])
    # Phase and composition
    phase, composition = imol.get_phase_and_composition()
    assert phase == 'l'
    assert_allclose(composition.to_array(), [0.5, 0., 0.5])
    empty = idx.ChemicalMolarFlowIndexer('l', chemicals=chemicals)
    with pytest.raises(RuntimeError): empty.get_phase_and_composition()
    assert '(empty)' in empty._info(None)
    # Representation
    assert 'phase' in repr(imol)
    assert 'phase' in format(imol, '0')
    assert 'Water' in imol._info(None)
    assert 'Water' in imol._info(1)
    imol.show()
    # Copy
    deep = imol.copy()
    assert deep.data is not imol.data and deep.data.sparse_equal(imol.data)
    shallow = imol.copy(deep=False)
    assert shallow.data is imol.data
    new = pickle.loads(pickle.dumps(imol))
    assert new.data.sparse_equal(imol.data)
    # Phase locking
    locked = idx.ChemicalIndexer('l', chemicals=chemicals, lock_phase=True)
    with pytest.raises(AttributeError): locked.phase = 'g'
    locked.phase = 'l'
    unlocked = idx.ChemicalIndexer.from_data(np.array([1., 0., 0.]), 'l', chemicals)
    unlocked.phase = 'g'
    assert unlocked.phase == 'g'
    # Reset chemicals
    other = get_other_chemicals()
    c = imol.copy()
    old = c.reset_chemicals(other)
    assert old.sparse_equal(imol.data)
    assert_allclose(c['Water', 'Ethanol'], imol['Water', 'Ethanol'])
    c = imol.copy()
    c.reset_chemicals(other, SparseVector.from_size(other.size))
    assert_allclose(c['Water', 'Ethanol'], imol['Water', 'Ethanol'])

def test_chemical_indexer_mixing_and_copying():
    chemicals = get_chemicals()
    other = get_other_chemicals()
    a = idx.ChemicalMolarFlowIndexer('l', chemicals=chemicals, Water=1., Ethanol=1.)
    b = idx.ChemicalMolarFlowIndexer('l', chemicals=other, Water=2., Ethanol=1.)
    m = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals, l=[('Water', 1.)], g=[('Ethanol', 3.)])
    m_other = idx.MolarFlowIndexer(phases='lg', chemicals=other, l=[('Water', 1.)], g=[('Ethanol', 3.)])
    mix = idx.ChemicalMolarFlowIndexer('l', chemicals=chemicals)
    mix.mix_from([a, b, m])
    assert_allclose(mix['Water', 'Ethanol'], [4., 5.])
    mix.separate_out(a)
    mix.separate_out(b)
    assert_allclose(mix['Water', 'Ethanol'], [1., 3.])
    # Copy like
    c = idx.ChemicalMolarFlowIndexer('g', chemicals=chemicals)
    c.copy_like(c)
    c.copy_like(a)
    assert c.phase == 'l' and c.data.sparse_equal(a.data)
    c.copy_like(b)
    assert_allclose(c['Water', 'Ethanol'], [2., 1.])
    # To material indexer
    mat = a.to_material_indexer('lg')
    assert_allclose(mat['l', ('Water', 'Ethanol')], [1., 1.])
    mat = a.to_material_indexer('Lg')
    assert_allclose(mat['L', ('Water', 'Ethanol')], [1., 1.])
    mat = idx.ChemicalMolarFlowIndexer('l', chemicals=chemicals).to_material_indexer('lg')
    assert mat.isempty()
    s = idx.ChemicalMolarFlowIndexer('S', chemicals=chemicals, Water=1.)
    mat = s.to_material_indexer('sl')
    assert mat['s', 'Water'] == 1.
    # Mass and volume
    mass = a.by_mass()
    assert mass is a.by_mass()
    assert_allclose(mass['Water'], chemicals.Water.MW, rtol=1e-6)
    assert mass.phase == 'l'
    TP = tmo.ThermalCondition(298.15, 101325.)
    vol = a.by_volume(TP)
    assert vol is a.by_volume(TP)
    assert vol['Water'] > 0 and np.isfinite(vol['Water'])
    with pytest.raises(AttributeError):
        vol['Alcohol'] = 1.

def test_material_indexer_getitem_setitem():
    chemicals = get_chemicals()
    m = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals,
                             l=[('Water', 2.), ('Ethanol', 1.)],
                             g=[('Methanol', 1.), ('Ethanol', 1.)])
    # Sum across phases
    assert m['Water'] == 2.
    assert m['Alcohol'] == 3.
    assert_allclose(m['Water', 'Alcohol'], [2., 3.])
    assert_allclose(m['Water', 'Ethanol'], [2., 2.])
    assert_allclose(m[...].to_array(), [2., 1., 2.])
    # By phase
    assert_allclose(m['l'].to_array(), [2., 0., 1.])
    assert_allclose(m['gl'].to_array(), m.data.to_array())
    assert_allclose(m[..., 'Water'], [0., 2.])
    assert_allclose(m[..., 'Alcohol'], [2., 1.])
    assert_allclose(m[..., ('Water', 'Alcohol')], [[0., 2.], [2., 1.]])
    assert_allclose(m[..., ('Water', 'Ethanol')], [[0., 1.], [2., 1.]])
    assert_allclose(m['lg', 'Water'], [2., 0.])
    assert_allclose(m['l', ('Water', 'Ethanol')], [2., 1.])
    assert m['l', 'Alcohol'] == 1.
    assert_allclose(m['l', ('Water', 'Alcohol')], [2., 1.])
    assert_allclose(m[['l', ['Water', 'Ethanol']]], [2., 1.])
    with pytest.raises(IndexError): m[1, 'Water']
    with pytest.raises(IndexError): m['l', 'Water', 'Ethanol']
    with pytest.raises(tmo.exceptions.UndefinedPhase): m['lx', 'Water']
    with pytest.raises(tmo.exceptions.UndefinedChemicalAlias): m['Glycerol']
    # Setting
    m['l'] = [1., 1., 1.]
    m['lg'] = 0.
    assert m.isempty()
    m['lg'] = [[1., 0., 0.], [0., 0., 1.]]
    m['lg'] = [0., 1., 0.]
    assert_allclose(m.data.to_array(), [[0., 1., 0.], [0., 1., 0.]])
    m[...] = 0.
    m['Water'] = 0.
    with pytest.raises(IndexError): m['Water'] = 1.
    m[..., 'Water'] = 1.
    assert_allclose(m[..., 'Water'], [1., 1.])
    m[..., ('Water', 'Ethanol')] = 0.
    m[..., 'Alcohol'] = 2.
    assert_allclose(m[..., ('Methanol', 'Ethanol')], [[1., 1.], [1., 1.]])
    m[..., ('Water', 'Alcohol')] = [3., 4.]
    assert_allclose(m[..., ('Water', 'Methanol', 'Ethanol')], [[3., 2., 2.], [3., 2., 2.]])
    m['lg', 'Water'] = 1.
    m['lg', ('Water', 'Ethanol')] = [[1., 2.], [3., 4.]]
    assert_allclose(m['lg', ('Water', 'Ethanol')], [[1., 2.], [3., 4.]])
    m['l', 'Alcohol'] = 4.
    assert_allclose(m['l', ('Methanol', 'Ethanol')], [2., 2.])
    m['l', ('Water', 'Alcohol')] = [1., 2.]
    assert_allclose(m['l', ('Water', 'Methanol', 'Ethanol')], [1., 1., 1.])
    # Units
    assert_allclose(m.get_data('kmol/s', 'l', 'Water'), 1. / 3600)
    m.set_data(1., 'kmol/s', 'l', 'Water')
    assert_allclose(m['l', 'Water'], 3600.)

def test_material_indexer_methods():
    chemicals = get_chemicals()
    other = get_other_chemicals()
    m = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals,
                             l=[('Water', 2.), ('Ethanol', 1.)],
                             g=[('Ethanol', 1.)])
    assert m.phases == ('g', 'l')
    assert m.get_phase_index('l') == 1
    assert not m.phases_are_empty('l')
    assert m.phases_are_empty('s')
    assert_allclose(m.sum_across_phases().to_array(), [2., 0., 2.])
    assert dict(m)['l'].to_array().tolist() == [2., 0., 1.]
    comp = dict(m.iter_composition())
    assert_allclose(comp['l'].to_array().sum() + comp['g'].to_array().sum(), 1.)
    assert 'MolarFlowIndexer' in repr(m)
    assert 'Ethanol' in m._info(None)
    assert 'Ethanol' in m._info(1)
    m.show()
    empty = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals)
    assert '(empty)' in empty._info(None)
    assert 'phases' in repr(empty)
    # Copies
    deep = m.copy()
    assert deep.data.sparse_equal(m.data) and deep.data is not m.data
    shallow = m.copy(deep=False)
    assert shallow.data is m.data
    new = pickle.loads(pickle.dumps(m))
    assert new.data.sparse_equal(m.data)
    # Mass and volume
    mass = m.by_mass()
    assert mass is m.by_mass()
    assert_allclose(mass['l', 'Water'], 2 * chemicals.Water.MW, rtol=1e-6)
    TP = tmo.ThermalCondition(298.15, 101325.)
    vol = m.by_volume(TP)
    assert vol is m.by_volume(TP)
    assert np.isfinite(vol['l', 'Water']) and vol['l', 'Water'] > 0
    # Reset chemicals
    c = m.copy()
    c.reset_chemicals(other)
    assert_allclose(c['l', ('Water', 'Ethanol')], [2., 1.])
    # Separate out
    total = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals)
    total.copy_like(m)
    total.separate_out(m)
    assert total.isempty()
    m_other = idx.MolarFlowIndexer(phases='lg', chemicals=other, l=[('Water', 2.), ('Ethanol', 1.)], g=[('Ethanol', 1.)])
    total.copy_like(m)
    total.separate_out(idx.MolarFlowIndexer(phases='lgs', chemicals=chemicals, l=[('Water', 2.)]))
    total.separate_out(idx.MolarFlowIndexer(phases='lgs', chemicals=other, l=[('Ethanol', 1.)]))
    total.separate_out(idx.ChemicalMolarFlowIndexer('g', chemicals=chemicals, Ethanol=0.5))
    total.separate_out(idx.ChemicalMolarFlowIndexer('g', chemicals=other, Ethanol=0.5))
    assert_allclose(total.data.to_array(), 0., atol=1e-12)
    with pytest.raises(ValueError): total.separate_out(idx.SplitIndexer(chemicals))
    with pytest.raises(ValueError): total.mix_from([idx.SplitIndexer(chemicals)])
    # Mixing with different phases and chemicals
    parent = idx.parent_indexer(chemicals)
    mix = parent.get_phases('lg')
    s = idx.ChemicalMolarFlowIndexer('s', chemicals=chemicals, Water=1.)
    L = idx.ChemicalMolarFlowIndexer('L', chemicals=chemicals, Water=1.)
    c_other = idx.ChemicalMolarFlowIndexer('l', chemicals=other, Water=1.)
    mix.mix_from([m, m_other, L, c_other])
    assert_allclose(mix.sum_across_phases().sum(), 2 * 4. + 1 + 1)
    assert_allclose(mix['l', 'Water'], 2 + 2 + 1 + 1)
    # Copy like
    mix2 = parent.get_phases('lg')
    mix2.copy_like(mix2)
    mix2.copy_like(L)
    assert mix2['l', 'Water'] == 1.
    mix3 = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals)
    mix3.copy_like(c_other)
    assert mix3['l', 'Water'] == 1.
    mix3.copy_like(m)
    assert mix3.data.sparse_equal(m.data)
    mix4 = idx.MolarFlowIndexer(phases='lgs', chemicals=chemicals)
    mix4.copy_like(m)
    assert_allclose(mix4.sum_across_phases().to_array(), m.sum_across_phases().to_array())
    mix4.copy_like(m_other)
    assert_allclose(mix4['Water', 'Ethanol'], [2., 2.])
    mix5 = idx.MolarFlowIndexer(phases='lg', chemicals=chemicals)
    mix5.copy_like(m_other)
    assert_allclose(mix5['Water', 'Ethanol'], [2., 2.])
    mix6 = parent.get_phases('lg')
    mix6.copy_like(idx.MolarFlowIndexer(phases='lgs', chemicals=chemicals, s=[('Water', 1.)]))
    assert mix6['s', 'Water'] == 1.
    mix7 = parent.get_phases('lg')
    mix7.copy_like(idx.MolarFlowIndexer(phases='lgs', chemicals=other, s=[('Water', 1.)]))
    assert mix7['s', 'Water'] == 1.

def test_material_indexer_phase_conversion():
    chemicals = get_chemicals()
    parent = idx.parent_indexer(chemicals)
    m = parent.get_phases('lg')
    m['l', 'Water'] = 1.
    m['g', 'Ethanol'] = 2.
    assert m.linked_copy().data.sparse_equal(m.data)
    chemical = m.to_chemical_indexer('l')
    assert_allclose(chemical['Water', 'Ethanol'], [1., 2.])
    phase = m.get_phase('l')
    assert phase['Water'] == 1.
    m2 = idx.parent_indexer(chemicals).get_phases('lg')
    m2['l', 'Water'] = 1.
    m2['g', 'Ethanol'] = 2.
    m3 = m2.to_material_indexer('Lg')
    assert_allclose(m3.sum_across_phases().to_array(), [1., 0., 2.])
    # Copies with parent
    deep = m3.copy()
    assert deep.data.sparse_equal(m3.data)
    shallow = m3.copy(deep=False)
    assert shallow.data.sparse_equal(m3.data)
    c = parent.get_phase('l')
    c['Water'] = 3.
    deep = c.copy()
    assert deep['Water'] == 3.
    c.phase = 'g'
    assert c['Water'] == 3.
    # Phase expansion and reset
    m4 = parent.get_phases(['l'])
    m4._expand_phases('gs')
    assert set(m4.phases) == {'g', 'l', 's'}
    m4 = parent.get_phases(['l'])
    assert m4.phases == ('l',)

if __name__ == '__main__':
    test_split_indexer()
    test_chemical_indexer()
    test_chemical_indexer_mixing_and_copying()
    test_material_indexer_getitem_setitem()
    test_material_indexer_methods()
    test_material_indexer_phase_conversion()
