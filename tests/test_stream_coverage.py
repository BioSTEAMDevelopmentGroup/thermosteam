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
from numpy.testing import assert_allclose

def set_thermo():
    chemicals = tmo.CompiledChemicals(['Water', 'Ethanol', 'Methanol'], cache=True)
    chemicals.define_group('Alcohol', ['Ethanol', 'Methanol'], composition=[0.5, 0.5])
    tmo.settings.set_thermo(chemicals)
    return chemicals

def get_other_thermo():
    return tmo.Thermo(['Ethanol', 'Water', 'Glycerol'], cache=True)

def test_stream_creation():
    chemicals = set_thermo()
    s = tmo.Stream('s_cov1', Water=1., Ethanol=1., units='kg/hr', total_flow=4.)
    assert_allclose(s.F_mass, 4., rtol=1e-6)
    s = tmo.Stream('s_cov2', Alcohol=2., units='kg/hr')
    assert_allclose(s.imass['Ethanol', 'Methanol'].sum(), 2., rtol=1e-6)
    s = tmo.Stream('s_cov3', Water=1., units='m3/hr')
    assert_allclose(s.F_vol, 1., rtol=1e-6)
    with pytest.raises(ValueError):
        tmo.Stream('s_cov4', Alcohol=1., units='m3/hr')
    s = tmo.Stream('s_cov5', Water=1., Ethanol=3., total_flow=2.)
    assert_allclose(s.F_mol, 2.)
    s = tmo.Stream('s_cov6', flow=('Water', 2.))
    assert s.imol['Water'] == 2.
    s = tmo.Stream('s_cov7', flow=[('Water', 1.), ('Ethanol', 2.)])
    assert_allclose(s.imol['Water', 'Ethanol'], [1., 2.])
    s = tmo.Stream('s_cov8', flow=np.array([1., 2., 3.]))
    assert_allclose(s.mol.to_array(), [1., 2., 3.])
    imol = tmo.indexer.ChemicalMolarFlowIndexer('l', chemicals=chemicals, Water=1.)
    s = tmo.Stream('s_cov9', flow=imol, phase='g')
    assert s.imol is imol and s.phase == 'g'
    with pytest.raises(AttributeError):
        s.price = float('inf')
    s.price = 2.
    assert_allclose(s.cost, 2. * s.F_mass)
    # Pickle round trip
    new = pickle.loads(pickle.dumps(s))
    assert_allclose(new.mol.to_array(), s.mol.to_array())
    assert new.price == s.price

def test_stream_basic_protocols():
    set_thermo()
    s = tmo.Stream('s_cov10', Water=1., phase='l')
    assert len(s) == 1
    assert list(s) == [s]
    assert s['l'] is s and s['L'] is s
    assert s.liquid_fraction == 1. and s.solid_fraction == 0. and s.vapor_fraction == 0.
    s.phase = 's'
    assert s.solid_fraction == 1. and s.liquid_fraction == 0.
    assert s.node_tag is None
    assert s.E_node is None and s.T_node is None and s.F_node is None
    imol, phase = s.material_reference
    assert phase == 's'
    assert not s.isfeed() and not s.isproduct()
    with pytest.raises(RuntimeError): s.get_feed_priority()
    with pytest.raises(RuntimeError): s.set_feed_priority(1)
    s.disconnect()
    assert s.source is None and s.sink is None
    # Equations
    f = lambda: 1
    assert s.material_balance(f) is f
    assert s.energy_balance(f) is f
    assert s.bulk_balance(f) is f
    assert s._create_material_balance_equations() == [1, 1]
    assert s._create_energy_balance_equations() == []
    assert 'Equations' in repr(s.equations)
    s._update_variable('T', 310.)
    assert s.T == 310.
    # Temporary conditions
    s.phase = 'l'
    with s.temporary(flow=np.array([2., 0., 0.]), T=350., P=2e5) as t:
        assert t.T == 350. and t.P == 2e5 and t.imol['Water'] == 2.
    assert s.T == 310. and s.imol['Water'] == 1.
    with s.temporary():
        pass
    with pytest.raises(ValueError):
        with s.temporary(T=350.):
            raise ValueError('expected')
    assert s.T == 310.
    with s.temporary_phase('g') as t:
        assert t.phase == 'g'
    assert s.phase == 'l'
    with pytest.raises(ValueError):
        with s.temporary_phase('g'):
            raise ValueError('expected')
    assert s.phase == 'l'
    # From data
    new = tmo.Stream.from_data(s.get_data(), ID='s_cov11', price=1.)
    assert_allclose(new.mol.to_array(), s.mol.to_array())

def test_stream_characterization_factors():
    set_thermo()
    s = tmo.Stream('s_cov12', Water=1., Ethanol=1., units='kg/hr')
    assert s.get_CF('GWP') == 0
    assert_allclose(s.get_CF('WU'), 0.5)
    s.set_CF('GWP', 2., basis='g')
    assert_allclose(s.get_CF('GWP'), 2000.)
    assert_allclose(s.get_CF('GWP', basis='g'), 2.)
    assert_allclose(s.get_impact('GWP'), 2000. * s.F_mass)
    assert s.get_impact('WU') == 0.
    assert s.get_impact('other') == 0.
    assert s.has_impact('GWP')
    assert not s.has_impact('other')

def test_stream_thermo_properties():
    set_thermo()
    s = tmo.Stream('s_cov13', Water=1., Ethanol=1., T=300.)
    assert np.isfinite(s.hf) and np.isfinite(s.G)
    assert s.LHV > 0 and s.HHV > s.LHV
    assert s.Hvap > 0 and s.hvap > 0
    assert_allclose(s.Hvap, s.hvap * s.F_mol, rtol=1e-6)
    # Setters that solve for temperature
    H = s.H
    s.T = 310.
    s.H = H
    assert_allclose(s.T, 300., rtol=1e-2)
    s.h = s.h
    assert_allclose(s.T, 300., rtol=1e-2)
    S = s.S
    s.T = 310.
    s.S = S
    assert_allclose(s.T, 300., rtol=1e-2)
    s.T = 310.
    s.s = s.s
    assert_allclose(s.T, 310., rtol=1e-2)
    assert np.isfinite(s.s)
    Hnet = s.Hnet
    s.T = 320.
    s.Hnet = Hnet
    assert_allclose(s.T, 310., rtol=1e-2)
    # Flow setters
    s.F_mol = 4.
    assert_allclose(s.F_mol, 4.)
    s.F_vol = 2 * s.F_vol
    assert_allclose(s.F_mol, 8., rtol=1e-6)
    s.F_mass = 0.
    assert s.isempty()
    # Empty stream properties
    s.H = 0.
    s.h = 0.
    s.S = 0.
    s.s = 0.
    assert s.rho is None and s.nu is None
    assert_allclose(s.z_mass.to_array() if hasattr(s.z_mass, 'to_array') else s.z_mass, 0.)
    assert s.get_concentration('Water') == 0.
    for f in (s.get_normalized_mol, s.get_normalized_mass, s.get_normalized_vol):
        with pytest.raises(RuntimeError): f(['Water'])
    s.imol['Water'] = 1.
    assert s.vlle_chemicals == s.vle_chemicals or len(s.vlle_chemicals) >= 1
    assert s.rho > 0 and s.nu > 0

def test_stream_arithmetic():
    set_thermo()
    s1 = tmo.Stream('s_cov14', Water=1., Ethanol=2.)
    s2 = tmo.Stream('s_cov15', Water=3.)
    total = s1 + s2
    assert_allclose(total.F_mol, 6.)
    total = sum([s1, s2])
    assert_allclose(total.F_mol, 6.)
    diff = total - s2
    assert_allclose(diff.mol.to_array(), s1.mol.to_array())
    # Unsupported operand (biosteam overrides integer subtraction for piping)
    assert s1.__sub__('not a stream') is NotImplemented
    neg = -s1
    assert_allclose(neg.mol.to_array(), -s1.mol.to_array())
    assert_allclose((s1 * 2).mol.to_array(), 2 * s1.mol.to_array())
    assert_allclose((2 * s1).mol.to_array(), 2 * s1.mol.to_array())
    assert_allclose((s1 / 2).mol.to_array(), s1.mol.to_array() / 2)
    s3 = s1.copy('s_cov16')
    s3 += s2
    assert_allclose(s3.F_mol, 6., rtol=1e-6)
    s3 -= s2
    assert_allclose(s3.mol.to_array(), s1.mol.to_array(), rtol=1e-6)
    s3 *= 2
    assert_allclose(s3.F_mol, 6.)
    s3 /= 3
    assert_allclose(s3.F_mol, 2.)
    s3.rescale(2)
    assert_allclose(s3.F_mol, 4.)

def test_stream_mixing_and_splitting():
    chemicals = set_thermo()
    s1 = tmo.Stream('s_cov17', Water=1., Ethanol=2., T=300.)
    s2 = tmo.Stream('s_cov18', Water=3., T=320., P=2e5)
    mix = tmo.Stream('s_cov19')
    mix.mix_flows([s1, s2])
    assert_allclose(mix.F_mol, 6.)
    mix.mix_from([s1, tmo.Stream('s_cov20')], energy_balance=False)
    assert_allclose(mix.mol.to_array(), s1.mol.to_array())
    assert mix.P == s1.P
    mix.mix_from([s2], energy_balance=False)
    assert mix.P == s2.P
    # Separate out
    mix.mix_from([s1, s2])
    H = mix.H
    mix.separate_out(s2)
    assert_allclose(mix.mol.to_array(), s1.mol.to_array(), rtol=1e-6)
    assert_allclose(mix.H, H - s2.H, rtol=1e-2, atol=1.)
    mix.separate_out(mix)
    assert mix.isempty()
    # Split to streams with a different property package
    other = get_other_thermo()
    out1 = tmo.Stream('s_cov21', thermo=other)
    out2 = tmo.Stream('s_cov22', thermo=other)
    s1.split_to(out1, out2, 0.25)
    assert_allclose(out1.imol['Water', 'Ethanol'], [0.25, 0.5])
    assert_allclose(out2.imol['Water', 'Ethanol'], [0.75, 1.5])
    # Copy flows from a different property package
    c = tmo.Stream('s_cov23')
    c.copy_flow(out2)
    assert_allclose(c.imol['Water', 'Ethanol'], [0.75, 1.5])
    c.copy_flow(out2, exclude=True)
    assert_allclose(c.imol['Water', 'Ethanol'], [0.75, 1.5])
    # Copy from a multi-phase stream with one phase
    ms = tmo.MultiStream('ms_cov1', phases='lg', g=[('Water', 1.)])
    ms.phases = ('g',)
    c.copy_like(ms)
    assert c.phase == 'g' and c.imol['Water'] == 1.
    ms2 = tmo.MultiStream('ms_cov2', l=[('Water', 1.)], g=[('Ethanol', 1.)])
    with pytest.raises(ValueError): c.copy_phase(ms2)
    c.copy_phase(s1)
    assert c.phase == 'l'
    # Copy with a different property package
    new = s1.copy('s_cov24', thermo=other)
    assert_allclose(new.imol['Water', 'Ethanol'], [1., 2.])

def test_stream_phase_equilibrium_objects():
    set_thermo()
    s = tmo.Stream('s_cov25', Water=1., Ethanol=1., phase='g')
    s.lle
    assert set(s.phases) == {'L', 'l'}
    s = tmo.Stream('s_cov26', Water=1., phase='g')
    s.sle
    assert set(s.phases) == {'s', 'l'}

def test_stream_representation():
    set_thermo()
    s = tmo.Stream('s_cov27', Water=1., Ethanol=2., Methanol=0.5)
    for layout in ('mol', 'vol', 'wt', 'cwt', '%mol', 'mol2', '2', 'wts', 'wt%'):
        s.show(layout)
    with pytest.raises(ValueError): s.show('abc')
    s.show(N=0)
    s.show(IDs=['Water', 'Ethanol'])
    s.show(N=1)
    df = s._info(df=True)
    assert df.shape[0] > 0
    df = s._info(composition=True, df=True)
    assert df.shape[0] > 0
    df = s._info(N=1, df=True)
    assert df.shape[0] > 0
    assert s._get_tooltip_string('txt', True) == ''
    assert 'Temperature' in s._get_tooltip_string('svg', False)
    assert 'Temperature' in s._get_tooltip_string('html', False)
    assert s._get_tooltip_string('html', True)
    empty = tmo.Stream('s_cov28')
    assert empty._get_tooltip_string('html', True) == '(empty)'
    df = empty._info(df=True)
    assert df.shape == (1, 1)
    ms = tmo.MultiStream('ms_cov3', l=[('Water', 1.)], g=[('Ethanol', 1.)])
    assert 'Temperature' in ms._get_tooltip_string('svg', False)
    assert ms._info(df=True).shape[0] > 0

def test_multi_stream():
    chemicals = set_thermo()
    ms = tmo.MultiStream('ms_cov4', l=[('Water', 1.), ('Alcohol', 2.)], g=[('Ethanol', 1.)], units='kg/hr')
    assert_allclose(ms.imass['l', ('Ethanol', 'Methanol')].sum(), 2., rtol=1e-6)
    assert_allclose(ms.F_mass, 4., rtol=1e-6)
    ms = tmo.MultiStream('ms_cov5', l=[('Water', 1.)], g=[('Ethanol', 1.)], units='kg/hr', total_flow=4.)
    assert_allclose(ms.F_mass, 4., rtol=1e-6)
    ms = tmo.MultiStream('ms_cov7', l=[('Water', 1.)], g=[('Ethanol', 3.)], total_flow=2.)
    assert_allclose(ms.F_mol, 2.)
    imol = tmo.indexer.MolarFlowIndexer(phases='lg', chemicals=chemicals, l=[('Water', 1.)])
    ms = tmo.MultiStream('ms_cov8', flow=imol)
    assert ms.imol is imol
    ms = tmo.MultiStream('ms_cov9', flow=[[1., 0., 0.], [0., 1., 0.]], phases='gl')
    assert_allclose(ms.F_mol, 2.)
    with pytest.raises(ValueError): ms[('l', 'g')]
    # From streams
    s1 = tmo.Stream('s_cov29', Water=1., phase='l')
    s2 = tmo.Stream('s_cov30', Ethanol=1., phase='g')
    ms = tmo.MultiStream.from_streams([s1, s2])
    assert_allclose(ms.F_mol, 2.)
    assert ms['l'] is s1 or ms['l'].imol['Water'] == 1.
    with pytest.raises(ValueError): tmo.MultiStream.from_streams([])
    with pytest.raises(ValueError):
        tmo.MultiStream.from_streams([s1, tmo.Stream('s_cov31', Water=1., phase='l')])
    # Thermo properties
    ms = tmo.MultiStream('ms_cov10', T=300., l=[('Water', 1.)], g=[('Ethanol', 1.)])
    h = ms.h
    ms.T = 310.
    ms.h = h
    assert_allclose(ms.T, 300., rtol=1e-2)
    S = ms.S
    ms.T = 310.
    ms.S = S
    assert_allclose(ms.T, 300., rtol=1e-2)
    empty = tmo.MultiStream('ms_cov11', phases='lg')
    assert empty.H == 0. and empty.h is None
    empty.h = 0.
    empty.S = 0.
    # Copy flow with exclusion from a single phase stream
    ms = tmo.MultiStream('ms_cov12', phases='lg')
    s = tmo.Stream('s_cov35', Water=1., Ethanol=2.)
    ms.copy_flow(s, 'l', 'Water', exclude=True, remove=True)
    assert ms.imol['l', 'Ethanol'] == 2. and ms.imol['l', 'Water'] == 0.
    assert s.imol['Water'] == 1. and s.imol['Ethanol'] == 0.
    ms.empty()
    s.imol['Ethanol'] = 2.
    ms.copy_flow(s, 'l', 'Ethanol', remove=True)
    assert ms.imol['l', 'Ethanol'] == 2. and s.imol['Ethanol'] == 0.
    # Phase handling
    ms = tmo.MultiStream('ms_cov13', phases='lg')
    ms.sle
    assert 's' in ms.phases
    ms.as_stream()
    assert len(ms.phase) == 1 and isinstance(ms, tmo.Stream)
    ms = tmo.MultiStream('ms_cov14', l=[('Water', 1.)], g=[('Ethanol', 1.)])
    with pytest.raises(RuntimeError): ms.as_stream()
    ms.reduce_phases()
    assert set(ms.phases) == {'g', 'l'}
    ms.phase = ''
    assert ms.phase == 'l' and not ms.isempty()
    # Representation with many chemicals
    ms = tmo.MultiStream('ms_cov15', l=[('Water', 1.), ('Ethanol', 1.), ('Methanol', 1.)])
    ms.show(N=1)
    ms.show(N=1, composition=True)

if __name__ == '__main__':
    test_stream_creation()
    test_stream_basic_protocols()
    test_stream_characterization_factors()
    test_stream_thermo_properties()
    test_stream_arithmetic()
    test_stream_mixing_and_splitting()
    test_stream_phase_equilibrium_objects()
    test_stream_representation()
    test_multi_stream()
