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
import os
import tempfile
from numpy.testing import assert_allclose

def test_ideal_mixture():
    chemicals = tmo.CompiledChemicals(['Water', 'Ethanol'], cache=True)
    mixture = tmo.IdealMixture.from_chemicals(chemicals, include_excess_energies=True)
    assert 'IdealMixture' in repr(mixture) and 'IdealMixture' in mixture._info()
    mol = np.array([0.5, 0.5])
    T, P = 330., 101325.
    rho = mixture.rho('l', mol, T, P)
    assert 700 < rho < 1000
    assert mixture.rho('l', np.zeros(2), T, P) == 0
    Cp = mixture.Cp('l', mol, T)
    assert 2 < Cp < 6
    assert mixture.Cp('l', np.zeros(2), T) == 0
    assert 0 < mixture.alpha('l', mol, T, P) < 1e-6
    assert 0 < mixture.nu('l', mol, T, P) < 1e-5
    assert 1 < mixture.Pr('l', mol, T, P) < 100
    phase_mol = [('l', mol), ('g', mol)]
    assert mixture.xrho(phase_mol, T, P) > rho
    assert mixture.xCp(phase_mol, T) > Cp
    assert mixture.xalpha(phase_mol, T, P) > 0
    assert mixture.xnu(phase_mol, T, P) > 0
    assert mixture.xPr(phase_mol, T, P) > 0
    assert_allclose(mixture.get_property('rho', 'g/L', 'l', mol, T, P), rho, rtol=1e-2)
    with pytest.raises(ValueError):
        mixture.get_property('xrho', 'kg/m3', phase_mol, T, P)
    Cn = mixture.Cn('l', mol, T, P)
    assert Cn > 0
    assert_allclose(mixture.Cn('l', tmo.base.SparseVector(mol), T, P), Cn, rtol=1e-2)
    assert mixture('h', mol=mol, T=T, P=P, phase='l') > 0
    assert mixture('H', mol=np.zeros(2), T=T, P=P, phase='l') == 0
    assert mixture('V', mol=np.zeros(2), T=T, P=P, phase='l') is None
    assert mixture('sigma', mol=mol, T=T, P=P) > 0
    assert mixture('h', mol=np.array([mol, mol]), T=T, P=P, phases=('g', 'l')) > 0
    assert mixture('sigma', mol=np.array([mol, mol]), T=T, P=P, phases=('g', 'l')) > 0
    S = mixture.S('l', mol, 350., P)
    assert_allclose(mixture.solve_T_at_SP('l', mol, S, 300., P), 350., rtol=1e-2)

def test_eos_mixture():
    chemicals = tmo.CompiledChemicals(['Water', 'Ethanol'], cache=True)
    mixture = tmo.PRMixture.from_chemicals(chemicals)
    mol = np.array([0.5, 0.5])
    T, P = 330., 101325.
    for phase in ('l', 'g'):
        H = mixture.H(phase, mol, T, P)
        S = mixture.S(phase, mol, T, P)
        Cn = mixture.Cn(phase, mol, T, P)
        assert np.isfinite(H) and np.isfinite(S) and Cn > 0
        assert_allclose(mixture.solve_T_at_HP(phase, mol, H, 320., P), T, rtol=1e-2)
        assert_allclose(mixture.solve_T_at_SP(phase, mol, S, 320., P), T, rtol=1e-2)
        dh = mixture.dh_dep_dzs(phase, mol, T, P)
        assert np.isfinite(dh).all()
    for i in ('H', 'Cn'):
        assert getattr(mixture, i)('s', mol, T, P) is not None
    assert (mixture.dh_dep_dzs('s', mol, T, P) == 0).all()
    phase_mol = [('l', mol), ('g', mol)]
    H = mixture.xH(phase_mol, T, P)
    assert_allclose(mixture.xsolve_T_at_HP(phase_mol, H, 320., P), T, rtol=1e-2)
    S = mixture.xS(phase_mol, T, P)
    assert_allclose(mixture.xsolve_T_at_SP(phase_mol, S, 320., P), T, rtol=1e-2)
    mixture.H('l', tmo.base.SparseVector(mol), T, P)

def test_free_energy_functions():
    from thermosteam import free_energy as fe
    Water = tmo.Chemical('Water', cache=True)
    eos = Water.eos
    T, P = 350., 2e5
    for f in (fe.Excess_Liquid_Enthalpy_Ref_Gas, fe.Excess_Liquid_Entropy_Ref_Gas):
        assert np.isfinite(f(T, P, eos))
        assert np.isfinite(f(T, P, eos, ref=True))
        assert f(T, 101325, eos) == 0
    for f in (fe.Excess_Gas_Enthalpy_Ref_Gas, fe.Excess_Gas_Enthalpy_Ref_Liquid,
              fe.Excess_Gas_Entropy_Ref_Gas, fe.Excess_Gas_Entropy_Ref_Liquid):
        assert np.isfinite(f(T, P, eos, 0.))
        assert f(T, P, eos, 1., ref=True) == 1.
    for f in (fe.Excess_Liquid_Enthalpy_Ref_Liquid, fe.Excess_Liquid_Enthalpy_Ref_Solid,
              fe.Excess_Solid_Enthalpy_Ref_Solid, fe.Excess_Solid_Enthalpy_Ref_Liquid,
              fe.Excess_Solid_Enthalpy_Ref_Gas, fe.Excess_Liquid_Entropy_Ref_Liquid,
              fe.Excess_Liquid_Entropy_Ref_Solid, fe.Excess_Solid_Entropy_Ref_Solid,
              fe.Excess_Solid_Entropy_Ref_Liquid, fe.Excess_Solid_Entropy_Ref_Gas,
              fe.Excess_Gas_Entropy_Ref_Solid):
        assert f(T, P) == 0
    assert fe.Excess_Gas_Enthalpy_Ref_Solid(T) == 0
    Cn = Water.Cn.l
    assert_allclose(fe.Enthalpy(T, P, Cn, 298.15, 0.), 75.3 * 51.85, rtol=5e-2)
    assert fe.Entropy(T, P, Cn, 298.15, 70.) > 70.
    assert fe.EntropyGas(T, P, Water.Cn.g, 298.15, 101325., 188.) > 0

def test_thermo():
    thermo = tmo.Thermo(['Water', 'Ethanol'], cache=True)
    with pytest.raises(ValueError):
        tmo.Thermo(['Water'], mixture=1, cache=True)
    with pytest.raises(ValueError):
        tmo.Thermo(['Water'], pkg='not a package', cache=True)
    for pkg in ('ideal gas', 'Peng Robinson', 'Peng Robinson | Dortmund-UNIFAC'):
        assert tmo.Thermo(['Water', 'Ethanol'], pkg=pkg, cache=True).Phi is not None
    PR = tmo.Thermo(['Water', 'Ethanol'], pkg='Peng Robinson', cache=True)
    assert 'Gamma=None' not in repr(thermo)
    PR.show()
    sub = PR.subset(['Water'])
    assert [i.ID for i in sub.chemicals] == ['Water']
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Methanol'], cache=True)
    chemicals.compile()
    chemicals.define_group('Alcohols', ['Ethanol', 'Methanol'])
    thermo = tmo.Thermo(chemicals)
    assert thermo.subset(thermo.chemicals) is thermo
    sub = thermo.subset(['Water', 'Ethanol'])
    assert sub.chemicals.chemical_group_members('Alcohols') == ['Ethanol']
    extended = sub.extended([tmo.Chemical('Glucose', cache=True)])
    assert len(extended.chemicals) == 3
    IDs = thermo.chemicals[['Water', 'Ethanol']]
    assert 'Fugacities' in type(thermo.fugacities('g', IDs)).__name__
    assert 'Fugacities' in type(thermo.fugacities('l', IDs)).__name__
    with pytest.raises(NotImplementedError):
        thermo.fugacities('s', ['Water'])
    ideal = thermo.ideal()
    assert thermo.ideal() is ideal and ideal.ideal() is ideal
    assert 'IdealThermo' in repr(ideal)
    assert ideal.subset(ideal.chemicals) is ideal
    sub = ideal.subset(['Water', 'Ethanol'])
    assert sub.chemicals.chemical_group_members('Alcohols') == ['Ethanol']

def test_thermo_data():
    from thermosteam._thermo_data import as_IDs, ThermoData
    assert as_IDs(['Water']) == ['Water']
    with pytest.raises(ValueError):
        as_IDs([1])
    with pytest.raises(ValueError):
        as_IDs(1)
    data = ThermoData({
        'Chemicals': {
            'Water': {},
            'Ethanol': None,
            'Steam': {'Copy of': 'Water', 'Tb': 374.},
            'Ethanol2': {'Copy of': 'Methanol'},
        },
        'Streams': {'water': {'Water': 1}, 'empty': None},
    })
    all_data = data.chemical_data
    all_data['Methanol'] = {}
    chemicals = data.create_chemicals()
    assert [i.ID for i in chemicals] == ['Water', 'Ethanol', 'Steam', 'Ethanol2', 'Methanol']
    chemicals = data.create_chemicals(['Water', 'Steam'])
    assert [i.ID for i in chemicals] == ['Water', 'Steam']
    assert data.synonym_data == {}
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    stream = data.create_stream('water')
    assert stream.F_mol == 1
    assert data.create_stream('empty').isempty()
    streams = data.create_streams(['water'])
    assert len(streams) == 1
    data.data.clear()
    for name in ('chemical_data', 'stream_data', 'synonym_data'):
        with pytest.raises(AttributeError):
            getattr(data, name)
    with tempfile.TemporaryDirectory() as folder:
        file = os.path.join(folder, 'data.yaml')
        with open(file, 'w') as f: f.write("Chemicals:\n  Water:\n")
        assert 'Water' in ThermoData.from_yaml(file).chemical_data
        file = os.path.join(folder, 'data.json')
        with open(file, 'w') as f: f.write('{"Chemicals": {"Water": {}}}')
        assert 'Water' in ThermoData.from_json(file).chemical_data

def test_settings():
    settings = tmo.settings
    assert tmo._settings.ProcessSettings() is settings
    thermo = tmo.Thermo(['Water', 'Ethanol'], cache=True)
    original = settings._thermo if hasattr(settings, '_thermo') else None
    try:
        settings.set_thermo(thermo)
        assert settings.get_thermo() is thermo
        assert settings.get_chemicals() is thermo.chemicals
        assert settings.mixture is thermo.mixture
        assert settings.get_default_thermo(None) is thermo
        assert settings.get_default_thermo(thermo) is thermo
        with pytest.raises(ValueError):
            settings.get_default_thermo(1)
        assert settings.get_default_chemicals(None) is thermo.chemicals
        assert settings.get_default_chemicals(thermo.chemicals) is thermo.chemicals
        with pytest.raises(ValueError):
            settings.get_default_chemicals(1)
        assert settings.get_default_mixture(None) is thermo.mixture
        assert settings.get_default_mixture(thermo.mixture) is thermo.mixture
        with pytest.raises(ValueError):
            settings.get_default_mixture(1)
        assert settings.get_default_flashpkg(1) == 1
        assert settings.get_default_flasher(1) == 1
        settings.set_thermo(['Water', 'Ethanol'], cache=True, ideal=True)
        assert isinstance(settings.thermo, tmo.IdealThermo)
        del settings._thermo
        with pytest.raises(RuntimeError):
            settings.thermo
        with pytest.raises(RuntimeError):
            tmo._settings.raise_no_flashpkg_error()
    finally:
        if original is None:
            if hasattr(settings, '_thermo'): del settings._thermo
        else:
            settings.set_thermo(original)

def test_preferences():
    from thermosteam import _preferences as module
    preferences = tmo.preferences
    original = preferences.to_dict()
    original_file = module.__file__
    try:
        with tempfile.TemporaryDirectory() as folder:
            module.__file__ = os.path.join(folder, '_preferences.py')
            with preferences.temporary() as p:
                assert p is preferences
                p.flow = 'kg/hr:.4g'
                p.T = 'degC:.3g'
                p.P = 'bar:.3g'
                p.composition = True
                p.N = 5
                p.sort = True
                assert p.flow == 'kg/hr:.4g'
                assert p.T == 'degC:.3g'
                assert p.P == 'bar:.3g'
                assert p.composition and p.N == 5 and p.sort
            assert preferences.flow == original['flow']
            preferences.light_mode()
            assert preferences.unit_label_color == 'black'
            preferences.dark_mode()
            assert preferences.unit_label_color == 'white'
            preferences.update(N=8, save=True)
            assert os.path.exists(module.__file__.replace('_preferences.py', 'preferences.yaml'))
            preferences.reset()
            assert preferences.N == 7
            preferences.autoload()
            assert preferences.N == 8
            preferences.show()
            preferences.reset(save=True)
            preferences.light_mode(save=True)
            with pytest.raises(ZeroDivisionError):
                with preferences.temporary():
                    1 / 0
    finally:
        module.__file__ = original_file
        preferences.update(**original)

if __name__ == '__main__':
    test_ideal_mixture()
    test_eos_mixture()
    test_free_energy_functions()
    test_thermo()
    test_thermo_data()
    test_settings()
    test_preferences()
