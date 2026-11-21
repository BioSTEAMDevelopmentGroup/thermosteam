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

def test_chemical_creation_options():
    Water = tmo.Chemical('Water', cache=True)
    with pytest.warns(RuntimeWarning):
        assert tmo.Chemical('Water', cache=True, phase='l') is Water
    with pytest.raises(ValueError):
        tmo.Chemical('Water', db='not a database')
    with pytest.raises(ValueError):
        tmo.Chemical('Water', Cp=4.18)
    Methanol = tmo.Chemical('Methanol', phase='l', Cp=2.5, rho=790., mu=5e-4,
                            kappa=0.2, Cn=None, sigma=0.022, epsilon=33.,
                            Psat=1e4, Hvap=35000.)
    assert_allclose(Methanol.Cn(300.), 2.5 * Methanol.MW, rtol=1e-2)
    assert_allclose(Methanol.V(300., 101325.), Methanol.MW / 790. / 1000., rtol=1e-2)
    Glucose = tmo.Chemical('Glucose', db='BioSTEAM')
    assert Glucose.MW > 100
    Ethanol = tmo.Chemical('Ethanol', CAS='64-17-5', method='POLY_FIT')
    assert Ethanol.CAS == '64-17-5'
    O2 = tmo.Chemical('O2', synonyms='Oxygen-coverage', search_db=False,
                      formula='O2', phase='g')
    assert 'Oxygen-coverage' in O2.aliases
    CO2 = tmo.Chemical.blank('CO2', synonyms=['Carbon-dioxide-coverage'], formula={'C': 1, 'O': 2})
    assert 'Carbon-dioxide-coverage' in CO2.aliases
    assert_allclose(CO2.MW, 44, rtol=1e-2)
    X = tmo.Chemical.blank('X', MW=('18', 'g/mol') if False else 18., Hf=(1, 'kJ/mol'))
    assert_allclose(X.Hf, 1000., rtol=1e-2)
    Y = tmo.Chemical.blank('Y', MW=18., Hf=(1, 'kJ/kg'))
    assert_allclose(Y.Hf, 18., rtol=1e-2)
    assert tmo._chemical.as_chemical(Water) is Water
    assert tmo._chemical.as_chemical('Water').CAS == Water.CAS
    assert 'CompressibilityFactor' in repr(tmo._chemical.CompressibilityFactor(None))
    Z = tmo._chemical.CompressibilityFactor(lambda T, P: Water.V('g', T, P))
    assert_allclose(Z.Z(400., 101325.), 1., rtol=5e-2)
    assert tmo._chemical.create_eos(None, 1, 1, 1) is not None

def test_chemical_setters():
    Water = tmo.Chemical('Water')
    Water.InChI = Water.InChI
    Water.InChI_key = Water.InChI_key
    Water.common_name = 'water'
    Water.iupac_name = 'oxidane'
    assert Water.iupac_name == ('oxidane',)
    Water.iupac_name = ['oxidane']
    Water.pubchemid = Water.pubchemid
    Water.smiles = Water.smiles
    assert Water.smiles == 'O'
    Water.formula = {'H': 2, 'O': 1}
    assert_allclose(Water.MW, 18.015, rtol=1e-2)
    for name in ('mu', 'kappa', 'V', 'Cn', 'Psat', 'Hvap', 'Hsub', 'Psub', 'sigma',
                 'epsilon', 'MW'):
        with pytest.raises(AttributeError):
            setattr(Water, name, None)
    assert Water.Hsub is not None and Water.Psub is not None
    Water.Tm = Water.Tm
    Water.Tb = Water.Tb
    Water.Pt = Water.Pt
    Water.Tt = Water.Tt
    Water.Tc = Water.Tc
    Water.Pc = Water.Pc
    Water.Vc = Water.Vc
    Water.Hfus = Water.Hfus
    Water.Sfus = 22.
    Water.omega = Water.omega
    Water.dipole = Water.dipole
    Water.similarity_variable = Water.similarity_variable
    Water.iscyclic_aliphatic = Water.iscyclic_aliphatic
    Water.Hf = Water.Hf
    Water.S0 = Water.S0
    Water.LHV = Water.LHV
    Water.HHV = Water.HHV
    Water.combustion = Water.combustion
    Water.N_solutes = 2
    assert Water.N_solutes == 2
    Water.atoms = {'H': 2, 'O': 1}
    assert_allclose(Water.MW, 18.015, rtol=1e-2)
    Water.phase_ref = 'l'
    with pytest.raises(ValueError):
        Water.phase_ref = 'x'
    assert_allclose(Water.Cn('l', 300.), 75.3, rtol=5e-2)
    assert Water.charge == 0
    with pytest.raises(ValueError):
        Water.get_property('Tsat', 'degC', 101325)
    assert_allclose(Water.get_property('Psat', 'bar', 373.15), 1.013, rtol=2e-2)

def test_chemical_methods():
    Water = tmo.Chemical('Water')
    assert Water.get_phase(200.) == 's'
    assert Water.get_phase(400.) == 'g'
    assert Water.get_phase(300.) == 'l'
    with pytest.raises(ValueError):
        Water.Tsat(1e9)
    assert_allclose(Water.Tsat(101325), 373.15, rtol=1e-2)
    assert_allclose(Water.Tsat(2e5), 393., rtol=1e-2)
    Water._Tb = None
    assert_allclose(Water.Tsat(2e5), 393., rtol=1e-2)
    assert_allclose(Water.Tsat(2e5, Tguess=390.), 393., rtol=1e-2)
    Water.reset_combustion_data(formula={'H': 2, 'O': 1})
    Water.reset_combustion_data(formula='H2O')
    assert np.isfinite(Water.LHV)
    assert Water.Stiel_Polar is not None
    Water.reset_free_energies()
    Water._eos = None
    Water.reset_free_energies()
    assert Water.H('l', 298.15) == 0
    Water.at_state('l')
    Water.at_state('l')
    with pytest.raises(TypeError):
        Water.at_state('g')
    with pytest.raises(ValueError):
        tmo.Chemical('Ethanol').at_state('')
    Ethanol = tmo.Chemical('Ethanol')
    gas = Ethanol.at_state('g', copy=True)
    assert gas.locked_state == 'g' and Ethanol.locked_state is None
    gas.show()
    Water.show()
    Water.set_method('POLY_FIT')

def test_chemical_reset():
    Water = tmo.Chemical('Water')
    with pytest.raises(RuntimeError):
        Water.reset('7732-18-5', formula='CO2')
    with pytest.raises(RuntimeError):
        Water.reset('7732-18-5', MW=100.)
    Water.reset('7732-18-5', synonyms='H2O-coverage')
    assert_allclose(Water.MW, 18.015, rtol=1e-2)
    X = tmo.Chemical('X', search_db=False)
    X.reset('64-17-5', synonyms=['X-coverage'])
    assert_allclose(X.MW, 46., rtol=1e-2)
    Glycerol = tmo.Chemical('Glycerol')
    assert Glycerol.Dortmund

def test_chemical_default():
    X = tmo.Chemical.blank('X', formula='C6H12O6')
    missing = X.default(['V', 'mu', 'kappa', 'Cn', 'Hf', 'HHV', 'LHV', 'combustion',
                         'phase_ref', 'eos', 'epsilon', 'sigma', 'Hfus', 'Sfus'])
    assert not missing.intersection(['Hf', 'Cn'])
    assert np.isfinite(X.Hf)
    X = tmo.Chemical.blank('X', formula='C6H12O6', Hf=-1e6)
    X.default(['LHV', 'HHV', 'combustion'])
    assert np.isfinite(X.LHV)
    X = tmo.Chemical.blank('X', formula='Xe', Hf=-1e6)
    X.default(['LHV', 'HHV', 'combustion', 'Hf'])
    Y = tmo.Chemical.blank('Y', phase='l')
    Y.default(['MW', 'V', 'mu', 'kappa', 'Cn'])
    assert_allclose(Y.V(300., 101325.), 1e-6, rtol=1e-2)

def test_chemical_copy_models_from():
    Water = tmo.Chemical('Water', cache=True)
    X = tmo.Chemical.blank('X', phase_ref='l')
    with pytest.raises(ValueError):
        X.copy_models_from(Water, ['not a model'])
    X.copy_models_from(Water, ['Cn', 'Psat', 'V'])
    assert_allclose(X.Psat(373.15), 101325, rtol=2e-2)
    Y = tmo.Chemical.blank('Y', phase_ref='l')
    Y.copy_models_from(Water)
    assert_allclose(Y.Cn('l', 300.), Water.Cn('l', 300.), rtol=1e-2)
    Z = tmo.Chemical.blank('Z', phase='l')
    Z.copy_models_from(Water, ['Cn', 'Psat'])
    assert_allclose(Z.Cn(300.), Water.Cn('l', 300.), rtol=1e-2)
    W = tmo.Chemical.blank('W', phase_ref='l')
    W.copy_models_from(Z, ['Cn'])
    assert_allclose(W.Cn('l', 300.), Water.Cn('l', 300.), rtol=1e-2)

def test_chemical_extras():
    from thermosteam import _chemical
    Cellulose = tmo.Chemical('Cellulose')
    assert Cellulose.MW > 100
    Water = tmo.Chemical('Water', cache=True)
    metadata = tmo.chemicals.identifiers.pubchem_db.search('Water')
    assert tmo.Chemical('Water', metadata=metadata).CAS == Water.CAS
    with pytest.raises(Exception):
        _chemical.raise_helpful_handle_error('V', None)
    Liquid = tmo.Chemical('Water', phase='l')
    Liquid.Hfus = 6000.
    assert Liquid.Hfus == 6000.
    try:
        import CoolProp
    except: 
        return
    Water = tmo.Chemical('Water')
    Water.use_coolprop_free_energies()
    assert np.isfinite(Water.H('l', 350., 101325.))

def test_elements_and_phase_handles():
    from thermosteam.chemicals.elements import mass_fractions
    fractions = mass_fractions({'D': 2, 'T': 1, 'Ash': 1, 'O': 1})
    assert_allclose(sum(fractions.values()), 1, rtol=1e-2)
    with pytest.raises(ValueError):
        mass_fractions({'Xx': 1}, MW=1.)
    from thermosteam.base.phase_handle import MockPhaseHandle
    Water = tmo.Chemical('Water', cache=True)
    handle = MockPhaseHandle('Cn', Water.Cn.l)
    assert handle.S is handle.L is handle.s is handle.l is handle.g
    assert len(list(handle)) == 3 and handle

if __name__ == '__main__':
    test_chemical_creation_options()
    test_chemical_setters()
    test_chemical_methods()
    test_chemical_reset()
    test_chemical_default()
    test_chemical_copy_models_from()
    test_chemical_extras()
    test_elements_and_phase_handles()
