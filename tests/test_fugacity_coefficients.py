# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
Tests for fugacity coefficient values, compared against thermo's mixture
cubic equations of state.
"""
import pytest
import numpy as np
import thermosteam as tmo
from thermosteam import equilibrium as eq
from thermo.interaction_parameters import IPDB
from numpy.testing import assert_allclose

# %% Utilities

def get_eos_classes():
    return [
        i for i in eq.fugacity_coefficient_classes
        if i.__name__ != 'PSRKFugacityCoefficients' # Requires extra arguments
    ]

def get_cases():
    return [
        # Chemicals, molar fractions, temperature, pressure, phase
        (['Water', 'Ethanol'], [0.4, 0.6], 300., 101325., 'l'),
        (['Water', 'Ethanol'], [0.4, 0.6], 400., 101325., 'g'),
        (['Methane', 'Hexane'], [0.1, 0.9], 300., 5e6, 'l'),
        (['Methane', 'Ethane', 'Propane'], [0.7, 0.2, 0.1], 300., 2e6, 'g'),
    ]

def thermo_fugacity_coefficients(Phi, chemicals, x, T, P, phase):
    if Phi.chemsep_db is None:
        kijs = None
    else:
        kijs = IPDB.get_ip_asymmetric_matrix(Phi.chemsep_db, [i.CAS for i in chemicals], 'kij')
    eos = Phi.EOS(
        T=T, P=P, zs=x, kijs=kijs,
        Tcs=[i.Tc for i in chemicals],
        Pcs=[i.Pc for i in chemicals],
        omegas=[i.omega for i in chemicals],
        only_l=phase == 'l', only_g=phase == 'g',
    )
    # Use the other root if not available (e.g., ideal gas has no liquid root)
    Z = getattr(eos, 'Z_' + phase, None)
    if Z is None: return 1.
    return np.exp(eos.fugacity_coefficients(Z))

# %% Tests

def test_eos_fugacity_coefficients():
    for Phi in get_eos_classes():
        for IDs, x, T, P, phase in get_cases():
            chemicals = tmo.Chemicals(IDs, cache=True)
            phi = Phi(chemicals, phase)
            ref = thermo_fugacity_coefficients(Phi, chemicals, x, T, P, phase)
            assert_allclose(phi(np.array(x), T, P), ref, rtol=1e-10, err_msg=Phi.__name__)
            assert_allclose(phi.f(np.array(x), T, P, *phi.args), ref, rtol=1e-10, err_msg=Phi.__name__)

def test_eos_fugacity_coefficients_after_updating_conditions():
    # Updates of composition, temperature, and pressure (using the cached
    # EOS object) give the same values as a new EOS object
    for Phi in get_eos_classes():
        for IDs, x, T, P, phase in get_cases():
            chemicals = tmo.Chemicals(IDs, cache=True)
            phi = Phi(chemicals, phase)
            phi(np.array(x), T, P)
            x2 = [i * j for i, j in zip(x, np.linspace(1, 2, len(x)))]
            x2 = [i / sum(x2) for i in x2]
            T2 = T * 1.01
            P2 = P * 1.1
            ref = thermo_fugacity_coefficients(Phi, chemicals, x2, T2, P2, phase)
            assert_allclose(phi(np.array(x2), T2, P2), ref, rtol=1e-10, err_msg=Phi.__name__)

def test_fugacity_coefficients_vapor_liquid_consistency():
    # At the bubble point of a mixture described entirely by the
    # Peng-Robinson EOS, liquid and vapor fugacities are equal
    chemicals = tmo.Chemicals(['Methane', 'Propane'], cache=True)
    tmo.settings.set_thermo(tmo.Thermo(chemicals, pkg='Peng Robinson'))
    s = tmo.Stream(None, Methane=0.2, Propane=0.8)
    bp = s.bubble_point_at_T(250.)
    phi_l = eq.PR78FugacityCoefficients(chemicals, 'l')
    phi_g = eq.PR78FugacityCoefficients(chemicals, 'g')
    f_l = bp.z * phi_l(bp.z, bp.T, bp.P)
    f_g = bp.y * phi_g(bp.y, bp.T, bp.P)
    assert_allclose(f_l, f_g, rtol=1e-3)

def test_ideal_fugacity_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    x = np.array([0.5, 0.5])
    for phase in 'lg':
        phi = eq.IdealFugacityCoefficients(chemicals, phase)
        assert phi(x, 300., 101325.) == 1.
        assert phi.f(x, 300., 101325., *phi.args) == 1.
    with pytest.raises(ValueError):
        eq.IdealFugacityCoefficients(chemicals, 'x')

if __name__ == '__main__':
    test_eos_fugacity_coefficients()
    test_eos_fugacity_coefficients_after_updating_conditions()
    test_fugacity_coefficients_vapor_liquid_consistency()
    test_ideal_fugacity_coefficients()
