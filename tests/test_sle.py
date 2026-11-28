# -*- coding: utf-8 -*-
"""
Tests for solid-liquid equilibrium (SLE) of simple eutectic systems.
Consistency tests check equilibrium conditions, mass and energy balances.
Benchmark tests compare against experimental data compiled in the IUPAC-NIST
Solubility Database (NIST SRD 106) and peer-reviewed literature. Unless
otherwise stated, the default property package (modified UNIFAC) and a heat
capacity of fusion of zero are used.
"""
import pytest
import numpy as np
import thermosteam as tmo
from math import log, exp
from numpy.testing import assert_allclose
from chemicals import solubility_eutectic
from thermosteam.constants import R
from thermosteam.equilibrium import (
    SLE, log_ideal_solubility, IdealActivityCoefficients,
)

P = 101325.

def sle_stream(chemicals, flows, Gamma=None, **kwargs):
    if Gamma is None:
        tmo.settings.set_thermo(chemicals, cache=True)
    else:
        tmo.settings.set_thermo(chemicals, Gamma=Gamma, cache=True)
    s = tmo.Stream(None, **dict(zip(chemicals, flows)))
    if kwargs: s.sle(**kwargs)
    return s

def liquid_mol_fraction(s, ID):
    return s.imol['l', ID] / s.imol['l'].sum()

def assert_equilibrium(s, tol=1e-8):
    # Saturated chemicals: x * gamma = psi; others: x * gamma <= psi
    T = s.T
    liquid = s.imol['l'].to_array()
    solid = s.imol['s'].to_array()
    chemicals = s.chemicals
    index = [i for i, j in enumerate(liquid) if j > 0.]
    x = liquid[index] / liquid[index].sum()
    eq_chemicals = [chemicals.tuple[i] for i in index]
    gamma = s.thermo.Gamma(eq_chemicals)(x, T)
    for i, chemical, xi, gi in zip(index, eq_chemicals, x, gamma):
        if not (chemical.Tm and chemical.Hfus): continue
        d = np.log(xi * gi) - log_ideal_solubility(chemical, T)
        if solid[i] > 0.:
            assert abs(d) < tol, (chemical.ID, d)
        else:
            assert d < tol, (chemical.ID, d)

# %% Consistency

equilibrium_cases = [
    (('Toluene', 'Naphthalene'), (1, 1), 298.15),
    (('Toluene', 'Naphthalene'), (1, 0.01), 179.1),
    (('Ethanol', 'Naphthalene'), (1, 1), 298.15),
    (('Cyclohexane', 'Naphthalene'), (1, 1), 298.15),
    (('Benzene', 'Naphthalene', 'Biphenyl'), (1, 1, 1), 300.),
    (('Benzene', 'Naphthalene', 'Biphenyl'), (1, 1, 1), 260.),
]

@pytest.mark.parametrize('chemicals, flows, T', equilibrium_cases)
def test_sle_equilibrium_conditions(chemicals, flows, T):
    s = sle_stream(chemicals, flows, T=T)
    assert_allclose(s.mol, flows) # Mass balance
    assert (s.imol['s'].to_array() >= 0).all()
    assert (s.imol['l'].to_array() >= 0).all()
    assert_equilibrium(s)

energy_balance_cases = [
    (('Toluene', 'Naphthalene'), (1, 1), 298.15),
    (('Toluene', 'Naphthalene'), (1, 1), 150.),
    (('Cyclohexane', 'Naphthalene'), (1, 1), 298.15),
    (('Benzene', 'Naphthalene', 'Biphenyl'), (1, 1, 1), 280.),
]

@pytest.mark.parametrize('chemicals, flows, T', energy_balance_cases)
def test_sle_energy_balance(chemicals, flows, T):
    # Solving at the enthalpy of a solution at T recovers T and phases
    s = sle_stream(chemicals, flows, T=T)
    H = s.H
    liquid = s.imol['l'].to_array()
    s.sle(T=T + 30)
    s.sle(H=H)
    assert_allclose(s.H, H, rtol=1e-6)
    assert_allclose(s.T, T, atol=1e-5)
    assert_allclose(s.imol['l'].to_array(), liquid, rtol=1e-5, atol=1e-8)

def test_sle_duty_specification():
    s = sle_stream(['Toluene', 'Naphthalene'], [1, 1], T=298.15)
    H = s.H
    s.sle(T=320)
    s.sle(Q=H - s.H)
    assert_allclose(s.T, 298.15, atol=1e-5)

def test_sle_pure_chemical():
    s = sle_stream(['Tetradecanol'], [30], T=300)
    assert s.imol['s', 'Tetradecanol'] == 30
    s.sle(T=320)
    assert s.imol['l', 'Tetradecanol'] == 30
    Tm = s.chemicals.Tetradecanol.Tm
    H_solid = s.mixture.H('s', s.mol, Tm, P)
    H_liquid = s.mixture.H('l', s.mol, Tm, P)
    s.sle(H=0.25 * H_liquid + 0.75 * H_solid) # At melting point
    assert_allclose(s.T, Tm, atol=1e-6)
    assert_allclose(s.imol['l', 'Tetradecanol'], 7.5, rtol=1e-6)

def test_sle_eutectic_plateau():
    # At the eutectic point (an invariant point), enthalpy specifications
    # between the completely solid and the saturated liquid states result in
    # the eutectic temperature with varying solid fraction
    s = sle_stream(['Toluene', 'Naphthalene'], [10, 1])
    T_eutectic = s.sle.T_solidus()
    dT = 1e-3
    s.sle(T=T_eutectic - dT)
    assert s.sle.solid_fraction == 1.
    H_solid = s.mixture.xH(s.imol, T_eutectic, P)
    s.sle(T=T_eutectic + dT)
    assert s.sle.solid_fraction < 1.
    H_liquid = s.mixture.xH(s.imol, T_eutectic, P)
    solid_fractions = []
    for f in np.linspace(0.1, 0.9, 5):
        H = H_solid + f * (H_liquid - H_solid)
        s.sle(H=H)
        assert_allclose(s.H, H, rtol=1e-9)
        assert_allclose(s.T, T_eutectic, atol=dT)
        solid_fractions.append(s.sle.solid_fraction)
    assert (np.diff(solid_fractions) < 0).all()

def test_sle_specified_solubility():
    s = sle_stream(['Water', 'Glucose'], [10, 1], T=298.15, solute='Glucose', solubility=0.0833)
    assert_allclose(liquid_mol_fraction(s, 'Glucose'), 0.0833)
    assert s.imol['s', 'Water'] == 0.

def test_sle_specified_solutes():
    # Only the specified solute may crystallize
    s = sle_stream(['Water', 'Glucose'], [10, 3], Gamma=IdealActivityCoefficients,
                   T=250., solute='Glucose')
    assert s.imol['s', 'Water'] == 0.
    assert s.imol['s', 'Glucose'] > 0.
    s.sle(T=250.)
    assert s.imol['s', 'Water'] > 0.

def test_sle_nonequilibrium_chemicals():
    # Chemicals not in equilibrium are left as is
    tmo.settings.set_thermo(['Water', 'Glucose', 'Cellulose'], cache=True)
    s = tmo.Stream(None, Water=10, Glucose=3, Cellulose=1)
    s.sle(T=298.15)
    assert_allclose(s.imol['Cellulose'], 1)

def test_sle_heat_capacity_of_fusion():
    # Constant heat capacity of fusion agrees with chemicals' implementation
    naphthalene = tmo.Chemical('Naphthalene')
    T = 298.15
    for dCp in (0., 20., 50.):
        assert_allclose(
            exp(log_ideal_solubility(naphthalene, T, dCp)),
            solubility_eutectic(T, naphthalene.Tm, naphthalene.Hfus, dCp, 0., 1.),
            rtol=1e-6, # Slightly different gas constants
        )
    s = sle_stream(['Benzene', 'Naphthalene'], [1, 10])
    s.phases = ('s', 'l')
    sle = SLE(s.imol, s.thermal_condition, thermo=s.thermo, dCp={'Naphthalene': 20.})
    sle(T=T)
    x_dCp = liquid_mol_fraction(s, 'Naphthalene')
    sle.dCp = None
    sle(T=T)
    x = liquid_mol_fraction(s, 'Naphthalene')
    assert x_dCp > x # Positive heat capacity of fusion increases solubility

def test_sle_ideal_freezing_point_depression():
    # Analytical freezing point of water (ice is a pure solid) in an ideal
    # solution with glucose (not saturated at the freezing point):
    # ln(x_water) = Hfus / R * (1 / Tm - 1 / T)
    s = sle_stream(['Water', 'Glucose'], [0.98, 0.02], Gamma=IdealActivityCoefficients)
    water = s.chemicals.Water
    T = 1. / (1. / water.Tm - R * log(0.98) / water.Hfus)
    assert_allclose(s.sle.T_liquidus(), T, atol=1e-6)
    s.sle(T=T - 1.)
    assert s.sle.solids == ('Water',)

# %% Benchmarks

# Kravchenko (1939), NIST SRD 106 sysID 59_126: (T [K], naphthalene mol fraction)
benzene_naphthalene_liquidus = [
    (276.5, 0.031), (272.2, 0.095), (277.2, 0.169), (290.4, 0.248),
    (301.7, 0.332), (312.8, 0.428), (322.2, 0.531), (331.4, 0.648),
    (340.2, 0.776), (349.7, 0.921),
]

@pytest.mark.parametrize('T, x', benzene_naphthalene_liquidus)
def test_benzene_naphthalene_liquidus(T, x):
    s = sle_stream(['Benzene', 'Naphthalene'], [1 - x, x])
    assert_allclose(s.sle.T_liquidus(), T, atol=1.)

# (chemicals, eutectic composition, eutectic temperature [K], tolerance [K])
eutectic_points = [
    # Kravchenko (1939), NIST SRD 106 sysID 59_126
    (('Benzene', 'Naphthalene'), (0.866, 0.134), 269.6, 0.5),
    # Kravchenko (1939), NIST SRD 106 sysID 59_140
    (('m-Xylene', 'Naphthalene'), (0.964, 0.036), 224.2, 1.),
    # Rice, Fu & Suuberg, J. Chem. Eng. Data 2010, 55, 3598
    (('Naphthalene', 'Biphenyl'), (0.442, 0.558), 312.4, 2.),
]

@pytest.mark.parametrize('chemicals, z, T, atol', eutectic_points)
def test_eutectic_temperature(chemicals, z, T, atol):
    s = sle_stream(chemicals, z)
    assert_allclose(s.sle.T_solidus(), T, atol=atol)

# Naphthalene solubility at ~298 K: (solvent, mol fraction, relative tolerance)
naphthalene_solubility = [
    ('Benzene', 0.292, 0.05), # Chang (1969), sysID 59_123
    ('Toluene', 0.287, 0.05), # Cui et al., J. Chem. Eng. Data 2009, 54, 1065
    ('Acetone', 0.2185, 0.05), # Cui et al., J. Chem. Eng. Data 2009, 54, 1065
    ('Cyclohexane', 0.1487, 0.10), # Heric & Posey, J. Chem. Eng. Data 1964, 9, 35
    ('Hexane', 0.119, 0.15), # Domanska (1981), sysID 59_95
    ('Ethanol', 0.032, 0.10), # Domanska (1981), sysID 59_198
    ('Methanol', 0.0235, 0.15), # Dickhut et al., J. Chem. Eng. Data 1989, 34, 438
]

@pytest.mark.parametrize('solvent, x, rtol', naphthalene_solubility)
def test_naphthalene_solubility(solvent, x, rtol):
    s = sle_stream([solvent, 'Naphthalene'], [1, 10], T=298.15)
    assert s.sle.solids == ('Naphthalene',)
    assert_allclose(liquid_mol_fraction(s, 'Naphthalene'), x, rtol=rtol)

# Alves, Silva & Giulietti, J. Chem. Eng. Data 2007, 52, 2166:
# (T [K], glucose mol fraction). Aqueous glucose solutions are nearly ideal,
# so the ideal solubility (anhydrous glucose melting properties) is used; 
# UNIFAC-based models overestimate the activity coefficient of glucose. 
# Note that alpha-glucose monohydrate is the stable solid below ~50 C, so 
# agreement within 10% is expected only from the near-ideality of the system.
glucose_solubility = [
    (283.1, 0.0635), (293.15, 0.0839), (303.15, 0.1028),
    (313.15, 0.1319), (323.15, 0.1700), (333.15, 0.2224),
]

@pytest.mark.parametrize('T, x', glucose_solubility)
def test_glucose_solubility(T, x):
    s = sle_stream(['Water', 'Glucose'], [1, 1], Gamma=IdealActivityCoefficients, T=T)
    assert s.sle.solids == ('Glucose',)
    assert_allclose(liquid_mol_fraction(s, 'Glucose'), x, rtol=0.1)


def test_plot_sle_binary_phase_envelope():
    # Eutectic point of benzene-naphthalene: 269.6 K at a naphthalene mol
    # fraction of 0.134 (Kravchenko, 1939; NIST SRD 106 sysID 59_126)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from thermosteam.equilibrium import plot_sle_binary_phase_envelope
    tmo.settings.set_thermo(['Naphthalene', 'Benzene'], cache=True)
    T, x = plot_sle_binary_phase_envelope(['Naphthalene', 'Benzene'], N=10)
    plt.close('all')
    assert_allclose(T, 269.6, atol=0.5)
    assert_allclose(x, 0.134, atol=0.01)

if __name__ == '__main__':
    for case in equilibrium_cases:
        test_sle_equilibrium_conditions(*case)
    for case in energy_balance_cases:
        test_sle_energy_balance(*case)
    test_sle_duty_specification()
    test_sle_pure_chemical()
    test_sle_eutectic_plateau()
    test_sle_specified_solubility()
    test_sle_specified_solutes()
    test_sle_nonequilibrium_chemicals()
    test_sle_heat_capacity_of_fusion()
    test_sle_ideal_freezing_point_depression()
    for case in benzene_naphthalene_liquidus:
        test_benzene_naphthalene_liquidus(*case)
    for case in eutectic_points:
        test_eutectic_temperature(*case)
    for case in naphthalene_solubility:
        test_naphthalene_solubility(*case)
    for case in glucose_solubility:
        test_glucose_solubility(*case)
    test_plot_sle_binary_phase_envelope()
