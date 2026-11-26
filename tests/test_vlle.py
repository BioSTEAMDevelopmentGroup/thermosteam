# -*- coding: utf-8 -*-
"""
Tests for vapor-liquid-liquid equilibrium (VLLE) at given temperature and
pressure. Reference phase fractions were obtained independently by global
minimization of the Gibbs free energy (multistart) using the same
thermodynamic models (identical liquids merged).
"""
import pytest
import numpy as np
import thermosteam as tmo
from numpy.testing import assert_allclose
from thermosteam.equilibrium.fugacities import LiquidFugacities, GasFugacities

P = 101325.

# (chemicals, flows, T, reference phase fractions [vapor, smaller liquid, larger liquid])
cases = [
    # Three phases
    (('Water', 'Ethanol', 'Octane'), (1, 0.5, 2), 353.0, (0.260611, 0.163187, 0.576202)),
    (('Water', 'Ethanol', 'Octane'), (2, 0.5, 1), 355.0, (0.362575, 0.221935, 0.415490)),
    (('Water', 'Ethanol', 'Octane'), (1, 1, 1), 350.5, (0.632351, 0.119351, 0.248298)),
    (('Water', 'Ethanol', 'Toluene'), (1, 0.7, 1.5), 349.0, (0.412071, 0.178227, 0.409702)),
    (('Water', 'Ethanol', 'Toluene'), (2, 0.3, 1), 352.0, (0.200720, 0.254327, 0.544954)),
    (('Water', 'Ethanol', 'Benzene'), (1, 0.5, 2), 339.0, (0.621315, 0.142741, 0.235944)),
    # Vapor and one liquid
    (('Water', 'Ethanol', 'Octane'), (1, 0.5, 2), 358.0, (0.531494, 0.0, 0.468505)),
    (('Water', 'Ethanol', 'Octane'), (1, 1, 1), 357.0, (0.871877, 0.0, 0.128123)),
    (('Water', 'Ethanol', 'Toluene'), (1, 0.7, 1.5), 355.0, (0.844275, 0.0, 0.155725)),
    # Two liquids (subcooled)
    (('Water', 'Ethanol', 'Octane'), (2, 0.5, 1), 350.0, (0.0, 0.324283, 0.675717)),
    (('Water', 'Ethanol', 'Toluene'), (2, 0.3, 1), 348.0, (0.0, 0.340078, 0.659922)),
    (('Water', 'Ethanol', 'Benzene'), (1, 0.5, 2), 326.0, (0.0, 0.323287, 0.676713)),
    (('Water', 'Ethanol', 'Benzene'), (1, 0.5, 2), 330.0, (0.0, 0.317780, 0.682220)),
    (('Water', 'Ethanol', 'Benzene'), (1, 0.5, 2), 337.0, (0.0, 0.308275, 0.691725)),
]

def vlle_stream(chemicals, flows, T):
    tmo.settings.set_thermo(chemicals, cache=True)
    s = tmo.Stream(None, **dict(zip(chemicals, flows)))
    s.phases = 'gLl'
    s.vlle(T=T, P=P)
    return s

def phase_fractions(s):
    total = s.F_mol
    liquids = sorted([s.imol['L'].sum() / total, s.imol['l'].sum() / total])
    return np.array([s.imol['g'].sum() / total, *liquids])

@pytest.mark.parametrize('chemicals, flows, T, reference', cases)
def test_vlle_phase_fractions(chemicals, flows, T, reference):
    s = vlle_stream(chemicals, flows, T)
    assert_allclose(s.mol, flows, rtol=1e-9) # Mass balance
    assert_allclose(phase_fractions(s), reference, atol=1e-4)

@pytest.mark.parametrize('chemicals, flows, T, reference', cases)
def test_vlle_equal_fugacities(chemicals, flows, T, reference):
    s = vlle_stream(chemicals, flows, T)
    chemicals = s.chemicals
    fugacities = {
        'g': GasFugacities(chemicals),
        'L': LiquidFugacities(chemicals),
        'l': LiquidFugacities(chemicals),
    }
    present = [i for i in 'gLl' if s.imol[i].sum() > 1e-9 * s.F_mol]
    log_fugacities = {}
    for i in present:
        mol = s.imol[i]
        x = mol / mol.sum()
        log_fugacities[i] = np.log(fugacities[i](x, T, P))
    reference_phase, *other_phases = present
    # Three-phase results are converged by the simultaneous correction;
    # other results are from the (early terminated) sequential iteration.
    atol = 1e-6 if len(present) == 3 else 1e-3
    for i in other_phases:
        mask = (s.imol[i] > 0) & (s.imol[reference_phase] > 0)
        assert_allclose(log_fugacities[i][mask], log_fugacities[reference_phase][mask], atol=atol)
    if 'L' in present and 'l' in present: # Liquids must be distinct
        xL = s.imol['L'] / s.imol['L'].sum()
        xl = s.imol['l'] / s.imol['l'].sum()
        assert np.abs(xL - xl).max() > 1e-2

def test_vlle_subcooled_liquid_split_depends_on_temperature():
    # Regression: the LLE cache was reused across temperatures
    chemicals = ('Water', 'Ethanol', 'Benzene')
    flows = (1, 0.5, 2)
    low = phase_fractions(vlle_stream(chemicals, flows, 326.))
    high = phase_fractions(vlle_stream(chemicals, flows, 337.))
    assert np.abs(low - high).max() > 1e-2

def test_vlle_dew_point_with_second_liquid():
    # Regression: the dew point must account for the liquid that forms first
    # (water-rich), not a metastable hydrocarbon-rich liquid (~348 K)
    tmo.settings.set_thermo(['Water', 'Ethanol', 'Toluene'], cache=True)
    s = tmo.Stream(None, Water=2, Ethanol=0.3, Toluene=1)
    s.phases = 'gLl'
    s.vlle._setup(None)
    assert_allclose(s.vlle.dew_point_at_P(P), 359.9856, atol=1e-2)
    s.vlle(T=352., P=P) # Below the dew point: not all vapor
    assert s.imol['g'].sum() / s.F_mol < 0.5

if __name__ == '__main__':
    for case in cases:
        test_vlle_phase_fractions(*case)
        test_vlle_equal_fugacities(*case)
    test_vlle_subcooled_liquid_split_depends_on_temperature()
    test_vlle_dew_point_with_second_liquid()
