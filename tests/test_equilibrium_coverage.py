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
import numpy as np
import thermosteam as tmo
from thermosteam import equilibrium as eq
from numpy.testing import assert_allclose

# %% Utilities

def check_phase_fractions(stream):
    V = stream.vapor_fraction
    assert 0. <= V <= 1.
    assert (stream.mol >= 0.).all()

def check_mass_balance(stream, mol):
    assert_allclose(stream.mol, mol, rtol=1e-3, atol=1e-6)

def water_ethanol_stream(T=298.15, P=101325., **kwargs):
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    if not kwargs: kwargs = dict(Water=1., Ethanol=1.)
    return tmo.Stream(None, T=T, P=P, **kwargs)

def water_ethanol_octane_stream(T=298.15, P=101325.):
    tmo.settings.set_thermo(['Water', 'Ethanol', 'Octane'], cache=True)
    return tmo.Stream(None, Water=10., Ethanol=1., Octane=10., T=T, P=P)

# %% Fugacity coefficients, Poyinting correction factors, and fugacities

def test_ideal_fugacity_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    phi = eq.IdealFugacityCoefficients(chemicals, 'g')
    assert phi.phase == 'g'
    assert [i.ID for i in phi.chemicals] == ['Water', 'Ethanol']
    assert phi(np.array([0.5, 0.5]), 300., 101325.) == 1.
    assert 'IdealFugacityCoefficients' in repr(phi)
    with pytest.raises(ValueError):
        eq.IdealFugacityCoefficients(chemicals, 'x')

def test_eos_fugacity_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    z = np.array([0.5, 0.5])
    for cls in eq.fugacity_coefficient_classes:
        if cls.__name__ == 'PSRKFugacityCoefficients': continue # Requires extra arguments
        for phase in ('g', 'l'):
            phi = cls(chemicals, phase)
            values = phi(z, 355., 101325.)
            assert np.isfinite(values).all()
            assert (np.asarray(values) > 0.).all()
    phi = eq.PRFugacityCoefficients(chemicals, 'g')
    assert phi.phase == 'g'
    assert len(phi.chemicals) == 2
    assert 'PRFugacityCoefficients' in repr(phi)
    # Ideal gas-like behavior at low pressure
    assert_allclose(phi(z, 355., 101325.), 1., rtol=5e-2)
    # Second call uses fast EOS update
    assert_allclose(phi(z, 360., 101325.), 1., rtol=5e-2)
    with pytest.raises(ValueError):
        eq.PRFugacityCoefficients(tmo.Chemicals(['Water', 'Methanol'], cache=True), 'x')

def test_poyinting_correction_factors():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    mock = eq.MockPoyintingCorrectionFactors(chemicals)
    assert mock(300., 101325.) == 1.
    assert 'MockPoyintingCorrectionFactors' in repr(mock)
    pcf = eq.IdealGasPoyintingCorrectionFactors(chemicals)
    assert [i.ID for i in pcf.chemicals] == ['Water', 'Ethanol']
    cls, args = pcf.__reduce__()
    assert cls is eq.IdealGasPoyintingCorrectionFactors
    values = pcf(355., 1e6)
    assert (values >= 1.).all() and (values < 1.1).all()
    Psats = np.array([i.Psat(355.) for i in chemicals])
    assert_allclose(pcf(355., 1e6, Psats), values, rtol=1e-6)
    # No correction below the saturation pressure
    assert_allclose(pcf(355., 100., Psats), 1.)

def test_fugacities():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    z = np.array([0.5, 0.5])
    T = 355.
    P = 101325.
    # Activity coefficient model
    thermo = tmo.Thermo(chemicals)
    f_l = eq.LiquidFugacities(chemicals, thermo)
    assert 'LiquidFugacities' in repr(f_l)
    assert_allclose(f_l(z, T, P), z * f_l.unweighted(z, T, P))
    assert_allclose(f_l(z, T, P, reduce=True) * 0, 0)
    # Equation of state model (with and without poyinting correction)
    for PCF in (None, eq.IdealGasPoyintingCorrectionFactors):
        thermo = tmo.Thermo(chemicals, pkg='Peng Robinson', PCF=PCF)
        f_l = eq.LiquidFugacities(chemicals, thermo)
        f = f_l(z, T, P)
        assert_allclose(f, z * f_l.unweighted(z, T, P))
        assert_allclose(f_l(z, T, P, reduce=True) * P, f, rtol=0.05)
        f_g = eq.GasFugacities(chemicals, thermo)
        assert 'GasFugacities' in repr(f_g)
        assert_allclose(f_g(z, T, P), z * f_g.unweighted(z, T, P))
        assert_allclose(f_g(z, T, P, reduce=True) * P, f_g(z, T, P))
        fs = eq.Fugacities(chemicals, thermo)
        assert_allclose(fs('l', z, T, P), f)
        assert_allclose(fs.unweighted('g', z, T, P), f_g.unweighted(z, T, P))
    # Gas fugacities are close to partial pressures
    assert_allclose(f_g(z, T, P), z * P, rtol=0.05)
    # Both models active is invalid
    f_l.gamma = eq.DortmundActivityCoefficients(chemicals)
    with pytest.raises(RuntimeError):
        f_l(z, T, P)
    with pytest.raises(RuntimeError):
        f_l.unweighted(z, T, P)

# %% Activity coefficients

def test_activity_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    x = np.array([0.5, 0.5])
    T = 350.
    ideal = eq.IdealActivityCoefficients(chemicals)
    assert_allclose(ideal(x, T), 1.)
    assert 'IdealActivityCoefficients' in repr(ideal)
    for cls in (eq.UNIFACActivityCoefficients, eq.DortmundActivityCoefficients):
        gamma = cls(chemicals)
        assert gamma is cls(chemicals) # Cached
        values = gamma(x, T)
        assert np.isfinite(values).all() and (values > 0.).all()
        cls_, args = gamma.__reduce__()
        assert cls_ is cls
        assert cls.__name__ in repr(gamma)
    # Chemicals without functional groups behave ideally
    gamma = eq.DortmundActivityCoefficients(tmo.Chemicals(['Water'], cache=True))
    assert isinstance(gamma, eq.IdealActivityCoefficients)

def test_NIST_activity_coefficients():
    Water, Ethanol = chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    Ethanol.NIST.set_group_counts_by_name({'CH3':1, 'CH2':1, 'OH prim':1})
    Water.NIST.set_group_counts_by_name({'H2O':1})
    gamma = eq.NISTActivityCoefficients(chemicals)
    x = np.array([0.5, 0.5])
    values = gamma(x, 350.)
    assert (values > 1.).all() and (values < 3.).all()
    assert_allclose(gamma.activity_coefficients(x, 350.), values, rtol=1e-2)

# %% Thermo-package constructors (flash package)

def test_flash_package_constructors():
    import thermo as tm
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    data = tmo.ChemicalData(chemicals)
    for version in (0, 1, 2):
        GE = tm.UNIFAC.from_data(data, version=version)
        assert GE is not None
    with pytest.raises(RuntimeError):
        tm.UNIFAC.from_data(data, version=3)
    assert tm.CEOSGas.from_data(data) is not None
    assert tm.CEOSLiquid.from_data(data) is not None
    assert tm.IdealGas.from_data(data) is not None

# %% Tangent plane stability analysis

def test_tangent_plane_stability_analysis():
    tmo.settings.set_thermo(['Water', 'Octane'], cache=True)
    chemicals = tmo.settings.chemicals
    TPSA = eq.TangentPlaneStabilityAnalysis('lL', chemicals)
    z = np.array([0.5, 0.5])
    report = TPSA(z, 300., 101325.)
    assert report.unstable
    assert report.tpd < 0
    assert_allclose(report.candidate.sum(), 1., rtol=1e-6)
    # Early exit when samples are given
    report = TPSA(z, 300., 101325., samples=[np.array([0.99, 0.01])])
    assert report.unstable and report.early_exit
    # Liquid vs vapor phases
    TPSA = eq.TangentPlaneStabilityAnalysis('lg', chemicals)
    report = TPSA(z, 300., 101325., 'l', 'g')
    assert np.isfinite(report.tpd)

def test_edge_points_simplex():
    from thermosteam.equilibrium.tangential_plane_stability import edge_points_simplex_masked
    points = edge_points_simplex_masked(np.array([0.98, 0.01, 0.01]), points_per_edge=4)
    assert points.shape[1] == 3
    assert_allclose(points.sum(axis=1), 1.)
    assert (points > 0).all()

def test_lle_tangential_plane_analysis():
    from thermosteam.equilibrium.tangential_plane_stability import lle_tangential_plane_analysis
    chemicals = tmo.Chemicals(['Water', 'Octane'], cache=True)
    gamma = eq.DortmundActivityCoefficients(chemicals)
    z = np.array([0.5, 0.5])
    report = lle_tangential_plane_analysis(gamma, z, 300., 101325., samples=[np.array([0.99, 0.01])])
    assert report.unstable and report.early_exit
    report = lle_tangential_plane_analysis(gamma, z, 300., 101325., samples=[z])
    assert report.unstable

# %% Bubble and dew points

def test_bubble_point_values():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    BP = eq.BubblePoint()
    assert 'BubblePoint' in repr(BP)
    z = np.array([0.5, 0.5])
    bp = BP(z, T=355.)
    assert 'BubblePointValues' in repr(bp)
    assert_allclose(bp.x, z)
    assert_allclose(bp.K, bp.y / z)
    assert 9e4 < bp.P < 1.3e5
    bp = BP(z, P=101325.)
    assert 345. < bp.T < 360.
    with pytest.raises(ValueError):
        BP(z, T=355., P=101325.)
    with pytest.raises(ValueError):
        BP(z)
    with pytest.raises(ValueError):
        BP.solve_Ty(np.zeros(2), 101325.)
    with pytest.raises(ValueError):
        BP.solve_Py(np.zeros(2), 355.)
    # Single component
    T, y = BP.solve_Ty(np.array([1., 0.]), 101325.)
    assert_allclose(T, 373.12, rtol=1e-2)
    T, y = BP.solve_Ty(np.array([1., 0.]), 1e9) # Supercritical
    assert_allclose(T, BP.chemicals[0].Tc)
    P, y = BP.solve_Py(np.array([1., 0.]), 373.12)
    assert_allclose(P, 101325., rtol=1e-2)
    P, y = BP.solve_Py(np.array([1., 0.]), 1000.) # Supercritical
    assert_allclose(P, BP.chemicals[0].Pc)
    # Out of domain temperatures are clipped
    P, y = BP.solve_Py(z, 10.)
    assert P > 0.
    P, y = BP.solve_Py(z, 1e4)
    assert P > 0.

def test_bubble_point_lle():
    tmo.settings.set_thermo(['Water', 'Octane'], cache=True)
    BP = eq.BubblePoint()
    z = np.array([0.5, 0.5])
    T, y, alpha = BP.solve_Ty(z, 101325., lle=True, full=True)
    assert 340. < T < 373.
    assert_allclose(y.sum(), 1.)

def test_dew_point_values():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    DP = eq.DewPoint()
    assert 'DewPoint' in repr(DP)
    z = np.array([0.5, 0.5])
    dp = DP(z, T=355.)
    assert 'DewPointValues' in repr(dp)
    assert_allclose(dp.y, z)
    assert 8e4 < dp.P < 1.1e5
    dp = DP(z, P=101325.)
    assert 350. < dp.T < 365.
    # Using a guess
    T, x = DP.solve_Tx(z, 101325., guess=(dp.T, dp.x, dp.T + 1))
    assert_allclose(T, dp.T, rtol=1e-3)
    P, x = DP.solve_Px(z, 355., guess=(9e4, dp.x, 9e4 + 10))
    assert 8e4 < P < 1.1e5
    with pytest.raises(ValueError):
        DP(z, T=355., P=101325.)
    with pytest.raises(ValueError):
        DP(z)
    with pytest.raises(ValueError):
        DP.solve_Tx(np.zeros(2), 101325.)
    with pytest.raises(ValueError):
        DP.solve_Px(np.zeros(2), 355.)
    # Single component
    T, x = DP.solve_Tx(np.array([1., 0.]), 101325.)
    assert_allclose(T, 373.12, rtol=1e-2)
    T, x = DP.solve_Tx(np.array([1., 0.]), 1e9) # Supercritical
    assert_allclose(T, DP.chemicals[0].Tc)
    P, x = DP.solve_Px(np.array([1., 0.]), 373.12)
    assert_allclose(P, 101325., rtol=1e-2)
    P, x = DP.solve_Px(np.array([1., 0.]), 1000.) # Supercritical
    assert_allclose(P, DP.chemicals[0].Pc)

def test_bubble_and_dew_points_with_eos():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    z = np.array([0.5, 0.5])
    for pkg in ('Peng Robinson', 'Peng Robinson | Dortmund-UNIFAC'):
        for PCF in (None, eq.IdealGasPoyintingCorrectionFactors):
            thermo = tmo.Thermo(chemicals, pkg=pkg, PCF=PCF)
            BP = eq.BubblePoint(chemicals, thermo)
            DP = eq.DewPoint(chemicals, thermo)
            bp_T = BP(z, P=101325.).T
            dp_T = DP(z, P=101325.).T
            assert 340. < bp_T <= dp_T < 370.
            bp_P = BP(z, T=355.).P
            dp_P = DP(z, T=355.).P
            assert 8e4 < dp_P <= bp_P < 1.5e5
    thermo = tmo.Thermo(chemicals, pkg='Peng Robinson')
    BP = eq.BubblePoint(chemicals, thermo)
    with pytest.raises(NotImplementedError):
        BP.solve_Ty(z, 101325., lle=True)
    with pytest.raises(NotImplementedError):
        BP.solve_Py(z, 355., lle=True)

# %% VLE

def test_vle_binary_composition_specifications():
    s = water_ethanol_stream()
    s.vle(T=355., x=np.array([0.6, 0.4]))
    assert 9e4 < s.P < 1.3e5
    check_phase_fractions(s)
    s = water_ethanol_stream()
    s.vle(P=101325., x=np.array([0.6, 0.4]))
    assert 345. < s.T < 360.
    check_phase_fractions(s)
    s = water_ethanol_stream()
    s.vle(T=355., y=np.array([0.4, 0.6]))
    assert 9e4 < s.P < 1.3e5
    check_phase_fractions(s)
    s = water_ethanol_stream()
    s.vle(P=101325., y=np.array([0.4, 0.6]))
    assert 345. < s.T < 360.
    check_phase_fractions(s)
    # Infeasible lever rule
    s = water_ethanol_stream()
    with pytest.raises(tmo.exceptions.InfeasibleRegion):
        s.vle(P=101325., x=np.array([0.01, 0.99]))

def test_vle_single_component():
    s = water_ethanol_stream(Water=1.)
    s.vle(T=400., P=101325.)
    assert s.vapor_fraction == 1.
    s.vle(T=300., P=101325.)
    assert s.vapor_fraction == 0.
    s.vle(T=700., P=101325.) # Supercritical
    assert s.vapor_fraction == 1.
    s.vle(V=0.5, P=101325.)
    assert_allclose(s.T, 373.12, rtol=1e-3)
    assert_allclose(s.vapor_fraction, 0.5)
    # Enthalpy specifications
    s.vle(V=0.5, P=101325.)
    H = s.H
    S = s.S
    s.vle(H=H, P=101325.)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    s.vle(H=H + 1e5, P=101325.) # Superheated
    assert s.vapor_fraction == 1. and s.T > 374.
    s.vle(V=0.5, P=101325.)
    s.vle(H=H - 2e4, P=101325.) # Less vapor
    assert s.vapor_fraction < 0.5 and s.T < 374.
    # Entropy specifications
    s.vle(V=0.5, P=101325.)
    s.vle(S=S, P=101325.)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    s.vle(S=S + 100., P=101325.) # Superheated
    assert s.vapor_fraction == 1. and s.T > 374.
    # Temperature specifications
    s.vle(V=0.5, P=101325.)
    s.vle(H=H, T=373.12)
    assert_allclose(s.P, 101325., rtol=1e-2)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    s.vle(S=S, T=373.12)
    assert_allclose(s.P, 101325., rtol=1e-2)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    for kwargs in (dict(H=H + 1e5), dict(H=H - 1e5), dict(S=S + 100), dict(S=S - 100)):
        with pytest.raises(NotImplementedError):
            s.vle(T=373.12, **kwargs)

def test_vle_no_chemicals():
    tmo.settings.set_thermo(['Water', 'Ethanol', tmo.Chemical('O2', phase='g')], cache=True)
    s = tmo.Stream(None, O2=1., T=300.)
    H = s.H
    S = s.S
    s.vle(T=300., P=101325.)
    s.vle(H=H + 1e3, P=101325.)
    assert s.T > 300.
    s.vle(S=S, P=101325.)
    assert_allclose(s.T, 300., rtol=1e-2)
    with pytest.raises(tmo.exceptions.NoEquilibrium):
        s.vle(V=0.5, T=300.)
    with pytest.raises(tmo.exceptions.NoEquilibrium):
        s.vle(V=0.5, P=101325.)
    with pytest.raises(tmo.exceptions.NoEquilibrium):
        s.vle(H=H, T=300.)
    with pytest.raises(tmo.exceptions.NoEquilibrium):
        s.vle(S=S, T=300.)

def test_vle_binary():
    s = water_ethanol_stream()
    mol = s.mol.copy()
    s.vle(V=0.5, P=101325.)
    check_phase_fractions(s)
    check_mass_balance(s, mol)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    T = s.T
    H = s.H
    S = s.S
    s.vle(V=0.5, T=T)
    assert_allclose(s.P, 101325., rtol=1e-2)
    s.vle(V=0., T=T)
    assert s.vapor_fraction == 0.
    s.vle(V=1., T=T)
    assert s.vapor_fraction == 1.
    s.vle(V=0., P=101325.)
    assert s.vapor_fraction == 0.
    s.vle(V=1., P=101325.)
    assert s.vapor_fraction == 1.
    # Boil-up ratio and duty
    s.vle(B=1., P=101325.)
    assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
    s.vle(B=np.inf, P=101325.)
    assert s.vapor_fraction == 1.
    s.vle(Q=0., P=101325.)
    check_phase_fractions(s)
    # Energy specifications
    s.vle(H=H, P=101325.)
    assert_allclose(s.vapor_fraction, 0.5, rtol=2e-2)
    s.vle(S=S, P=101325.)
    assert_allclose(s.vapor_fraction, 0.5, rtol=2e-2)
    s.vle(H=H, T=T)
    assert_allclose(s.P, 101325., rtol=2e-2)
    s.vle(S=S, T=T)
    assert_allclose(s.P, 101325., rtol=2e-2)
    for kwargs in (dict(H=H + 1e6), dict(H=H - 1e6), dict(S=S + 1e3), dict(S=S - 1e3)):
        with pytest.raises(NotImplementedError):
            s.vle(T=T, **kwargs)
    check_mass_balance(s, mol)
    # Superheated and subcooled
    s.vle(H=H + 1e5, P=101325.)
    assert s.vapor_fraction == 1.
    s.vle(H=H - 5e4, P=101325.)
    assert s.vapor_fraction == 0.
    s.vle(S=S + 200., P=101325.)
    assert s.vapor_fraction == 1.
    s.vle(S=S - 100., P=101325.)
    assert s.vapor_fraction < 0.5
    check_mass_balance(s, mol)
    with pytest.raises(AssertionError):
        s.vle(T=T)

def test_vle_with_light_and_heavy_chemicals():
    O2 = tmo.Chemical('O2', phase='g')
    Glucose = tmo.Chemical('Glucose', phase='l', default=True)
    tmo.settings.set_thermo(['Water', 'Ethanol', O2, Glucose], cache=True)
    s = tmo.Stream(None, Water=10., Ethanol=2., O2=1., Glucose=1.)
    mol = s.mol.copy()
    for kwargs in (dict(V=0.5, P=101325.), dict(V=0.5, T=360.),
                   dict(V=0., P=101325.), dict(V=1., P=101325.),
                   dict(V=0., T=360.), dict(V=1., T=360.),
                   dict(T=360., P=101325.)):
        s.vle(**kwargs)
        check_phase_fractions(s)
        check_mass_balance(s, mol)
        assert s.imol['g', 'O2'] == 1.
        assert s.imol['l', 'Glucose'] == 1.
    s.vle(V=0.5, P=101325.)
    H = s.H
    S = s.S
    T = s.T
    s.vle(H=H, P=101325.)
    assert_allclose(s.T, T, rtol=1e-2)
    s.vle(S=S, P=101325.)
    assert_allclose(s.T, T, rtol=1e-2)
    s.vle(H=H, T=T)
    assert_allclose(s.P, 101325., rtol=5e-2)
    s.vle(S=S, T=T)
    assert_allclose(s.P, 101325., rtol=5e-2)
    check_mass_balance(s, mol)

def test_vle_supercritical_and_subcritical_exclusion():
    tmo.settings.set_thermo(['Water', 'Ethanol', 'CO2'], cache=True)
    s = tmo.Stream(None, Water=10., Ethanol=1., CO2=1.)
    mol = s.mol.copy()
    s.vle(T=320., P=101325.) # CO2 is supercritical
    assert s.imol['g', 'CO2'] == 1.
    check_mass_balance(s, mol)
    s.vle(T=320., P=1e7) # P above CO2 critical pressure
    check_phase_fractions(s)
    check_mass_balance(s, mol)

def test_vle_multicomponent():
    tmo.settings.set_thermo(['Water', 'Ethanol', 'Methanol'], cache=True)
    s = tmo.Stream(None, Water=10., Ethanol=2., Methanol=2.)
    mol = s.mol.copy()
    s.vle(V=0.3, P=101325.)
    check_mass_balance(s, mol)
    assert_allclose(s.vapor_fraction, 0.3, rtol=1e-2)
    T = s.T
    s.vle(T=T + 1, P=101325.)
    s.vle(T=T, P=101325.)
    assert_allclose(s.vapor_fraction, 0.3, rtol=2e-2)
    s.vle(T=T, P=101325.) # Repeat with same chemicals
    check_mass_balance(s, mol)

def test_vle_object():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    imol = tmo.indexer.MolarFlowIndexer(l=[('Water', 1.), ('Ethanol', 1.)], phases=('g', 'l'))
    vle = eq.VLE(imol)
    assert 'VLE' in repr(vle)
    vle(T=355., P=101325.)
    assert vle.thermal_condition.T == 355.
    assert 0. < vle.imol['g'].sum() < 2.
    vle.method = 'invalid'
    with pytest.raises(RuntimeError):
        vle(T=356., P=101325.)
    with pytest.raises(ValueError):
        tmo.settings.set_thermo(['Water'], cache=True)
        eq.VLE(imol)

def test_vle_with_eos():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    for PCF in (None, eq.IdealGasPoyintingCorrectionFactors):
        thermo = tmo.Thermo(chemicals, pkg='Peng Robinson', PCF=PCF)
        tmo.settings.set_thermo(thermo)
        s = tmo.Stream(None, Water=1., Ethanol=1.)
        mol = s.mol.copy()
        s.vle(T=350., P=101325.)
        check_phase_fractions(s)
        s.vle(V=0.5, P=101325.)
        assert_allclose(s.vapor_fraction, 0.5, rtol=1e-2)
        assert 340 < s.T < 370
        s.vle(V=0.5, T=355.)
        assert 5e4 < s.P < 2e5
        check_mass_balance(s, mol)
    s = tmo.Stream(None, Water=1., Ethanol=1.)
    s.vle.method = 'shgo'
    with pytest.raises(NotImplementedError):
        s.vle(T=350., P=101325.)

def test_vle_shgo():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    s = tmo.Stream(None, Water=1., Ethanol=1.)
    s.vle.method = 'shgo'
    s.vle(T=353., P=101325.)
    check_phase_fractions(s)
    check_mass_balance(s, np.array([1., 1.]))

# %% LLE

def lle_indexer():
    tmo.settings.set_thermo(['Water', 'Ethanol', 'Octane'], cache=True)
    return tmo.indexer.MolarFlowIndexer(
        l=[('Water', 10.), ('Ethanol', 1.)], L=[('Octane', 10.)]
    )

def check_lle(lle, mol):
    imol = lle.imol
    assert (imol.data >= 0.).all()
    assert_allclose(imol['l'] + imol['L'], mol, rtol=1e-3, atol=1e-9)
    # Water and octane split into different phases
    l_water = imol['l', 'Water'] / imol['l'].sum()
    L_water = imol['L', 'Water'] / imol['L'].sum()
    assert abs(l_water - L_water) > 0.5

def test_lle():
    imol = lle_indexer()
    mol = imol['l'] + imol['L']
    lle = eq.LLE(imol)
    assert 'LLE' in repr(lle)
    lle(T=300.)
    check_lle(lle, mol)
    lle(T=300.) # Use cache
    check_lle(lle, mol)
    lle(T=300., P=101325., use_cache=False)
    check_lle(lle, mol)
    lle(T=300., top_chemical='Water')
    assert imol['L', 'Water'] > imol['l', 'Water']
    lle(T=300., top_chemical='Methanol') # Not present
    check_lle(lle, mol)
    chemicals, K, gamma_y, phi = lle(T=301., update=False)
    assert 0. < phi < 1.
    assert len(chemicals) == len(K) == len(gamma_y) == 3
    check_lle(lle, mol)

def test_lle_differential_evolution():
    imol = lle_indexer()
    mol = imol['l'] + imol['L']
    lle = eq.LLE(imol, method='differential evolution')
    lle(T=300.)
    check_lle(lle, mol)
    lle.method = 'invalid'
    with pytest.raises(ValueError):
        lle(T=300., use_cache=False)

def test_lle_miscible():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    imol = tmo.indexer.MolarFlowIndexer(l=[('Water', 10.), ('Ethanol', 10.)], phases=('L', 'l'))
    lle = eq.LLE(imol)
    lle(T=300.)
    total = imol['l'].sum() + imol['L'].sum()
    assert_allclose(total, 20.)
    assert min(imol['l'].sum(), imol['L'].sum()) == 0.
    chemicals, K, gamma_y, phi = lle(T=300., update=False)
    assert phi in (0., 1.)
    lle(T=300.)
    lle(T=300., update=False, use_cache=False)

def test_lle_single_component():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    imol = tmo.indexer.MolarFlowIndexer(l=[('Water', 10.)], phases=('L', 'l'))
    lle = eq.LLE(imol)
    lle(T=300.)
    assert_allclose(imol['l'].sum() + imol['L'].sum(), 10.)

def test_solve_lle_mol():
    from thermosteam.equilibrium.lle import solve_lle_mol
    chemicals = tmo.Chemicals(['Water', 'Ethanol', 'Octane'], cache=True)
    gamma = eq.DortmundActivityCoefficients(chemicals)
    z = np.array([0.45, 0.1, 0.45])
    x = solve_lle_mol(gamma, z, 300., 101325.)
    assert_allclose(x.sum(), 1.)
    assert x[0] > 0.5 # Water-rich phase
    z = np.array([0.5, 0.5, 1e-4])
    x = solve_lle_mol(gamma, z, 300., 101325.) # Stable
    assert_allclose(x, z, rtol=1e-3)

def test_vle_cache():
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    imol = tmo.indexer.MolarFlowIndexer(l=[('Water', 1.), ('Ethanol', 1.)], phases=('g', 'l'))
    cache = eq.VLECache(imol, tmo.ThermalCondition(300., 101325.), tmo.settings.thermo)
    assert isinstance(cache.retrieve(), eq.VLE)


# %% Small utilities and containers

def test_ideal_coefficient_function():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    gamma = eq.IdealActivityCoefficients(chemicals)
    assert gamma.f(np.array([0.5, 0.5]), 300.) == 1.
    assert gamma.args == ()

def test_partition_coefficients():
    chemicals = tmo.Chemicals(['Water', 'Ethanol'], cache=True)
    thermo = tmo.Thermo(chemicals)
    K = eq.PartitionCoefficients('gl', chemicals, thermo)
    assert 'PartitionCoefficients' in repr(K)
    values = K(np.array([0.3, 0.7]), np.array([0.7, 0.3]), 355., 101325.)
    assert (values > 0.).all() and np.isfinite(values).all()
    with pytest.raises(ValueError):
        eq.PartitionCoefficients('glL', chemicals, thermo)

def test_phase_stability_score():
    from thermosteam.equilibrium.vle import phase_stability_score, stable_phase
    K = np.array([1e-6, 1., 1e6])
    z = np.array([0.2, 0.3, 0.5])
    score = phase_stability_score(K, z)
    assert score > 0.
    assert phase_stability_score(K, z, 0.1, 0.1) > score
    assert stable_phase(K, z)
    assert not stable_phase(np.ones(3), z)

def test_point_value_containers():
    from thermosteam.equilibrium.bubble_point import ReactiveBubblePointValues
    from thermosteam.equilibrium.dew_point import ReactiveDewPointValues
    z = np.array([0.5, 0.5])
    y = np.array([0.4, 0.6])
    bp = ReactiveBubblePointValues(350., 101325., ('Water', 'Ethanol'), z, 0 * z, y, z)
    assert_allclose(bp.K, y / z)
    assert 'ReactiveBubblePointValues' in repr(bp)
    dp = ReactiveDewPointValues(350., 101325., ('Water', 'Ethanol'), z, 0 * z, z, y)
    assert 'ReactiveDewPointValues' in repr(dp)

def test_equilibrium_base_class():
    from thermosteam.equilibrium.equilibrium import Equilibrium
    with pytest.raises(NotImplementedError):
        class NoPhases(Equilibrium):
            def __call__(self): pass
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    vle = eq.VLE()
    assert 'VLE' in format(vle, '')
    vle.show()


if __name__ == '__main__':
    test_ideal_fugacity_coefficients()
    test_eos_fugacity_coefficients()
    test_poyinting_correction_factors()
    test_fugacities()
    test_activity_coefficients()
    test_NIST_activity_coefficients()
    test_flash_package_constructors()
    test_tangent_plane_stability_analysis()
    test_edge_points_simplex()
    test_lle_tangential_plane_analysis()
    test_bubble_point_values()
    test_bubble_point_lle()
    test_dew_point_values()
    test_bubble_and_dew_points_with_eos()
    test_vle_binary_composition_specifications()
    test_vle_single_component()
    test_vle_no_chemicals()
    test_vle_binary()
    test_vle_with_light_and_heavy_chemicals()
    test_vle_supercritical_and_subcritical_exclusion()
    test_vle_multicomponent()
    test_vle_object()
    test_vle_with_eos()
    test_vle_shgo()
    test_lle()
    test_lle_differential_evolution()
    test_lle_miscible()
    test_lle_single_component()
    test_solve_lle_mol()
    test_vle_cache()
    test_ideal_coefficient_function()
    test_partition_coefficients()
    test_phase_stability_score()
    test_point_value_containers()
    test_equilibrium_base_class()
