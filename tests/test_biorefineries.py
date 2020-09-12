# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import numpy as np
from biosteam.process_tools import UnitGroup

def test_sugarcane():
    from biorefineries import sugarcane as sc
    sc.load()
    units = UnitGroup('Biorefinery', sc.sugarcane_tea.units)
    assert np.allclose(sc.sugarcane_tea.IRR, 0.10319729375184486, rtol=1e-2)
    assert np.allclose(sc.sugarcane_tea.sales, 87641104.25548652, rtol=1e-2)
    assert np.allclose(sc.sugarcane_tea.material_cost, 60096508.050463505, rtol=1e-2)
    assert np.allclose(sc.sugarcane_tea.installed_equipment_cost, 111512561.90122335, rtol=1e-2)
    assert np.allclose(sc.sugarcane_tea.utility_cost, -8589365.38733437, rtol=1e-2)
    assert np.allclose(units.get_heating_duty(), 272.4872352787688, rtol=1e-2)
    assert np.allclose(units.get_cooling_duty(), 344.88392082847315, rtol=1e-2)
    assert np.allclose(units.get_electricity_consumption(), 9.853875285162502, rtol=1e-2)
    assert np.allclose(units.get_electricity_production(), 37.383892552259844, rtol=1e-2)

def test_lipidcane():
    from biorefineries import lipidcane as lc
    lc.load()
    units = UnitGroup('Biorefinery', lc.lipidcane_tea.units)
    assert np.allclose(lc.lipidcane_tea.IRR, 0.1783634347376759, rtol=1e-2)
    assert np.allclose(lc.lipidcane_tea.sales, 102410581.5597974, rtol=1e-2)
    assert np.allclose(lc.lipidcane_tea.material_cost, 61762971.048900105, rtol=1e-2)
    assert np.allclose(lc.lipidcane_tea.installed_equipment_cost, 139559824.39299855, rtol=1e-2)
    assert np.allclose(lc.lipidcane_tea.utility_cost, -25886609.91411801, rtol=1e-2)
    assert np.allclose(units.get_heating_duty(), 197.74264093327824, rtol=1e-2)
    assert np.allclose(units.get_cooling_duty(), 305.1670349131174, rtol=1e-2)
    assert np.allclose(units.get_electricity_consumption(), 9.65815356491132, rtol=1e-2)
    assert np.allclose(units.get_electricity_production(), 92.62805713580244, rtol=1e-2)

def test_cornstover():
    from biorefineries import cornstover as cs
    cs.load()
    MESP = cs.cornstover_tea.solve_price(cs.ethanol)
    units = UnitGroup('Biorefinery', cs.cornstover_tea.units)
    assert np.allclose(MESP, 0.7383610895500932, rtol=1e-2)
    assert np.allclose(cs.cornstover_tea.sales, 134055638.87837549, rtol=1e-2)
    assert np.allclose(cs.cornstover_tea.material_cost, 82902092.80482696, rtol=1e-2)
    assert np.allclose(cs.cornstover_tea.installed_equipment_cost, 220012587.48251432, rtol=1e-2)
    assert np.allclose(cs.cornstover_tea.utility_cost, -11047774.698866017, rtol=1e-2)
    assert np.allclose(units.get_heating_duty(), 318.05635812416523, rtol=1e-2)
    assert np.allclose(units.get_cooling_duty(), 365.67806026618115, rtol=1e-2)
    assert np.allclose(units.get_electricity_consumption(), 22.371322764496814, rtol=1e-2)
    assert np.allclose(units.get_electricity_production(), 45.33827889984683, rtol=1e-2)
    
def test_LAOs():
    from biorefineries import LAOs as laos
    laos.load()
    MPSP = laos.get_LAOs_MPSP()
    units = UnitGroup('Biorefinery', laos.LAOs_tea.units)
    assert np.allclose(MPSP, 1226.0016718824597, rtol=1e-2)
    assert np.allclose(laos.LAOs_tea.sales, 168789574.412942, rtol=1e-2)
    assert np.allclose(laos.LAOs_tea.material_cost, 135661583.58974993, rtol=1e-2)
    assert np.allclose(laos.LAOs_tea.installed_equipment_cost, 76661769.51734665, rtol=1e-2)
    assert np.allclose(laos.LAOs_tea.utility_cost, 3811246.803774001, rtol=1e-2)
    assert np.allclose(units.get_heating_duty(), 59.19501587183199, rtol=1e-2)
    assert np.allclose(units.get_cooling_duty(), 134.66774238880174, rtol=1e-2)
    assert np.allclose(units.get_electricity_consumption(), 3.689496361470118, rtol=1e-2)
    assert np.allclose(units.get_electricity_production(), 3.6894963614701215, rtol=1e-2)   