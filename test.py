# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
# import biosteam as bst
import thermosteam as tmo

# %% Set thermo
CO2 = tmo.Chemical('CO2')
CO2 = CO2.at_state(phase='g')
chems = tmo.Chemicals(['Water', 'Ethanol',
                       'Methanol', 'Glycerol',
                       'Propanol', CO2])
tmo.settings.thermo = thermo = tmo.Thermo(chems)

# %% Initialize stream

imol = tmo.indexer.MolarFlowIndexer(
    l=[('Water', 304), ('Ethanol', 30), ('Glycerol', 10)],
    g=[('Ethanol', 201), ('Methanol', 40), ('Propanol', 1)])


# %% Test bubble point and dew point and compare with BioSTEAM

eq = tmo.equilibrium

# thermosteam is 5x faster than BioSTEAM

chemicals = chems.retrieve(['Water', 'Ethanol', 'Methanol',
                            'Glycerol', 'Propanol'])
bp_tmo = eq.BubblePoint(chemicals)
dp_tmo = eq.DewPoint(chemicals)

equilibrium_IDs = ('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol')
mol = imol[..., equilibrium_IDs].sum(0)
z =  mol / mol.sum()
T_bp = 350.
P_bp = 101325 / 5
bP_tmo, by_at_T_tmo = bp_tmo.solve_Py(z, T_bp)
bT_tmo, by_at_P_tmo = bp_tmo.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_tmo, dx_at_T_tmo = dp_tmo.solve_Px(z, T_dp)
dT_tmo, dx_at_P_tmo = dp_tmo.solve_Tx(z, P_dp)


# %% Test Equilibrium

# thermosteam is 3-10x faster than BioSTEAM
vle = eq.VLE(imol)
vle(T=400, P=101325)

# %% Test property array

s1_tmo = tmo.Stream(flow=imol.data.sum(0), T=400, phase='g')

# %% Test thermo

# thermosteam is 2x faster than BioSTEAM when handling multiple streams

s2_tmo = tmo.Stream(phase='g', Water=304, Ethanol=30, Glycerol=10)
H2_tmo_sum = s1_tmo.H + s2_tmo.H

xs_1 = tmo.MultiStream()
xs_1.imol['l', ('Water', 'Ethanol')] = 20
xs_1.imol['g', ('Water', 'Propanol')] = 20
xs_1.vle(P=101325, V=0.5)