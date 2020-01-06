# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import thermosteam as tmo

# %% Set thermo

species = bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol')

chems = tmo.Chemicals(['Water', 'Ethanol',
                       'Methanol', 'Glycerol',
                       'Propanol'])
tmo.settings.thermo = thermo = tmo.Thermo(chems)
bst.Stream.species = species

# %% Initialize stream

imol = tmo.indexer.MolarFlowIndexer(
    l=[('Water', 304), ('Ethanol', 30), ('Glycerol', 10)],
    g=[('Ethanol', 201), ('Methanol', 40), ('Propanol', 1)])

s1_bst = bst.MixedStream(T=300, P=101325)
s1_bst.setflow('l', Water=304, Ethanol=30, Glycerol=10)
s1_bst.setflow('g', Ethanol=201, Methanol=40, Propanol=1)


# %% Test bubble point and dew point and compare with BioSTEAM

eq = tmo.equilibrium

# thermosteam is 5x faster than BioSTEAM
d_bst = bst.Dortmund(*species)
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = chems.retrieve(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_tmo = eq.BubblePoint(chemicals)
dp_tmo = eq.DewPoint(chemicals)

mol = imol.data.sum(0)
z =  mol / mol.sum()
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_tmo, by_at_T_tmo = bp_tmo.solve_Py(z, T_bp)
bT_tmo, by_at_P_tmo = bp_tmo.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_tmo, dx_at_T_tmo = dp_tmo.solve_Px(z, T_dp)
dT_tmo, dx_at_P_tmo = dp_tmo.solve_Tx(z, P_dp)


# %% Test Equilibrium

# thermosteam is 3-10x faster than BioSTEAM
vle = eq.VLE(imol)
vle(T=400, P=101325)

s1_bst.VLE(T=400, P=101325)

# %% Test property array

s1_tmo = tmo.Stream(flow=imol.data.sum(0), T=s1_bst.T, phase='g')

# %% Test thermo

# thermosteam is 2x faster than BioSTEAM when handling multiple streams

s2_bst = bst.Stream(phase='g', Water=304, Ethanol=30, Glycerol=10)
s2_tmo = tmo.Stream(phase='g', Water=304, Ethanol=30, Glycerol=10)
H1_bst_sum = s1_bst.H + s2_bst.H
H2_tmo_sum = s1_tmo.H + s2_tmo.H

xs_1 = tmo.MultiStream()
xs_1.imol['l', ('Water', 'Ethanol')] = 20
xs_1.imol['g', ('Water', 'Propanol')] = 20