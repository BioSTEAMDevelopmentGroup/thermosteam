# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import ether
import numpy as np

# %% Set thermo

chems = ether.Chemicals(['Water', 'Ethanol',
                         'Methanol', 'Glycerol',
                         'Propanol', 'Lignin'])
chems.Lignin.default()
chems.Lignin.lock_state('l', 298.15, 101325)
chems.compile()
thermo = ether.Thermo(chems)

# %% Test bubble point and dew point

d_bst = bst.Dortmund(*bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = chems.retrieve(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_ether = ether.BubblePoint(chemicals)
dp_ether = ether.DewPoint(chemicals)

z = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_ether, by_at_T_ether = bp_ether.solve_Py(z, T_bp)
bT_ether, by_at_P_ether = bp_ether.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_ether, dx_at_T_ether = dp_ether.solve_Px(z, T_dp)
dT_ether, dx_at_P_ether = dp_ether.solve_Tx(z, P_dp)

# %% Test Equilibrium
vle = ether.VLE()
phases = ('l', 'g')
material_data = np.array([chems.kwarray(Water=30, Ethanol=10),
                          chems.kwarray(Glycerol=3, Ethanol=10, Propanol=5)])
phase_data = tuple(zip(phases, material_data))
vle(phases, material_data, T=400, P=101325)
