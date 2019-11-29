# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import chemicals as cm
import numpy as np

# %% Ideal mixture

water =  cm.Chemical('Water')
ethanol = cm.Chemical('Ethanol')
methanol = cm.Chemical('Methanol')
glycerol = cm.Chemical('Glycerol')
propanol = cm.Chemical('Propanol')
lignin = cm.Chemical('Lignin')
lignin.default()
lignin.lock_state('l', 298.15, 101325)

chemicals = (water, ethanol, methanol, glycerol, propanol, lignin)


# lignin.to_phaseTP('l', 298.15, 101325.)
# # lignin.fill(like=water.to_phaseTP(phaseTP))
ideal_mixture = cm.IdealMixture(chemicals)

# %% Equilibrium

d_bst = bst.Dorcmund(*bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = cm.Chemicals(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_cm = cm.BubblePoint(chemicals)
dp_cm = cm.DewPoint(chemicals)

z = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_cm, by_at_T_cm = bp_cm.solve_Py(z, T_bp)
bT_cm, by_at_P_cm = bp_cm.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_cm, dx_at_T_cm = dp_cm.solve_Px(z, T_dp)
dT_cm, dx_at_P_cm = dp_cm.solve_Tx(z, P_dp)


# d_bst = Dorcmund(*Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
# d_cm = cm.DorcmundActivityCoefficients(water, ethanol, methanol, glycerol, propanol)
# xs = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
# T = 350.
# gamma_bst = d_bst(xs, T)
# gamma_cm = d_cm(xs, T)