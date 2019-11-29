# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import thermodynamics as tm
import numpy as np

# %% Ideal mixture

water =  tm.Chemical('Water')
ethanol = tm.Chemical('Ethanol')
methanol = tm.Chemical('Methanol')
glycerol = tm.Chemical('Glycerol')
propanol = tm.Chemical('Propanol')
lignin = tm.Chemical('Lignin')
lignin.default()
lignin.lock_state('l', 298.15, 101325)

chemicals = (water, ethanol, methanol, glycerol, propanol, lignin)


# lignin.to_phaseTP('l', 298.15, 101325.)
# # lignin.fill(like=water.to_phaseTP(phaseTP))
ideal_mixture = tm.IdealMixture(chemicals)

# %% Equilibrium

d_bst = bst.Dortmund(*bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = tm.Chemicals(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_tm = tm.BubblePoint(chemicals)
dp_tm = tm.DewPoint(chemicals)

z = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_tm, by_at_T_tm = bp_tm.solve_Py(z, T_bp)
bT_tm, by_at_P_tm = bp_tm.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_tm, dx_at_T_tm = dp_tm.solve_Px(z, T_dp)
dT_tm, dx_at_P_tm = dp_tm.solve_Tx(z, P_dp)


# d_bst = Dortmund(*Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
# d_tm = tm.DortmundActivityCoefficients(water, ethanol, methanol, glycerol, propanol)
# xs = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
# T = 350.
# gamma_bst = d_bst(xs, T)
# gamma_tm = d_tm(xs, T)