# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import thermotree as t3
import numpy as np

# %% Ideal mixture

# water =  t3.Chemical('Water')
# ethanol = t3.Chemical('Ethanol')
# methanol = t3.Chemical('Methanol')
# glycerol = t3.Chemical('Glycerol')
# propanol = t3.Chemical('Propanol')
# lignin = t3.Chemical('Lignin')

# lignin.to_phaseTP('l', 298.15, 101325.)
# # lignin.fill(like=water.to_phaseTP(phaseTP))
# ideal_mixture = t3.IdealMixture(chemicals=(water, ethanol, methanol, lignin))

# %% Equilibrium

d_bst = bst.Dortmund(*bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = t3.Chemicals('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol')
bp_t3 = t3.BubblePoint(chemicals)
dp_t3 = t3.DewPoint(chemicals)

z = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_t3, by_at_T_t3 = bp_t3.solve_Py(z, T_bp)
bT_t3, by_at_P_t3 = bp_t3.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_t3, dx_at_T_t3 = dp_t3.solve_Px(z, T_dp)
dT_t3, dx_at_P_t3 = dp_t3.solve_Tx(z, P_dp)


# d_bst = Dortmund(*Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
# d_t3 = t3.DortmundActivityCoefficients(water, ethanol, methanol, glycerol, propanol)
# xs = np.array([0.2, 0.4, 0.05, 0.25, 0.1])
# T = 350.
# gamma_bst = d_bst(xs, T)
# gamma_t3 = d_t3(xs, T)