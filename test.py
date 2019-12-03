# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import ether as eth
import numpy as np

# %% Set thermo

chems = eth.Chemicals(['Water', 'Ethanol',
                       'Methanol', 'Glycerol',
                       'Propanol'])
eth.settings.thermo = eth.Thermo(chems)

# %% Test bubble point and dew point and compare with BioSTEAM

import biosteam as bst

d_bst = bst.Dortmund(*bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'))
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = chems.retrieve(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_eth = eth.BubblePoint(chemicals)
dp_eth = eth.DewPoint(chemicals)

molar_data = eth.ChemicalArray(Water=2, Ethanol=4, Methanol=2, Glycerol=1, Propanol=0.1)
z = molar_data.fraction()
T_bp = 350.
P_bp = 101325 / 5
bP_bst, by_at_T_bst = bp_bst.solve_Py(z, T_bp)
bT_bst, by_at_P_bst = bp_bst.solve_Ty(z, P_bp)
bP_eth, by_at_T_eth = bp_eth.solve_Py(z, T_bp)
bT_eth, by_at_P_eth = bp_eth.solve_Ty(z, P_bp)

T_dp = 450.
P_dp = 11325
dP_bst, dx_at_T_bst = dp_bst.solve_Px(z, T_dp)
dT_bst, dx_at_P_bst = dp_bst.solve_Tx(z, P_dp)
dP_eth, dx_at_T_eth = dp_eth.solve_Px(z, T_dp)
dT_eth, dx_at_P_eth = dp_eth.solve_Tx(z, P_dp)

# %% Test Equilibrium

molar_data = eth.MolarFlow(l=[('Water', 304), ('Ethanol', 30)],
                           g=[('Ethanol', 201), ('Glycerol', 10)])
vle = eth.VLE(molar_data)
vle(T=400, P=101325)
