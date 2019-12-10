# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import biosteam as bst
import ether as eth

# %% Set thermo

species = bst.Species('Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol')

chems = eth.Chemicals(['Water', 'Ethanol',
                       'Methanol', 'Glycerol',
                       'Propanol'])
eth.settings.thermo = thermo = eth.Thermo(chems)
bst.Stream.species = species

# %% Initialize stream

molar_flow = eth.MolarFlow(l=[('Water', 304), ('Ethanol', 30), ('Glycerol', 10)],
                           g=[('Ethanol', 201), ('Methanol', 40), ('Propanol', 1)])

s1_bst = bst.MixedStream(T=300, P=101325)
s1_bst.setflow('l', Water=304, Ethanol=30, Glycerol=10)
s1_bst.setflow('g', Ethanol=201, Methanol=40, Propanol=1)

# %% Test bubble point and dew point and compare with BioSTEAM

# Ether is 5x faster than BioSTEAM
d_bst = bst.Dortmund(*species)
bp_bst = bst.BubblePoint(d_bst)
dp_bst = bst.DewPoint(d_bst)

chemicals = chems.retrieve(['Water', 'Ethanol', 'Methanol', 'Glycerol', 'Propanol'])
bp_eth = eth.BubblePoint(chemicals)
dp_eth = eth.DewPoint(chemicals)

z = molar_flow.to_chemical_array().composition(data=True)
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

# Ether is 3x faster than BioSTEAM
vle = eth.VLE(molar_flow)
vle(T=400, P=101325)

s1_bst.VLE(T=400, P=101325)

# %% Test property array

mass_flow = molar_flow.as_mass_flow()
volumetric_flow = molar_flow.as_volumetric_flow(vle.thermal_condition)

s1_eth = eth.ChemicalStream(flow=1*s1_bst.mol, T=s1_bst.T, phase='g')

# %% Test thermo

# Ether is 2x faster than BioSTEAM when handling multiple streams
s1_bst.disable_phases('g')
s2_bst = bst.Stream(phase='g', Water=304, Ethanol=30, Glycerol=10)
s2_eth = eth.ChemicalStream(phase='g', Water=304, Ethanol=30, Glycerol=10)
H1_bst_sum = s1_bst.H + s2_bst.H
H2_eth_sum = s1_eth.H + s2_eth.H
