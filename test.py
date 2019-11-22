# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import thermotree as t3

# %% Ideal mixture

water =  t3.Chemical('Water')
ethanol = t3.Chemical('Ethanol')
methanol = t3.Chemical('Methanol')
glycerol = t3.Chemical('Glycerol')
lignin = t3.Chemical('Lignin')
lignin.to_phaseTP('l', 298.15, 101325.)
# lignin.fill(like=water.to_phaseTP(phaseTP))
ideal_mixture = t3.IdealMixture(chemicals=(water, ethanol, methanol, lignin))

# %% Equilibrium
from biosteam import Species, Dortmund

d_bst = Dortmund(*Species('Water', 'Ethanol', 'Methanol', 'Glycerol'))
d_t3 = t3.DortmundActivityCoefficients((water, ethanol, methanol, glycerol))
gamma_bst = d_bst([0.3, 0.4, 0.2, 0.2], 350)
gamma_t3 = d_t3([0.3, 0.4, 0.2, 0.2], 350)