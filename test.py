# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import thermotree as t3
water =  t3.Chemical('Water')
ethanol = t3.Chemical('Ethanol')
methanol = t3.Chemical('Methanol')
lignin = t3.Chemical('Lignin')
lignin.to_phaseTP('l', 298.15, 101325.)
# lignin.fill(like=water.to_phaseTP(phaseTP))
ideal_mixture = t3.IdealMixture(chemicals=(water, ethanol, methanol, lignin))