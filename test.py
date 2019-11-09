# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:37:43 2019

@author: yoelr
"""
import thermotree as tm 
water = tm.Chemical('Water')
ethanol = tm.Chemical('Ethanol')
methanol = tm.Chemical('Methanol')
ideal_mixture = tm.IdealMixture(chemicals=(water, ethanol, methanol))