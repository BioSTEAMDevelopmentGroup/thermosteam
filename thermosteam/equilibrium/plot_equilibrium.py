# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:53:34 2020

@author: yoelr
"""
import matplotlib.pyplot as plt
from .._settings import settings
from ..utils import colors, fix_axis_style
from .bubble_point import BubblePoint
import numpy as np

__all__ = ('plot_vle_binary_phase_envelope',)


# %% Utilities

def as_chemical(chemicals, chemical):
    return getattr(chemicals, chemical) if isinstance(chemical, str) else chemical


# %% Plot functions

def plot_vle_binary_phase_envelope(chemical_a, chemical_b,
                                   T=None, P=None, thermo=None):
    thermo = settings.get_default_thermo(thermo)
    chemicals = thermo.chemicals
    chemical_a = as_chemical(chemicals, chemical_a)
    chemical_b = as_chemical(chemicals, chemical_b)
    chemicals = (chemical_a, chemical_b)
    BP = BubblePoint(chemicals, thermo)
    zs_a = np.linspace(0, 1)
    zs_b = 1 - zs_a
    zs = np.vstack([zs_a, zs_b]).transpose()
    if P:
        bps = [BP(z, P=P) for z in zs]
        ms = [bp.T for bp in bps]
        ylabel = 'Temperature [K]'
    else:
        bps = [BP(z, T=T) for z in zs]
        ms = [bp.P for bp in bps]
        ylabel = 'Pressure [Pa]'
    ms = np.array(ms)
    ys_a = np.array([bp.y[0] for bp in bps])
    plt.figure()
    plt.xlim([0, 1])
    plt.plot(ys_a, ms, c=colors.grey_shade.RGBn,
             label=f"{chemical_a}-{chemical_b} Phase Envelope")
    plt.plot(zs_a, ms, c=colors.grey_shade.RGBn)
    plt.legend()
    plt.xlabel(f'{chemical_a} molar fraction')
    plt.ylabel(ylabel)
    fix_axis_style(xticks=np.linspace(0, 1, 5),
                   yticks=np.linspace(ms.min(), ms.max(), 5))
    
    