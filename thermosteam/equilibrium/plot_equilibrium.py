# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
from .._settings import settings
from ..utils import colors, style_axis
from chemicals.identifiers import to_searchable_format
from .bubble_point import BubblePoint
from .lle import LLE
from ..indexer import MaterialIndexer
import matplotlib.pyplot as plt
from math import floor
import thermosteam as tmo
import numpy as np

__all__ = ('plot_vle_binary_phase_envelope',
           'plot_lle_ternary_diagram',
           'ternary_composition_grid',
)


# %% Utilities

def ternary_composition_grid(N, grid=None): # pragma: no cover
    grid = grid or []
    for z in np.linspace(0, 1 - 1/N, N):
        xs = np.linspace(0, 1-z, N)
        ys = 1 - z - xs 
        zs = z * np.ones(N)
        mesh = np.array([xs, ys, zs])
        grid.append(mesh)
    mesh = np.array([0, 0, 1]).reshape([3, 1])
    grid.append(mesh)
    return np.hstack(grid).transpose()

def as_thermo(thermo, chemicals): # pragma: no cover
    try:
        thermo = settings.get_default_thermo(thermo)
    except:
        thermo = tmo.Thermo(chemicals)
    else:
        thermo = tmo.Thermo(chemicals,
                            Gamma=thermo.Gamma,
                            Phi=thermo.Phi,
                            PCF=thermo.PCF)
    return thermo

# %% Plot functions

def plot_vle_binary_phase_envelope(chemicals, T=None, P=None, vc=None, lc=None, thermo=None,
                                   yticks=None): # pragma: no cover
    """
    Plot the binary phase envelope of two chemicals at a given temperature or pressure.

    Parameters
    ----------
    chemicals : Iterable[Chemical or str]
        Chemicals in equilibrium.
    T : float, optional
        Temperature [K]. 
    P : float, optional
        Pressure [Pa]. 
    vc : str, optional
        Color of vapor line.
    lc : str, optional
        Color of liquid line.
    thermo : Thermo, optional
        Thermodynamic property package.

    Examples
    --------
    >>> # from thermosteam import equilibrium as eq
    >>> # eq.plot_vle_binary_phase_envelope(['Ethanol', 'Water'], P=101325)
    
    .. figure:: ../images/water_ethanol_binary_phase_envelope.png

    """
    thermo = as_thermo(thermo, chemicals)
    chemical_a, chemical_b = chemicals = [thermo.as_chemical(i) for i in chemicals]
    BP = tmo.BubblePoint(chemicals, thermo)
    zs_a = np.linspace(0, 1)
    zs_b = 1 - zs_a
    zs = np.vstack([zs_a, zs_b]).transpose()
    if P:
        assert not T, "must pass either T or P, but not both"
        bps = [BP(z, P=P) for z in zs]
        ms = [bp.T for bp in bps]
        ylabel = 'Temperature [K]'
    elif T:
        assert not P, "must pass either T or P, but not both"
        bps = [BP(z, T=T) for z in zs]
        ms = [bp.P for bp in bps]
        ylabel = 'Pressure [Pa]'
    else:
        raise AssertionError("must pass either T or P")
    ms = np.array(ms)
    ys_a = np.array([bp.y[0] for bp in bps])
    plt.figure()
    plt.xlim([0, 1])
    plt.plot(ys_a, ms, c=vc if vc is not None else colors.red.RGBn, label="vapor")
    plt.plot(zs_a, ms, c=lc if lc is not None else colors.green.RGBn, label='liquid')
    if yticks is not None: plt.yticks(yticks)
    plt.legend()
    plt.xlabel(f'{chemical_a} molar fraction')
    plt.ylabel(ylabel)
    style_axis(xticks=np.linspace(0, 1, 5), yticks=yticks)
    
def plot_lle_ternary_diagram(carrier, solvent, solute, T, P=101325, thermo=None, color=None,
                             tie_line_points=None,
                             tie_color=None,
                             N_tie_lines=15,
                             N_equilibrium_grids=15): # pragma: no cover
    """
    Plot the ternary phase diagram of chemicals in liquid-liquid equilibrium.

    Parameters
    ----------
    carrier : Chemical
    solvent : Chemical
    solute : Chemical
    T : float, optional
        Temperature [K]. 
    P : float, optional
        Pressure [Pa]. Defaults to 101325.
    thermo : Thermo, optional
        Thermodynamic property package.
    color : str, optional
        Color of equilibrium line.
    tie_line_points : 1d array(size=3), optional
        Additional composition points to create tie lines.
    tie_color : str, optional
        Color of tie lines.
    N_tie_lines : int, optional
        Number of tie lines. The default is 15.
    N_equilibrium_grids : int, optional
        Number of solute composition points to plot. The default is 15.

    Examples
    --------
    >>> # from thermosteam import equilibrium as eq
    >>> # eq.plot_lle_ternary_diagram('Water', 'Ethanol', 'EthylAcetate', T=298.15)
    
    .. figure:: ../images/water_ethanol_ethyl_acetate_ternary_diagram.png

    """
    import ternary
    chemicals = [carrier, solvent, solute]
    thermo = as_thermo(thermo, chemicals)
    chemicals = thermo.chemicals
    IDs = chemicals.IDs
    imol = MaterialIndexer(['l', 'L'], chemicals=chemicals)
    data = imol.data
    lle = LLE(imol, thermo=thermo)
    composition_grid = ternary_composition_grid(N_equilibrium_grids, tie_line_points)
    tie_lines = []
    MW = chemicals.MW
    for zs in composition_grid:
        data[:] = 0
        data[0] = zs
        lle(T, P)
        mass = data * MW
        lL = mass.sum(1, keepdims=True) 
        if (abs(lL) > 1e-5).all():
            tie_line = 100 * mass / lL
            tie_diff = np.abs(tie_line[0] - tie_line[1]) ** 0.75
            if tie_diff.sum() > 0.2:
                tie_lines.append(tie_line)
    tie_points = np.vstack(tie_lines)
    tie_points = tie_points[np.argsort(tie_points[:, 0])]
    scale = 100.
    fig, tax = ternary.figure(scale=scale)
    label = "-".join(IDs) + " Phase Diagram"
    tax.boundary(linewidth=2.0)
    tax.gridlines(color=colors.grey_tint.RGBn, multiple=10)
    fontsize = 12
    offset = 0.14
    C, A, S = [to_searchable_format(i) for i in IDs]
    tax.right_corner_label(8*' ' + 'Carrier', fontsize=fontsize)
    tax.top_corner_label('Solute\n', fontsize=fontsize)
    tax.left_corner_label('Solvent' + 8*' ', fontsize=fontsize)
    tax.left_axis_label(f'{S} wt. %', fontsize=fontsize,
                        offset=offset)
    tax.right_axis_label(f'{A} wt. %', fontsize=fontsize,
                         offset=offset)
    tax.bottom_axis_label(f'{C} wt. %', fontsize=fontsize,
                          offset=offset)
    if color is None: color = 'k'
    tax.plot(tie_points, c=colors.neutral_shade.RGBn, label=label)
    
    if tie_line_points is None:
        assert N_tie_lines, "must specify number of tie lines if no equilibrium points are given"
        xs, ys = zip(*tie_lines)
        xs = np.array(xs)
        ys = np.array(ys)
        index = np.argsort(xs[:, 0])
        N_total_lines = len(tie_lines) 
        step = floor(N_total_lines / N_tie_lines)
        partition_index = np.arange(0, N_total_lines, step, dtype=int)
        index = index[partition_index]
        xs = [xs[i] for i in index]
        ys = [ys[i] for i in index]
        tie_lines = list(zip(xs, ys))    
    else:
        assert N_tie_lines is None, "cannot specify number of tie lines if equilibrium points are given"
        N_tie_lines = len(tie_line_points)
        tie_lines = tie_lines[:N_tie_lines]
    if tie_color is None: tie_color = colors.grey_shade.RGBn
    for x, y in tie_lines:
        tax.line(x, y, ls='--', c=tie_color)
    
    tax.ticks(axis='lbr', multiple=10, linewidth=1, offset=0.025)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    
    