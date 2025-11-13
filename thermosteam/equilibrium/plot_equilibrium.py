# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
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
from numpy.linalg import lstsq
import flexsolve as flx
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter

__all__ = (
    'plot_vle_binary_phase_envelope',
    'plot_vle_phase_envelope',
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
    N = 50
    zs_a = np.linspace(0, 1, N)
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
    ms_liq = ms.copy()
    ms_gas = interpolate.interp1d(ys_a, ms, bounds_error=False, kind='slinear')(zs_a)
    if P:
        azeotrope = ms_liq > ms_gas
    elif T:
        azeotrope = ms_liq < ms_gas
    ms_liq[0] = 0.5 * (ms_liq[0] + ms_gas[0])
    ms_liq[-1] = 0.5 * (ms_liq[-1] + ms_gas[-1])
    if azeotrope.any():
        index = np.where(azeotrope)[0]
        left = index[0]
        right = index[-1]
        mid = int(np.median(index))
        azeotrope[left:right] = True
        azeotrope[mid] = False
        ms_gas[mid] = ms_liq[mid]
        ms_gas = interpolate.interp1d(zs_a[~azeotrope], ms_gas[~azeotrope], bounds_error=False, kind='slinear')(zs_a)
        
    top, bottom = (chemical_a, chemical_b) if ys_a.mean() > 0.5 else (chemical_a, chemical_b)
    plt.figure()
    plt.xlim([0, 1])
    plt.plot(zs_a, ms_gas, c=vc if vc is not None else colors.red.RGBn, label='vapor')
    plt.plot(zs_a, ms_liq, c=lc if lc is not None else colors.blue.RGBn, label='liquid')
    plt.ylim([ms.min(), ms.max()])
    if yticks is None: yticks, ytext = plt.yticks()
        
    plt.legend()
    plt.xlabel(f'{chemical_a} molar fraction')
    plt.ylabel(ylabel)
    style_axis(xticks=np.linspace(0, 1, 5), yticks=yticks)
    
def plot_vle_phase_envelope(
        IDs, zs, T_range=None, P_range=None, thermo=None, vc=None, lc=None, 
        xticks=None, yticks=None, N=None, line_styles=None, labels=None): # pragma: no cover
    """
    Plot the binary phase envelope of two chemicals at a given temperature or pressure.

    Parameters
    ----------
    IDs : Iterable[str]
        IDs of chemicals in equilibrium.
    zs : Iterable[float]
        Molar fraction of chemicals in equilibrium.
    T_range : tuple[float, float] optional
        Temperature [K]. 
    P_range : tuple[float, float] optional
        Pressure [atm]. 
    thermo : Thermo|Iterable[Thermo], optional
        Thermodynamic property package(s).
    vc : str, optional
        Color of vapor line.
    lc : str, optional
        Color of liquid line.

    """
    if line_styles is None: line_styles = ['-', '-.', '--']
    if thermo is None: 
        thermo = [tmo.settings.get_thermo()]
    elif isinstance(thermo, (tmo.Thermo, tmo.IdealThermo)):
        thermo = [thermo]
    if labels is None:
        N_thermo = len(thermo)
        if N_thermo == 1:
            labels = ['']
        else:
            labels = [str(i) for i in range(N_thermo)]
    BPs = [tmo.BubblePoint(t.chemicals[IDs], t) for t in thermo]
    DPs = [tmo.DewPoint(t.chemicals[IDs], t) for t in thermo]
    if N is None: N = 20
    if P_range is not None:
        assert T_range is None, "must pass either T_range or P_range, but not both"
        xlim = P_range
        xs = np.linspace(*xlim, N)
        bpss = [
            [BP(zs, P=P * 101325) for P in xs]
            for BP in BPs
        ]
        dpss = [
            [DP(zs, P=P * 101325) for P in xs]
            for DP in DPs
        ]
        ylabel = 'Temperature [K]'
        xlabel = 'Pressure [Pa]'
        variable = 'T'
    elif T_range:
        assert P_range is None, "must pass either T_range or P_range, but not both"
        xlim = T_range
        xs = np.linspace(*xlim, N)
        bpss = [
            [BP(zs, T=T) for T in xs]
            for BP in BPs
        ]
        dpss = [
            [DP(zs, T=T) for T in xs]
            for DP in DPs
        ]
        ylabel = 'Pressure [Pa]'
        xlabel = 'Temperature [K]'
        variable = 'P'
    else:
        raise AssertionError("must pass either T or P")
    Lss = np.array([
        [getattr(bp, variable) for bp in bps]
        for bps in bpss
    ])
    Vss = np.array([
        [getattr(dp, variable) for dp in dps]
        for dps in dpss
    ])
    plt.figure()
    if vc is None: vc = colors.red.RGBn
    if lc is None: lc = colors.blue.RGBn
    for Vs, Ls, ls, label in zip(Vss, Lss, line_styles, labels):
        plt.plot(xs, Vs, c=vc, ls=ls, label=', '.join([label, 'vapor']))
        plt.plot(xs, Ls, c=lc, ls=ls, label=', '.join([label, 'liquid']))
    plt.xlim(xlim)
    plt.ylim([min(Lss.min(), Vss.min()), max(Lss.max(), Vss.max())])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    style_axis(xticks=xticks, yticks=yticks)
    
def plot_lle_ternary_diagram(
        carrier, solvent, solute, T, P=101325, thermo=None, color=None,
        tie_line_points=None,
        tie_color=None,
        N_tie_lines=8,
        N_equilibrium_grids=8,
        method=None
    ): # pragma: no cover
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
    chemicals = [carrier, solute, solvent]
    thermo = as_thermo(thermo, chemicals)
    chemicals = thermo.chemicals
    IDs = chemicals.IDs
    imol = MaterialIndexer(['l', 'L'], chemicals=chemicals)
    data = imol.data
    lle = LLE(imol, thermo=thermo, method=method)
    composition_grid = ternary_composition_grid(N_equilibrium_grids, tie_line_points)
    tie_lines = []
    MW = chemicals.MW
    for zs in composition_grid:
        data[0] = zs
        data[1] = 0
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
    
    def f(x):
        shape = np.shape(x)
        X = np.ones([*shape, 5])
        X[..., 0] = x2 = x * x
        X[..., 1] = np.sqrt(x)
        X[..., 2] = x
        X[..., 3] = x2 * x
        return X
    
    xs = tie_points[:, 0]
    ys = tie_points[:, 1]
    X = f(xs)
    result = lstsq(X, ys, rcond=None)
    coef = result[0]
    xs = np.linspace(0, 100, 500)
    ys = (f(xs) * coef).sum(axis=1)
    zs = 100 - xs - ys
    ys[ys < 0] = 0
    zs[zs < 0] = 0
    tie_points = np.zeros([len(xs), 3])
    tie_points[:, 0] = xs
    tie_points[:, 1] = ys
    tie_points[:, 2] = zs
    tie_points /= tie_points.sum(axis=1, keepdims=True) / 100
    fig, tax = ternary.figure(scale=100)
    label = "-".join(IDs) + " Phase Diagram"
    tax.boundary(linewidth=2.0)
    tax.gridlines(color=colors.grey_tint.RGBn, multiple=10)
    fontsize = 12
    offset = 0.14
    C, A, S = [to_searchable_format(i) for i in IDs]
    tax.right_corner_label(8*' ' + 'Solvent', fontsize=fontsize)
    tax.top_corner_label('Solute\n', fontsize=fontsize)
    tax.left_corner_label('Carrier' + 8*' ', fontsize=fontsize)
    tax.left_axis_label(f'{C} wt. %', fontsize=fontsize,
                        offset=offset)
    tax.right_axis_label(f'{A} wt. %', fontsize=fontsize,
                         offset=offset)
    tax.bottom_axis_label(f'{S} wt. %', fontsize=fontsize,
                          offset=offset)
    if color is None: color = 'k'
    tax.plot(tie_points, c=colors.neutral_shade.RGBn, label=label)
    
    if tie_line_points is None:
        assert N_tie_lines, "must specify number of tie lines if no equilibrium points are given"
        xs, ys = zip(*tie_lines)
        new_xs = np.array(xs)
        new_ys = np.array(ys)
        # new_xs = []
        # new_ys = []
        # for left, right in zip(xs, ys):
        #     x, *_ = left
        #     y = (f(x) * coef).sum()
        #     z = 100 - y - x
        #     if y < 0 or z < 0: continue
        #     left = np.array([x, y, z])
        #     left = 100 * left / left.sum()
        #     x, *_ = right
        #     y = (f(x) * coef).sum()
        #     z = 100 - y - x
        #     if y < 0 or z < 0: continue
        #     right = np.array([x, y, z])
        #     right = 100 * right / right.sum()
        #     new_xs.append(left)
        #     new_ys.append(right)
        # new_xs = np.array(new_xs)
        # new_ys = np.array(new_ys)
        # index = np.argsort(new_xs[:, 0])
        N_total_lines = len(new_xs) 
        step = floor(N_total_lines / N_tie_lines) or 1
        partition_index = np.arange(0, N_total_lines, step, dtype=int)
        # index = index[partition_index]
        new_xs = [new_xs[i] for i in partition_index]
        new_ys = [new_ys[i] for i in partition_index]
        tie_lines = list(zip(new_xs, new_ys))    
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