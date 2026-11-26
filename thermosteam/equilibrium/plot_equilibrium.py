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
from .lle import LLE, solve_lle_phase
from ..indexer import MaterialIndexer
import matplotlib.pyplot as plt
from math import floor
import thermosteam as tmo
import numpy as np
from numpy.linalg import lstsq
import flexsolve as flx
from scipy import interpolate
from scipy.ndimage.filters import gaussian_filter
from typing import Iterable

__all__ = (
    'plot_vle_binary_phase_envelope',
    'plot_vle_phase_envelope',
    'plot_vlle_phase_envelope',
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
                                   yticks=None, line_styles=None, N=None, lle=False): # pragma: no cover
    """
    Plot the binary phase envelope of two chemicals at a given temperature or pressure.

    Parameters
    ----------
    chemicals : Iterable[Chemical or str]
        Chemicals in equilibrium.
    T : float|list[float], optional
        Temperature [K]. 
    P : float|list[float], optional
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
    
    .. figure:: /images/water_ethanol_binary_phase_envelope.png

    """
    if line_styles is None: line_styles = ['-', '-.', '--']
    thermo = as_thermo(thermo, chemicals)
    chemical_a, chemical_b = chemicals = [thermo.as_chemical(i) for i in chemicals]
    BP = tmo.BubblePoint(chemicals, thermo)
    if N is None: N = 100
    zs_a = np.linspace(0, 1, N)
    zs_b = 1 - zs_a
    zs = np.vstack([zs_a, zs_b]).transpose()
    if P:
        if not isinstance(P, Iterable): P = [P]
        Xs = P
        X_units = 'Pa'
        assert not T, "must pass either T or P, but not both"
        bpss = [[BP(z, P=i, lle=lle) for z in zs] for i in P]
        mss = [[bp.T for bp in bps] for bps in bpss]
        ylabel = 'Temperature [K]'
    elif T:
        if not isinstance(T, Iterable): T = [T]
        assert not P, "must pass either T or P, but not both"
        Xs = T
        X_units = 'K'
        bpss = [[BP(z, T=i, lle=lle) for z in zs] for i in T]
        mss = [[bp.P for bp in bps] for bps in bpss]
        ylabel = 'Pressure [Pa]'
    else:
        raise AssertionError("must pass either T or P")
    plt.figure()
    mss = np.array(mss)
    for X, ms, bps, ls in zip(Xs, mss, bpss, line_styles):
        ys_a = np.array([bp.y[0] for bp in bps])
        ms_liq = ms.copy()
        # ms_gas = interpolate.interp1d(ys_a, ms, bounds_error=False, kind='slinear')(zs_a)
        # if P:
        #     azeotrope = ms_liq > ms_gas
        # elif T:
        #     azeotrope = ms_liq < ms_gas
        # ms_liq[0] = 0.5 * (ms_liq[0] + ms_gas[0])
        # ms_liq[-1] = 0.5 * (ms_liq[-1] + ms_gas[-1])
        # if azeotrope.any():
        #     index = np.where(azeotrope)[0]
        #     left = index[0]
        #     right = index[-1]
        #     mid = int(np.median(index))
        #     azeotrope[left:right] = True
        #     azeotrope[mid] = False
        #     ms_gas[mid] = ms_liq[mid]
        #     ms_gas = interpolate.interp1d(zs_a[~azeotrope], ms_gas[~azeotrope], bounds_error=False, kind='slinear')(zs_a)
        plt.plot(ys_a, ms, c=vc if vc is not None else colors.red.RGBn, label=f'vapor [{int(X)} {X_units}]', ls=ls)
        plt.plot(zs_a, ms_liq, c=lc if lc is not None else colors.blue.RGBn, label=f'liquid [{int(X)} {X_units}]', ls=ls)
    plt.xlim([0, 1])
    plt.ylim([mss.min(), mss.max()])
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
        xlabel = 'Pressure [atm]'
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
        ylabel = 'Pressure [atm]'
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

def plot_vlle_phase_envelope(
        chemicals, T=None, P=None, thermo=None, N=None,
        ylim=None, vc=None, lc=None, llc=None, tc=None, xticks=None, yticks=None,
    ): # pragma: no cover
    """
    Plot the vapor-liquid-liquid equilibrium (VLLE) phase envelope of a binary
    mixture at a given temperature or pressure. The molar fraction of one
    chemical is plotted on the x-axis. If the pressure is given, the temperature
    is plotted on the y-axis; if the temperature is given, the pressure is
    plotted on the y-axis.

    The envelope includes the bubble point (liquid) and dew point (vapor) curves,
    the three-phase line where vapor and two liquid phases coexist, and the
    liquid-liquid solubility curves.

    Parameters
    ----------
    chemicals : Iterable[Chemical|str]
        The two chemicals in equilibrium. 
        The first chemical's molar fraction is plotted along the x-axis.
    T : float, optional
        Temperature [K]. If given, the pressure [Pa] is plotted on the y-axis.
    P : float, optional
        Pressure [Pa]. If given, the temperature [K] is plotted on the y-axis.
    thermo : Thermo, optional
        Thermodynamic property package.
    N : int, optional
        Number of points along each segment of the bubble point, dew point,
        and liquid-liquid solubility curves. Defaults to 50.
    ylim : tuple[float, float], optional
        Limits of the y-axis. By default, the liquid-liquid region is shown
        down to (at given pressure) or up to (at given temperature) 30% of the
        span of the vapor-liquid region beyond the three-phase line.
    vc : str, optional
        Color of vapor (dew point) line.
    lc : str, optional
        Color of liquid (bubble point) line.
    llc : str, optional
        Color of liquid-liquid solubility lines.
    tc : str, optional
        Color of three-phase line.

    Examples
    --------
    >>> # from thermosteam import equilibrium as eq
    >>> # eq.plot_vlle_phase_envelope(['Water', 'Hexane'], chemical='Hexane', P=101325)

    """
    thermo = as_thermo(thermo, chemicals)
    chemicals = [thermo.as_chemical(i) for i in chemicals]
    if len(chemicals) != 2:
        raise ValueError(f'must pass exactly two chemicals, not {len(chemicals)}')
    IDs = [i.ID for i in chemicals]
    if (T is None) == (P is None):
        raise ValueError('must pass either T or P, but not both')
    if P is None:
        variable = 'P'
        specification = dict(T=T)
        ylabel = 'Pressure [Pa]'
        sign = 1 # Liquid-liquid region is at higher pressures
    else:
        variable = 'T'
        specification = dict(P=P)
        ylabel = 'Temperature [K]'
        sign = -1 # Liquid-liquid region is at lower temperatures
    if N is None: N = 50
    BP = tmo.BubblePoint(chemicals, thermo)
    DP = tmo.DewPoint(chemicals, thermo)
    gamma = BP.gamma

    def composition(x):
        z = np.zeros(2)
        z[0] = x
        z[1] = 1. - x
        return z

    def bubble_points(xs):
        return [getattr(BP(composition(x), lle=True, **specification), variable) for x in xs]

    def dew_points(ys):
        return [getattr(DP(composition(y), lle=True, **specification), variable) for y in ys]

    # Find the three-phase point (if any) by searching for a liquid that
    # splits into two liquid phases at the bubble point.
    three_phase_point = None
    for x in np.linspace(0.05, 0.95, 19):
        bp = BP(composition(x), lle=True, **specification)
        if bp.two_liquids:
            three_phase_point = bp
            break
    if three_phase_point is None:
        xs = ys = np.linspace(0, 1, N)
        liquid = [(xs, bubble_points(xs))]
        vapor = [(ys, dew_points(ys))]
        Ms = [*liquid[0][1], *vapor[0][1]]
        if ylim is None: ylim = (min(Ms), max(Ms))
        lle = None
    else:
        M3 = getattr(three_phase_point, variable)
        y3 = three_phase_point.y[0]
        x3_lb, x3_ub = sorted([three_phase_point.xα[0], three_phase_point.xβ[0]])
        liquid = []
        vapor = []
        for x_lb, x_ub, y_lb, y_ub in ((0., x3_lb, 0., y3), (x3_ub, 1., y3, 1.)):
            # The three-phase point is excluded from calculations and set directly.
            xs = np.linspace(x_lb, x_ub, N)
            ys = np.linspace(y_lb, y_ub, N)
            if x_lb == 0.:
                xs_calc = xs[:-1]
                ys_calc = ys[:-1]
                liquid.append((xs, [*bubble_points(xs_calc), M3]))
                vapor.append((ys, [*dew_points(ys_calc), M3]))
            else:
                xs_calc = xs[1:]
                ys_calc = ys[1:]
                liquid.append((xs, [M3, *bubble_points(xs_calc)]))
                vapor.append((ys, [M3, *dew_points(ys_calc)]))
        Ms = [j for _, i in liquid + vapor for j in i]
        if ylim is None:
            if sign < 0:
                ylim = (M3 - 0.3 * (max(Ms) - M3), max(Ms))
            else:
                ylim = (min(Ms), M3 + 0.3 * (M3 - min(Ms)))

        # Trace liquid-liquid solubility curves away from the three-phase line
        # until the limit of the plot or the critical solution point.
        M_end = ylim[0] if sign < 0 else ylim[1]
        Ms_lle = [M3]
        xs_lb = [x3_lb]
        xs_ub = [x3_ub]
        for M in np.linspace(M3, M_end, N)[1:]:
            z = composition(0.5 * (xs_lb[-1] + xs_ub[-1]))
            if variable == 'T':
                x, ϕ = solve_lle_phase(gamma, z, M, P)
            else:
                x, ϕ = solve_lle_phase(gamma, z, T, M)
            if ϕ >= 1.: break # Single liquid phase
            w = z - ϕ * x
            w[w < 0.] = 0.
            w /= w.sum()
            x_lb, x_ub = sorted([x[0], w[0]])
            if x_ub - x_lb < 1e-6: break # Trivial split
            Ms_lle.append(M)
            xs_lb.append(x_lb)
            xs_ub.append(x_ub)
        lle = (xs_lb, xs_ub, Ms_lle)

    if vc is None: vc = colors.red.RGBn
    if lc is None: lc = colors.blue.RGBn
    if llc is None: llc = colors.purple.RGBn
    if tc is None: tc = colors.neutral_shade.RGBn
    plt.figure()
    for i, (ys, Ms) in enumerate(vapor):
        plt.plot(ys, Ms, c=vc, label=None if i else 'Dew point (vapor)')
    for i, (xs, Ms) in enumerate(liquid):
        plt.plot(xs, Ms, c=lc, label=None if i else 'Bubble point (liquid)')
    if lle is not None:
        xs_lb, xs_ub, Ms_lle = lle
        plt.plot(xs_lb, Ms_lle, c=llc, label='Liquid-liquid solubility')
        plt.plot(xs_ub, Ms_lle, c=llc)
        plt.plot(
            [min(x3_lb, y3), max(x3_ub, y3)], [M3, M3], c=tc,
            label='Three-phase line'
        )
    plt.xlim([0, 1])
    plt.ylim(ylim)
    plt.legend()
    plt.xlabel(f'{IDs[0]} molar fraction')
    plt.ylabel(ylabel)
    if xticks is None: xticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    style_axis(xticks=xticks, yticks=yticks)

def plot_lle_ternary_diagram(
        carrier, solvent, solute, T, P=101325, thermo=None, color=None,
        tie_line_points=None,
        tie_color=None,
        N_tie_lines=5,
        N_equilibrium_grids=20,
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
        Number of tie lines. The default is 5.
    N_equilibrium_grids : int, optional
        Number of solute composition points to plot. The default is 20.

    Examples
    --------
    >>> # from thermosteam import equilibrium as eq
    >>> # eq.plot_lle_ternary_diagram('Water', 'Ethanol', 'EthylAcetate', T=298.15)
    
    .. figure:: /images/water_ethanol_ethyl_acetate_ternary_diagram.png

    """
    import ternary
    chemicals = [carrier, solute, solvent]
    thermo = as_thermo(thermo, chemicals)
    chemicals = thermo.chemicals
    IDs = chemicals.IDs
    imol = MaterialIndexer(['l', 'L'], chemicals=chemicals)
    data = imol.data
    lle = LLE(imol, thermo=thermo, method=method)
    MW = chemicals.MW
    
    def split(z):
        # Return the compositions [wt. %] of the carrier-rich and solvent-rich 
        # phases in equilibrium, or None if the feed does not split.
        data[0] = z
        data[1] = 0
        lle(T, P)
        mass = data.to_array() * MW
        total = mass.sum(1, keepdims=True)
        if (total < 1e-6 * total.sum()).any(): return None
        x, y = 100 * mass / total
        if x[0] < y[0]: x, y = y, x
        if np.abs(x - y).max() < 1: return None # Trivial split
        return x, y
    
    def crosses(line, other):
        # Whether two tie lines intersect (carrier and solute wt. % as 2D coordinates).
        (a, b), (c, d) = line, other
        def side(p, q, r): 
            return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return side(a, b, c) != side(a, b, d) and side(c, d, a) != side(c, d, b)
    
    # Trace tie lines from the carrier-solvent binary up to the plait point. 
    # Each feed is the midpoint of the last tie line shifted toward the solute,
    # which keeps it inside the two-phase region. Trivial splits and spurious 
    # splits that cross a previous tie line are discarded.
    step = 1 / N_equilibrium_grids
    pure_solute = np.array([0., 1., 0.])
    z = np.array([1 / MW[0], 0., 1 / MW[2]]) # Equal mass of carrier and solvent
    tie_line = split(z / z.sum())
    if tie_line is None: raise RuntimeError(f'{IDs[0]} and {IDs[2]} do not form two liquid phases')
    tie_lines = [tie_line]
    misses = 0
    for _ in range(10 * N_equilibrium_grids):
        x, y = tie_lines[-1]
        midpoint = (x + y) / MW
        midpoint /= midpoint.sum()
        shift = step * (misses + 1)
        tie_line = split((1 - shift) * midpoint + shift * pure_solute)
        if tie_line is None or any([crosses(tie_line, i) for i in tie_lines]):
            misses += 1
            if misses == 3: break
        else:
            tie_lines.append(tie_line)
            misses = 0
    
    # Fit a smooth binodal (solute vs carrier wt. %) through the ends of the tie lines.
    def f(x):
        shape = np.shape(x)
        X = np.ones([*shape, 5])
        X[..., 0] = x2 = x * x
        X[..., 1] = np.sqrt(x)
        X[..., 2] = x
        X[..., 3] = x2 * x
        return X
    
    ends = np.array([i for tie_line in tie_lines for i in tie_line])
    coef, *_ = lstsq(f(ends[:, 0]), ends[:, 1], rcond=None)
    xs = np.linspace(0, 100, 500)
    ys = (f(xs) * coef).sum(axis=1)
    binodal = np.clip([xs, ys, 100 - xs - ys], 0, None).T
    binodal /= binodal.sum(axis=1, keepdims=True) / 100
    fig, tax = ternary.figure(scale=100)
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
    tax.plot(binodal, c=colors.neutral_shade.RGBn, label=label)
    
    if tie_line_points is None:
        assert N_tie_lines, "must specify number of tie lines if no equilibrium points are given"
        # Evenly spaced tie lines, skipping the binary one along the boundary
        tie_lines = tie_lines[1:] or tie_lines
        index = np.linspace(0, len(tie_lines) - 1, N_tie_lines).round().astype(int)
        tie_lines = [tie_lines[i] for i in np.unique(index)]
    else:
        assert N_tie_lines is None, "cannot specify number of tie lines if equilibrium points are given"
        tie_lines = [split(np.asarray(z, float)) for z in tie_line_points]
        tie_lines = [i for i in tie_lines if i is not None]
    if tie_color is None: tie_color = colors.grey_shade.RGBn
    for x, y in tie_lines:
        tax.line(x, y, ls='--', c=tie_color)
    
    tax.ticks(axis='lbr', multiple=10, linewidth=1, offset=0.025)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()