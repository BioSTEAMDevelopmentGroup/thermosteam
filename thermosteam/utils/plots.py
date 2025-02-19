# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""

import matplotlib.pyplot as plt
from typing import Iterable
import matplotlib.ticker as ticker
import numpy as np

__all__ = (
    'set_font',
    'set_figure_size',
    'style_axis',
    'style_plot_limits',
    'fill_plot',
    'set_axes_labels',
    'set_axes_xlabels',
    'set_axes_ylabels',
)   

plt.rcParams.update({
    "figure.facecolor": (1, 1, 1, 0.95),
    "axes.facecolor": (1, 1, 1, 0.95),
    "legend.facecolor": (1, 1, 1, 0.95),
    "savefig.facecolor": (1, 1, 1, 0.95),
})

def set_font(size=8, family='sans-serif', font='Arial'):
    import matplotlib
    fontkwargs = {'size': size}
    matplotlib.rc('font', **fontkwargs)
    params = matplotlib.rcParams
    params['font.' + family] = font
    params['font.family'] = family
    
def set_figure_size(width=None, aspect_ratio=None, units=None): 
    # units default to inch
    # width defaults 6.614 inches
    # aspect ratio defaults to 0.65
    if aspect_ratio is None:
        aspect_ratio = 0.65
    if width is None or width == 'full':
        width = 6.6142
    elif width == 'half':
        width = 6.6142 / 2
    else:
        if units is not None:
            from thermosteam.units_of_measure import convert
            width = convert(width, units, 'inch')
    import matplotlib
    params = matplotlib.rcParams
    params['figure.figsize'] = (width, width * aspect_ratio)

def set_ticks(ax, ticks, which='x', ticklabels=(),
              labelrotation=0., ha=None, offset=False):
    if which == 'x':
        axis = ax.xaxis
        set_ticks = plt.xticks
        kwargs = dict(top=False)
    elif which == 'y':
        axis = ax.yaxis
        set_ticks = plt.yticks
        kwargs = dict(right=False)
    else:
        raise ValueError("which must be either 'x' or 'y'")
    if offset:
        ticks = np.array(ticks, float)
        ticks_offset = np.zeros(len(ticks) - 1)
        if len(ticks_offset) != 0:
            ticks_offset[:] = ticks[:-1] + 0.5 * np.diff(ticks)
        ax.tick_params(axis=which, which='major', length=4,
                       direction="inout", **kwargs)
        ax.tick_params(axis=which, which='minor', length=0,
                       labelrotation=labelrotation, **kwargs)
        set_ticks(ticks_offset, ())
        if ticklabels:
            axis.set_minor_locator(ticker.FixedLocator(ticks))
            axis.set_minor_formatter(ticker.FixedFormatter(ticklabels))
        ticks = ticks_offset
    else:
        set_ticks(ticks, ticklabels)
        ax.tick_params(axis=which, direction="inout", length=4, 
                       labelrotation=labelrotation, **kwargs)
    if ha is not None:
        for i in axis.get_majorticklabels():
            i.set_ha(ha)
    return ticks
    

def style_axis(ax=None, xticks=None, yticks=None, 
               xticklabels=None, yticklabels=None,
               top=True, right=True, trim_to_limits=False,
               xtick0=True, ytick0=True,
               xtickf=True, ytickf=True,
               offset_xticks=False,
               offset_yticks=False,
               xrot=None, xha=None,
               yrot=None, yha=None): # pragma: no cover
    if xticklabels is None: xticklabels = True
    if yticklabels is None: yticklabels = True
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    if xticks is None:
        xticks, xtext = plt.xticks()
    else:
        xtext = xticks
    if isinstance(xticklabels, Iterable):
        xtext = xticklabels
    elif xticklabels:
        xtext = list(xtext)
    else:
        xtext = len(xtext) * ['']
    if yticks is None:
        yticks, ytext = plt.yticks()
        if not any([str(i) for i in ytext]): ytext = yticks
    else:
        ytext = yticks
    if isinstance(yticklabels, Iterable):
        ytext = yticklabels
    elif yticklabels:
        ytext = list(ytext)
    else:
        ytext = len(ytext) * ['']
        
    xtext = list(xtext)
    ytext = list(ytext)
    if trim_to_limits:
        style_plot_limits(xticks, yticks)
    if not xtick0:
        xtext[0] = ''
    if not xtickf:
        xtext[-1] = ''
    if not ytick0:
        ytext[0] = ''
    if not ytickf:
        ytext[-1] = ''
    
    xticks = set_ticks(ax, xticks, 'x', xtext, xrot, xha, offset_xticks)
    yticks = set_ticks(ax, yticks, 'y', ytext, yrot, yha, offset_yticks)
    ax.zorder = 1
    xlim = plt.xlim()
    ylim = plt.ylim()
    axes = {'ax': ax}
    if right:
        x_twin = ax.twinx()
        ax._cached_xtwin = x_twin
        axes['twinx'] = x_twin 
        plt.sca(x_twin)
        x_twin.tick_params(axis='y', right=True, direction="in", length=4)
        x_twin.zorder = 2
        plt.ylim(ylim)
        plt.yticks(yticks, ())
    if top:
        y_twin = ax.twiny()
        ax._cached_ytwin = y_twin
        axes['twiny'] = y_twin 
        plt.sca(y_twin)
        y_twin.tick_params(axis='x', top=True, direction="in", length=4)
        y_twin.zorder = 2
        plt.xlim(xlim)
        plt.xticks(xticks, ())
    return axes
    
def style_plot_limits(xticks, yticks): # pragma: no cover
    plt.xlim([xticks[0], xticks[-1]])
    plt.ylim([yticks[0], yticks[-1]])
    
def fill_plot(color='k'): # pragma: no cover
    y_lb, y_ub = plt.ylim()
    plt.fill_between(plt.xlim(), [y_lb], [y_ub], color=color)
   
def set_axes_labels(axes, xlabel, ylabel): # pragma: no cover
    set_axes_xlabels(axes, xlabel)
    set_axes_ylabels(axes, ylabel)
   
def set_axes_xlabels(axes, xlabel): # pragma: no cover
    assert axes.ndim == 2
    N = axes.shape[1]
    if isinstance(xlabel, str): xlabel = N * [xlabel]
    for i in range(N): axes[-1, i].set_xlabel(xlabel[i])
    
def set_axes_ylabels(axes, ylabel): # pragma: no cover
    assert axes.ndim == 2
    N = axes.shape[0]
    if isinstance(ylabel, str): ylabel = N * [ylabel]
    for i in range(N): axes[i, 0].set_ylabel(ylabel[i])