# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2021, Yoel Cortes-Pena <yoelcortes@gmail.com>
# 
# This module is under the UIUC open-source license. See 
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import matplotlib.pyplot as plt

__all__ = ('style_axis',
           'style_plot_limits',
           'fill_plot',
           'set_axes_labels',
           'set_axes_xlabels',
           'set_axes_ylabels',
)   

def style_axis(ax=None, xticks=None, yticks=None, 
               xticklabels=True, yticklabels=True,
               top=True, right=True, trim_to_limits=False,
               xtick0=True, ytick0=True,
               xtickf=True, ytickf=True): # pragma: no cover
    if ax is None:
        ax = plt.gca()
    if xticks is None:
        xticks, _ = plt.xticks()
    if yticks is None:
        yticks, _ = plt.yticks()
    if trim_to_limits:
        style_plot_limits(xticks, yticks)
        if yticks[0] == 0.:
            yticks = yticks[1:]
    xtext = xticks.copy()
    ytext = yticks.copy()
    if not xtick0:
        xtext[0] = ''
    if not xtickf:
        xtext[-1] = ''
    if not ytick0:
        ytext[0] = ''
    if not ytickf:
        ytext[-1] = ''
    plt.xticks(xticks, xtext) if xticklabels else plt.xticks(xticks, ())
    plt.yticks(yticks, ytext) if yticklabels else plt.yticks(yticks, ())
    xlim = plt.xlim()
    ylim = plt.ylim()
    ax.tick_params(axis="x", top=False, direction="inout", length=4)
    ax.tick_params(axis="y", right=False, direction="inout", length=4)
    ax.zorder = 1
    
    axes = {'ax': ax}
    
    if right:
        x_twin = ax.twinx()
        axes['twinx'] = x_twin 
        plt.sca(x_twin)
        x_twin.tick_params(axis='y', right=True, direction="in", length=4)
        x_twin.zorder = 2
        plt.ylim(ylim)
        plt.yticks(yticks, ())
    if top:
        y_twin = ax.twiny()
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
    plt.fill_between(plt.xlim(), [y_lb], [y_ub], color='k')
   
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